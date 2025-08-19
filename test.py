import os
import cv2
import json
import torch
import logging
import argparse
import numpy as np
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from dataset import MyDataset
from tqdm import tqdm
from models.DictAS import MyDictionary
from models.model_CLIP import Load_CLIP, tokenize
from scipy.ndimage import gaussian_filter
import copy
from models.prompt_ensemble import encode_text_with_prompt_ensemble
from models.metric_and_visualization import calcuate_metric_pixel, calcuate_metric_image
from models.utils import norm_patch, setup_seed, normalize, apply_ad_scoremap, cal_iou, BESTSEGMENTATION
import open_clip_local



def test(args):
    img_size = args.image_size
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')

    # We retained the OpenCLIP interface to enable DictAS to support a broader range of backbones.
    # -------------------------------------------------------------------------------------------------
    
    # Example 1 : The pretrained model from huggingface laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
    '''
    model_CLIP, _, _ = open_clip_local.create_model_and_transforms("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", img_size= args.image_size) 
    tokenizer = open_clip_local.get_tokenizer("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
    model_CLIP = model_CLIP.to(device)
    model_CLIP.eval()
    '''

    # Example 2 : The pretrained model from OpenAI CLIP
    '''
    model_CLIP, _, _ = open_clip_local.create_model_and_transforms(args.model, pretrained= args.pretrained, img_size= args.image_size) 
    tokenizer = open_clip_local.get_tokenizer(args.model)
    model_CLIP = model_CLIP.to(device)
    model_CLIP.eval()
    '''
    
    
    # -------------------------------------------------------------------------------------------------
    # This is from our own implementation of the CLIP model, which only supports the OpenAI pretrained models ViT-B-16, ViT-L-14, and ViT-L-14-336.
    model_CLIP , preprocess_train , preprocess_test = Load_CLIP(img_size, args.pretrained_path, device=device)
    model_CLIP.to(device)
    model_CLIP.eval()
    tokenizer = tokenize

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    
    Mymodel = MyDictionary(model_configs, args).to(device)
    Mymodel.eval()

    checkpoint = torch.load(args.checkpoint_path, map_location= device)
    Mymodel.load_state_dict(checkpoint["Mymodel"])

    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    test_data = MyDataset(root=dataset_dir, transform=preprocess_test, target_transform=transform,
                                mode='test', k_shot = args.k_shot, args= args, dataset= dataset_name)
    
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.get_cls_names()

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['pr_sp'] = []
    results['gt_sp'] = []
    results['path'] = []

    
    idx = 0

    
    # text prompt
    with torch.no_grad():
        obj_list = test_data.get_cls_names()
        text_prompts = encode_text_with_prompt_ensemble(model_CLIP, obj_list, tokenizer, device)

    mem_good = {}
    from tqdm import tqdm 
    for items in tqdm(test_dataloader):
        idx += 1
        cls_name = items["cls_name"]
        img_ano = items['img_ano'].to(device)
        img_good = items["img_good"].to(device)
        img_good = img_good.squeeze(0)
        gt_ano = items['img_ano_mask'].squeeze().to(device)
        gt_ano[gt_ano > 0.5], gt_ano[gt_ano< 0.5] = 1, 0


        results['cls_names'].append(cls_name[0])
        results['imgs_masks'].append(gt_ano)
        results['gt_sp'].append(items['anomaly'].item())

        features_list = BESTSEGMENTATION(args, cls_name)
        with torch.no_grad():
            text_features = []
            for cls in cls_name:
                text_features.append(text_prompts[cls])
            text_features = torch.stack(text_features, dim=0).float()
            

            image_ano_features, _,  patch_ano_tokens = model_CLIP.encode_image(img_ano, features_list)
            if cls_name[0] not in mem_good.keys():
                mem_good = {}
                all_patch_tokens = [[] for _ in range(len(features_list))]
                if img_good.shape[0] > 4:
                    img_good_chucks = torch.chunk(img_good,int(np.ceil(img_good.shape[0] / 4.0)), dim = 0)
                    for img_good_chuck in img_good_chucks:
                        image_good_features, _ ,  patch_good_tokens_chunk = model_CLIP.encode_image(img_good_chuck, features_list)
                        for i in range(len(patch_good_tokens_chunk)):
                            all_patch_tokens[i].append(patch_good_tokens_chunk[i])
                    patch_good_tokens = [torch.cat(all_patch_token, dim = 0) for all_patch_token in all_patch_tokens]
                else:
                    image_good_features, _ ,  patch_good_tokens = model_CLIP.encode_image(img_good, features_list)
                patch_good_tokens = [norm_patch(patch_good_token, True) for patch_good_token in patch_good_tokens]
                mem_good[cls_name[0]] = copy.deepcopy(patch_good_tokens)
            else:
                patch_good_tokens = copy.deepcopy(mem_good[cls_name[0]])
            image_ano_features = image_ano_features / image_ano_features.norm(dim = -1, keepdim = True)
            patch_ano_tokens = [norm_patch(patch_ano_token, True) for patch_ano_token in patch_ano_tokens]
            patch_ano_tokens = [Mymodel.Value_Generator(patch_ano_token) for patch_ano_token in patch_ano_tokens]

            anomaly_map_list, Retrived_list_ClS =  Mymodel(patch_ano_tokens,patch_good_tokens, mode = "test")

            pro_img_query = (100.0 * image_ano_features.unsqueeze(1) @ text_features).softmax(dim=-1).squeeze()
            results['pr_sp'].append(pro_img_query[1].cpu().item())

            anomaly_maps = []
            for i in range(len(anomaly_map_list)):
                anomaly_map = anomaly_map_list[i]
                anomaly_map = F.interpolate(anomaly_map.unsqueeze(1),
                                            size = img_size, mode = 'bilinear', align_corners=True)
                anomaly_map = anomaly_map.squeeze()
                anomaly_maps.append(anomaly_map)
            anomaly_map = torch.mean(torch.stack(anomaly_maps,dim = 0), dim = 0).cpu().numpy()
           
            
            results['anomaly_maps'].append(anomaly_map)
            anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigm)  
            
            path = items['img_ano_path']
            results['path'].extend(path)
            
            # visualization
            '''
            path = items['img_ano_path']
            cls_anomaly = path[0].split('/')[-2]
            filename = path[0].split('/')[-1]
            vis_img = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
            mask = normalize(anomaly_map)
            mask_gt = normalize(gt_ano)
            mask_gt = mask_gt.squeeze().cpu().numpy()
            vis = apply_ad_scoremap(vis_img, mask)
            vis_gt = apply_ad_scoremap(vis_img, mask_gt)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            vis_gt = cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR)
            vis_img_new = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            jian = np.ones((vis_img_new.shape[0], 10, 3),dtype=np.uint8) * 255
            vis_con = np.concatenate([vis_img_new,jian, vis, jian, vis_gt], axis = 1)
            vis_con = cv2.resize(vis_con, (256*3+20, 256)).astype(np.uint8)
            save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls_anomaly)
            if not os.path.exists(save_vis):
                os.makedirs(save_vis)
            cv2.imwrite(os.path.join(save_vis, filename.replace("bmp", "png")), vis_con)
            # visualization

            '''
    
    calcuate_metric_pixel(results, obj_list, logger, alpha = args.alpha , sigm = args.sigm, args = args)


import shutil
def move(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("DictAS", add_help=True)
    parser.add_argument("--data_path", type=str, default="./dataset/mvisa/data", help="path to test dataset")
    parser.add_argument("--anomaly_source_path", type=str, default="./datasets/DTD/images", help="Path to DTD dataset for anomaly synthesis")
    parser.add_argument("--save_path", type=str, default='./results/test_mvtec/222/vit_large_14_336', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default="./DictAS_weight/train_visa.pth", help='path to checkpoint')
    parser.add_argument("--config_path", type=str, default='./open_clip_local/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")  # mvtec, visa, MPDD, BTAD, mvtec3D, RESC, BrasTS
    parser.add_argument("--image_size", type=int, default= 336, help="image size")
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="Source of pretrained weight")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt", help="Original pretrained CLIP path")

    parser.add_argument('--TEST_For_BESTSEGMENTATION', type=lambda x: x.lower() == 'true',
                        default=True, choices=[True, False], help= "True for the best segmentation performance, and False for the best classification performance.")
    parser.add_argument("--scale_list", type=int, nargs="+", default=[1,3], help= "A technique for neighborhood aggregation in feature maps")
    parser.add_argument("--alpha", type=float, default= 0.5, help="text classification weight")
    parser.add_argument("--sigm", type=int, default= 6, help="Gaussian kernel size for anomaly map post-processing")

    # few shot
    parser.add_argument("--k_shot", type=int, default= 4, help="e.g., 1-shot, 2-shot, 4-shot")
    
    parser.add_argument("--device_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=222, help="random seed")

    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)

    move(args.save_path)
    setup_seed(args.seed)
    test(args)