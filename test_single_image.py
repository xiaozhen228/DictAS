import os
import cv2
import json
import torch
import logging
import argparse
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.DictAS import MyDictionary
from models.model_CLIP import Load_CLIP, tokenize
from scipy.ndimage import gaussian_filter
import copy
from models.prompt_ensemble import encode_text_with_prompt_ensemble
from models.metric_and_visualization import calcuate_metric_pixel
from models.utils import norm_patch, setup_seed, normalize, apply_ad_scoremap, cal_iou
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop

def _convert_image_to_rgb(image):
    return image.convert("RGB")
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
def _transform_test(n_px):
    return Compose([
        Resize((n_px,n_px), interpolation=BICUBIC),
        CenterCrop((n_px,n_px)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def test(args):
    img_size = args.image_size
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    features_list = args.features_list
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')

    # clip
    model_CLIP , _ , _ = Load_CLIP(img_size, args.pretrained_path, device=device)
    model_CLIP.float().to(device)
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
    transform_gt = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    
    transform_image = _transform_test(img_size)


    # ----------------------------- Modify this for different query images ----------------------------- #
    query_img_path = "./demo_example/cable/mvtec_bent_wire_000521.bmp"
    query_mask_path = "./demo_example/cable/mvtec_bent_wire_000521.png"
    # query_mask_path = None
    support_path_list = ["./demo_example/cable/normal_support_images/mvtec_000368.bmp", 
                         "./demo_example/cable/normal_support_images/mvtec_000380.bmp",
                         "./demo_example/cable/normal_support_images/mvtec_000444.bmp",
                         "./demo_example/cable/normal_support_images/mvtec_000491.bmp"]


    query_img = Image.open(query_img_path)
    h, w = query_img.size[1], query_img.size[0]
    query_img = transform_image(query_img).unsqueeze(0).to(device)
    
    if query_mask_path is not None:
        query_mask = np.array(Image.open(query_mask_path).convert('L')) > 0
        query_mask = Image.fromarray(query_mask.astype(np.uint8) * 255, mode='L')
        query_mask = transform_gt(query_mask).to(device)
        query_mask[query_mask > 0.5], query_mask[query_mask <= 0.5] = 1, 0
    else:
        query_mask = Image.fromarray(np.zeros((h, w)), mode='L')
        query_mask = transform_gt(query_mask).to(device)

    support_img_list = [transform_image(Image.open(support_path)) for support_path in support_path_list]
    support_img = torch.stack(support_img_list, dim = 0).to(device)

    with torch.no_grad():
        image_ano_features, _ , patch_ano_tokens = model_CLIP.encode_image(query_img, features_list)

        all_patch_tokens = [[] for _ in range(len(features_list))]
        if support_img.shape[0] > 4:
            img_good_chucks = torch.chunk(support_img,int(np.ceil(support_img.shape[0] / 4.0)), dim = 0)
            for img_good_chuck in img_good_chucks:
                image_good_features, _, patch_good_tokens_chunk = model_CLIP.encode_image(img_good_chuck, features_list)
                for i in range(len(patch_good_tokens_chunk)):
                    all_patch_tokens[i].append(patch_good_tokens_chunk[i])
            patch_good_tokens = [torch.cat(all_patch_token, dim = 0) for all_patch_token in all_patch_tokens]
        else:
            image_good_features, _, patch_good_tokens = model_CLIP.encode_image(support_img, features_list)
        patch_good_tokens = [norm_patch(patch_good_token, True) for patch_good_token in patch_good_tokens]

        image_ano_features = image_ano_features / image_ano_features.norm(dim = -1, keepdim = True)
        patch_ano_tokens = [norm_patch(patch_ano_token, True) for patch_ano_token in patch_ano_tokens]
        patch_ano_tokens = [Mymodel.Value_Generator(patch_ano_token) for patch_ano_token in patch_ano_tokens]

        anomaly_map_list, Retrived_list_ClS =  Mymodel(patch_ano_tokens,patch_good_tokens, mode = "test")
        anomaly_maps = []
        for i in range(len(anomaly_map_list)):
            anomaly_map = anomaly_map_list[i]
            anomaly_map = F.interpolate(anomaly_map.unsqueeze(1),
                                        size = img_size, mode = 'bilinear', align_corners=True)
            anomaly_map = anomaly_map.squeeze()
            anomaly_maps.append(anomaly_map)
        anomaly_map = torch.mean(torch.stack(anomaly_maps,dim = 0), dim = 0).cpu().numpy()
        anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigm) 
        iou = cal_iou(query_mask.cpu().numpy().ravel(), (anomaly_map.ravel()> 0.1)) 
        print(iou)

        
    def add_title(img, text, font_scale=0.8, thickness=1):
        h, w = img.shape[:2]
        bar_height = 25
        new_img = np.ones((h + bar_height, w, 3), dtype=np.uint8) * 255 
        new_img[bar_height:, :] = img

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (bar_height + text_size[1]) // 2
        cv2.putText(new_img, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return new_img


    vis_img = cv2.cvtColor(cv2.resize(cv2.imread(query_img_path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
    mask = normalize(anomaly_map)
    mask_gt = normalize(query_mask)
    mask_gt = mask_gt.squeeze().cpu().numpy()
    vis = apply_ad_scoremap(vis_img, mask)
    vis_gt = apply_ad_scoremap(vis_img, mask_gt)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
    vis_gt = cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR)
    vis_img_new = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    jian = np.ones((vis_img_new.shape[0], 10, 3),dtype=np.uint8) * 255
    prediction_mask = (anomaly_map > 0.1).astype(np.uint8)
    prediction_mask = prediction_mask * 255

    prediction_mask_3ch = cv2.cvtColor(prediction_mask, cv2.COLOR_GRAY2BGR)
    
    titles = ["Original", "Anomaly Map", "GT Map", "Binary Prediction (0.1)"]

    vis_img_new = add_title(vis_img_new, titles[0])
    vis = add_title(vis, titles[1])
    vis_gt = add_title(vis_gt, titles[2])
    prediction_mask_3ch = add_title(prediction_mask_3ch, titles[3])

    jian = np.ones((vis_img_new.shape[0], 10, 3), dtype=np.uint8) * 255

    vis_con = np.concatenate([vis_img_new, jian, vis, jian, vis_gt, jian, prediction_mask_3ch], axis=1)
    vis_con = cv2.resize(vis_con, (256*4+30, 256)).astype(np.uint8)

    save_vis = os.path.join(save_path)
    if not os.path.exists(save_vis):
        os.makedirs(save_vis)

    cv2.imwrite(os.path.join(save_vis, "result_" + os.path.basename(query_img_path).replace("bmp", "png")), vis_con)


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
    parser.add_argument("--anomaly_source_path", type=str, default="./datasets/dtd/images", help="Path to DTD dataset for anomaly synthesis")
    parser.add_argument("--save_path", type=str, default='./results/test_mvtec/222/vit_large_14_336', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default="./DictAS_weight/train_visa.pth", help='path to checkpoint')
    parser.add_argument("--config_path", type=str, default='./open_clip_local/model_configs/ViT-L-14-336.json', help="model configs")
    # model

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

    parser.add_argument("--device_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=222, help="random seed")

    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)

    move(args.save_path)
    setup_seed(args.seed)
    test(args)