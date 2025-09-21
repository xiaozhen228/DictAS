import os 
import  json 
import argparse 
import numpy as np 
import os 
import torch 
from torch import Tensor, nn 
from torch.nn import functional as F 
import torchvision.transforms as transforms 
import logging 
from models.model_CLIP import Load_CLIP, tokenize
from collections import defaultdict
from dataset import Makedataset
from tqdm import tqdm
from models.DictAS import MyDictionary
from models.EMA import EMA
from models.evaluate import evaluate_epoch
from models.utils import norm_patch, setup_seed, _transform_test
from models.prompt_ensemble import encode_text_with_prompt_ensemble
import open_clip_local


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(save_path,"result.txt")
    features_list  = args.features_list 
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    
    # We retained the OpenCLIP interface to enable DictAS to support a broader range of backbones.
    # -------------------------------------------------------------------------------------------------
    
    # Example 1 : The pretrained model from huggingface laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
    '''
    model_CLIP, _, _ = open_clip_local.create_model_and_transforms("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K", img_size= args.image_size) 
    tokenizer = open_clip_local.get_tokenizer("hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
    model_CLIP = model_CLIP.to(device)
    model_CLIP.train()
    '''

    # Example 2 : The pretrained model from OpenAI CLIP
    '''
    model_CLIP, _, _ = open_clip_local.create_model_and_transforms(args.model, pretrained= args.pretrained, img_size= args.image_size) 
    tokenizer = open_clip_local.get_tokenizer(args.model)
    model_CLIP = model_CLIP.to(device)
    model_CLIP.train()
    '''
    
    
    # -------------------------------------------------------------------------------------------------
    # This is from our own implementation of the CLIP model, which only supports the OpenAI pretrained models ViT-B-16, ViT-L-14, and ViT-L-14-336.

    model_CLIP , _ , _ = Load_CLIP(args.image_size, args.pretrained_path , device=device) 
    model_CLIP.to(device)
    tokenizer = tokenize
    model_CLIP.train()
    

    # ------------------Log-----------------------#
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')

    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    
    logger.setLevel(logging.INFO)
    file_hander  = logging.FileHandler(log_path, mode = 'w')
    file_hander.setFormatter(formatter)
    logger.addHandler(file_hander)
    console_hander = logging.StreamHandler()
    console_hander.setFormatter(formatter)
    logger.addHandler(console_hander)
    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args,arg)}')
    # ------------------Log-----------------------#


    preprocess_test = _transform_test(args.image_size)
    Make_dataset = Makedataset(train_data_path = args.train_data_path , preprocess_test = preprocess_test, mode = "train_self", 
                               image_size = args.image_size)

    Make_dataset_val = Makedataset(train_data_path = args.train_data_path , preprocess_test = preprocess_test, mode = "val", 
                               image_size = args.image_size)
    

    train_dataloader, train_obj_list = Make_dataset.make_dataset(name = args.dataset, batchsize=args.batch_size, product_list= None, shuf= True, args= args)
    if args.dataset == "mvtec":
        val_dataset_name = "visa"
        val_product_list  = ["chewinggum", "cashew", "pipe_fryum","capsules", "candle"] 
    elif args.dataset == "visa":
        val_dataset_name = "mvtec"
        val_product_list  = ["bottle", "hazelnut","cable", "metal_nut" ,"leather", "pill"] 
    else:
        val_dataset_name = "mvtec"
        val_product_list  = val_product_list  = ["bottle", "hazelnut","cable", "metal_nut" ,"leather", "pill"] 
    val_dataloader, val_obj_list = Make_dataset_val.make_dataset(name = val_dataset_name, product_list= val_product_list, batchsize = 1, shuf= False, args= args)


    Mymodel = MyDictionary(model_configs, args).to(device)
    Mymodel.train()

    ema = EMA(Mymodel, decay= 0.1)
    ema.register()

    optimizer = torch.optim.Adam(Mymodel.parameters(), lr = args.learning_rate, betas = (0.5 , 0.999)) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True)

    if args.resume_path is not None:
        resume_path = args.resume_path
        checkpoint = torch.load(resume_path, map_location= device)
        Mymodel.load_state_dict(checkpoint["Mymodel"])
        logger.info(f"loading checkpoint from {resume_path}")
    


    loss_cross = nn.CrossEntropyLoss()
    
    # text prompt
    with torch.no_grad():
        text_prompts = encode_text_with_prompt_ensemble(model_CLIP, train_obj_list, tokenizer, device)
    
    ap_max = 0
    flag = True
    for epoch in range(args.epoch):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}, Learning Rate: {current_lr}")
        
        loss_CQC_list = []
        loss_TAC_query_list = []
        loss_TAC_Retrived_list = []
        loss_query_list = []

        idx = 0
        train_bar = tqdm(train_dataloader)
        for items in train_bar:
            idx += 1
            img_ano = items['img_ano'].to(device)   # Query Image or test image
            img_good = items["img_good"].to(device) # K-shot normal support image
            gt_ano = items['img_ano_mask'].squeeze().to(device)  # GT of Query Image
            cls_name = items["cls_name"]   # Product name
            gt_ano[gt_ano > 0.5], gt_ano[gt_ano< 0.5] = 1, 0
            anomaly = items['anomaly'].long().to(device)  # 0: normal, 1: anomaly
            gt_good = items['img_good_mask'].squeeze().to(device)
            gt_good[gt_good > 0.5], gt_good[gt_good< 0.5] = 1, 0
            with torch.no_grad():
                image_ano_features, _,  patch_ano_tokens = model_CLIP.encode_image(img_ano, features_list)
                image_good_features, _,  patch_good_tokens = model_CLIP.encode_image(img_good, features_list)
                text_features = []
                for cls in cls_name:
                    text_features.append(text_prompts[cls])
                text_features = torch.stack(text_features, dim=0)

            patch_good_tokens = [norm_patch(patch_good_token, True) for patch_good_token in patch_good_tokens]
            patch_ano_tokens = [norm_patch(patch_ano_token, True) for patch_ano_token in patch_ano_tokens]

            # At the beginning of training, the Value Generator is also trained to obtain a global receptive field through global self attention, 
            # and it is frozen when the flag is set to True. 
            if not flag: 
                patch_ano_tokens = [Mymodel.Value_Generator(patch_ano_token) for patch_ano_token in patch_ano_tokens]

            B, L, C = patch_good_tokens[0].shape
            H = int(np.sqrt(L))
            gt = F.interpolate(gt_ano.unsqueeze(1), size = (H,H), mode = 'bilinear', align_corners=True)
            gt[gt > 0.5], gt[gt< 0.5] = 1, 0
            gt_mask = torch.zeros_like(gt, device=gt.device)
            gt_mask[gt == 0] = 1.0

            losses, Retrived_list_ClS = Mymodel(patch_ano_tokens, patch_good_tokens, gt_normal= gt_mask, gt_abnormal = gt)

            loss_CQC = losses[0]
            loss_query = losses[1]

            for i in range(len(args.features_list)):
                patch_ano_tokens[i] = patch_ano_tokens[i] / patch_ano_tokens[i].norm(dim = -1, keepdim = True)
                Retrived_list_ClS[i] = Retrived_list_ClS[i] / Retrived_list_ClS[i].norm(dim = -1, keepdim = True)
            
            x_r = Mymodel.Fuse_Feature(Retrived_list_ClS)
            x_q = Mymodel.Fuse_Feature(patch_ano_tokens)
            x_r = x_r / x_r.norm(dim = -1, keepdim = True) 
            x_q = x_q / x_q.norm(dim = -1, keepdim = True) 

            pro_xq = (100.0 * x_q.unsqueeze(1) @ text_features).squeeze()
            loss_query_reg = loss_cross(pro_xq, anomaly)

            pro_xr = (100.0 * x_r.unsqueeze(1) @ text_features).squeeze()
            anomaly_Retrived = anomaly * 0
            loss_Retrived_reg = loss_cross(pro_xr, anomaly_Retrived)

            loss_TAC = loss_Retrived_reg + loss_query_reg
            loss = loss_query + args.lambda1 * loss_CQC  + args.lambda2 * loss_TAC
            # --------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_query_list.append(loss_query.item())
            loss_TAC_query_list.append(loss_query_reg.item())
            loss_TAC_Retrived_list.append(loss_Retrived_reg.item())
            loss_CQC_list.append(loss_CQC.item())
            print(loss_query.item(), loss_query_reg.item(), loss_Retrived_reg.item(), loss_CQC.item())
        scheduler.step(np.mean(loss_query_list))
        
        if (epoch + 1) % args.print_freq == 0:
            ap_raw = evaluate_epoch(val_dataloader, model_CLIP, Mymodel, device, args, val_obj_list)
            logger.info('epoch [{}/{}], loss_query:{:.4f} loss_TAC_query:{:.4f}  loss_TAC_Retrived:{:.4f} loss_CQC:{:.4f}  ap:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_query_list), 
                                                                                                                                             np.mean(loss_TAC_query_list), np.mean(loss_TAC_Retrived_list), np.mean(loss_CQC_list), ap_raw))

        if ap_raw > ap_max:
            ap_max = ap_raw
            ema.save_check()
            logger.info("save best")
        else:
            logger.info("not save best")
            if flag:
                logger.info("flag")
                ema.load_check()
                '''
                After a certain number of training epochs, we freeze the Value Generator to prevent the model from taking shortcuts 
                by mapping all query and support features to zero, which would harm its generalization ability.
                '''
                for key,value in Mymodel.named_parameters(): # 
                    if "Value_Generator" in key:
                        value.requires_grad = False
                        print(key)
                    else:
                        print("0")
                optimizer.state = defaultdict(dict)
                flag = False
            

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            save_dict = {'Mymodel': Mymodel.state_dict()}
            torch.save(save_dict, ckp_path)
            



if __name__ == '__main__':

    
    parser = argparse.ArgumentParser("DictAS", add_help=True)
    # path
    parser.add_argument("--train_data_path", type=str, default="./dataset/mvisa/data", help="path to auxiliary training dataset")
    parser.add_argument("--anomaly_source_path", type=str, default="./datasets/DTD/images", help="Path to DTD dataset for anomaly synthesis")
    parser.add_argument("--save_path", type=str, default='./exps/train_visa/222/vit_large_14_336', help='path to save checkpoint')
    parser.add_argument("--config_path", type=str, default='./open_clip_local/model_configs/ViT-L-14-336.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='visa', help="train dataset name")  # mvtec, visa, MPDD, BTAD, mvtec3D, RESC, BrasTS, VOC, Ade
    parser.add_argument("--model", type=str, default="ViT-L-14-336", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="Source of pretrained weight")
    '''
    During training, we select the patch features from the 6th, 12th, 18th, and 24th layers of CLIP, while at 
    inference time these layers can be dynamically adjusted depending on the dataset. 
    In our experiments, using layers 6, 12, 18, and 24 yields the best performance for most datasets, whereas a few datasets, 
    such as MVTec-AD, achieve better results when only layers 6 and 12 are selected during inference.
    '''
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--pretrained_path", type=str, default="./pretrained_weight/ViT-L-14-336px.pt", help="Original pretrained CLIP path")
    parser.add_argument("--resume_path", type=str, default= None, help="resume_path")

    parser.add_argument("--epoch", type=int, default=30, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default= 24, help="batch size")
    parser.add_argument("--image_size", type=int, default=336, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")

    parser.add_argument("--scale_list", type=int, nargs="+", default=[1,3], help="A technique for neighborhood aggregation in feature maps")
    parser.add_argument("--gen_anomaly_rate", type=float, default=0.7, help="features used")
    parser.add_argument("--lambda1", type=float, default=0.1, help="lambda1")
    parser.add_argument("--lambda2", type=float, default=0.1, help="lambda2")

    parser.add_argument("--seed", type= int, default= 222, help="random seed")
    parser.add_argument("--device_id", type=int, default=0, help="GPU ID")
    args = parser.parse_args()
    torch.cuda.set_device(args.device_id)
    setup_seed(args.seed)
    train(args)
    