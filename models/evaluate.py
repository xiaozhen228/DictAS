import torch 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score
import torch 
from torch.nn import functional as F
import numpy as np  
from sklearn.metrics import average_precision_score
import copy

def norm_patch(patches, is_forget = False):
    if is_forget:
        patches = patches[:,1:,:]
    return patches

def evaluate_epoch(val_dataloader, model_CLIP, MyModel, device, args, obj_list):
    model_CLIP.eval()
    MyModel.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['pr_sp'] = []
    results['gt_sp'] = []
    results['path'] = []
    mem_good = {}
    from tqdm import tqdm 
    for items in tqdm(val_dataloader):
        cls_name = items["cls_name"]
        img_ano = items['img_ano'].to(device)
        img_good = items["img_good"].to(device)
        img_good = img_good.squeeze(0)
        gt_ano = items['img_ano_mask'].squeeze()
        gt_ano[gt_ano > 0.5], gt_ano[gt_ano< 0.5] = 1, 0
        results['cls_names'].append(cls_name[0])
        results['imgs_masks'].append(gt_ano.numpy())
        results['gt_sp'].append(items['anomaly'].item())
        features_list = args.features_list
        with torch.no_grad():
    
            image_ano_features, _,  patch_ano_tokens = model_CLIP.encode_image(img_ano, features_list)
            if cls_name[0] not in mem_good.keys():
                image_good_features, _,  patch_good_tokens = model_CLIP.encode_image(img_good, features_list)
                mem_good[cls_name[0]] = copy.deepcopy(patch_good_tokens)
            else:
                patch_good_tokens = copy.deepcopy(mem_good[cls_name[0]])
            image_ano_features = image_ano_features / image_ano_features.norm(dim = -1, keepdim = True)

            patch_good_tokens = [norm_patch(patch_good_token, True) for patch_good_token in patch_good_tokens]
            patch_ano_tokens = [norm_patch(patch_ano_token, True) for patch_ano_token in patch_ano_tokens]
            patch_ano_tokens = [MyModel.Value_Generator(patch_ano_token) for patch_ano_token in patch_ano_tokens]

            anomaly_map_list, _ =  MyModel(patch_ano_tokens,patch_good_tokens, mode = "test")
            anomaly_maps = []
            for i in range(len(anomaly_map_list)):
                anomaly_map = anomaly_map_list[i]
                anomaly_map = F.interpolate(anomaly_map.unsqueeze(1),
                                            size = args.image_size, mode = 'bilinear', align_corners=True)
                anomaly_map = anomaly_map.squeeze()
                anomaly_maps.append(anomaly_map)
            anomaly_map = torch.mean(torch.stack(anomaly_maps,dim = 0), dim = 0).cpu().numpy()
           
            results['anomaly_maps'].append(anomaly_map)
    # metrics
    ap_px_ls = []
    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes])
                pr_px.append(results['anomaly_maps'][idxes])
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        ap_px_ls.append(ap_px)
    ap_mean = np.mean(ap_px_ls)
    model_CLIP.train()
    MyModel.train()
    del results, gt_px, pr_px
    return ap_mean 