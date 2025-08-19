import torch 
from torch import Tensor, nn 
from torch.nn import functional as F 
from models.SPM import Sparse_Lookup
import numpy as np 
from torch.nn.functional import scaled_dot_product_attention

class Global_Feature(nn.Module):
    def __init__(self, dim_i, dim_hid, dim_out, k):

        super(Global_Feature, self).__init__()
        
        self.fuse_modules = nn.Linear(dim_i * k, dim_hid)
        self.compress =  nn.Linear(dim_hid, 1)
        self.post_process = nn.Linear(dim_hid, dim_out)


    def forward(self, inps):
        x = torch.cat(inps, dim = 2)
        x = self.fuse_modules(x)
        x_temp = self.compress(x)
        attention_weights = nn.Softmax(dim=1)(x_temp) 
        x = torch.sum(attention_weights * x, dim=1)
        x = self.post_process(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, dim, proj_drop = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.drop = nn.Dropout(proj_drop)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.linear3 = nn.Linear(dim, dim, bias = False)
        #self.linear3 = nn.Linear(dim, dim)
    
    def forward(self, x, res):
        x = self.linear1(x)
        x = self.drop(x)
        x = x + res
        x = self.linear2(x)
        x = self.norm(x)
        x = self.linear3(x)
        return x





class Attn_Block(nn.Module):
    def __init__(self, dim, num_heads = 4, qkv_bias = False, qk_scale = None, proj_drop = 0.1):
        super().__init__()
        self.num_heads = num_heads 
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias = qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias = qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias = qkv_bias)
        self.norm1 = nn.LayerNorm(dim)
        self.TwoLayerMLP = MLP(dim, 0.1)

    def forward(self, fea):
        q =  k = v = self.norm1(fea)
        res = q.clone()
        B, N, C = q.shape 
        assert k.shape == v.shape 
        B, M, C = k.shape 
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        # ----------------- Way1 ------------------#
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = scaled_dot_product_attention(q, k, v, is_causal=False)  # [B, num_heads, N, head_dim]
        x = x.transpose(1, 2).reshape(B, N, C)
        # ----------------- Way1 ------------------#
        '''
        Way 1 and Way 2 are essentially the same, but scaled_dot_product_attention supports testing with more support 
        samples due to its internal optimization. This helps alleviate high GPU memory usage when the dictionary is very large.
        '''
        # ----------------- Way2 ------------------#
        # attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale
        # attn = attn.softmax(dim = -1)
        # x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)
        # ----------------- Way2 ------------------#

        x = self.TwoLayerMLP(x, res)
        return x

class GenValue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.AttnBlock = Attn_Block(dim = config["vision_cfg"]["width"])
    def forward(self, x):
        x = x + self.AttnBlock(x)
        return x

class MyDictionary(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.Query_Generator = Attn_Block(dim = config["vision_cfg"]["width"])
        self.Key_Generator = Attn_Block(dim = config["vision_cfg"]["width"])
        self.Value_Generator = GenValue(config)
        self.Fuse_Feature = Global_Feature(config["vision_cfg"]["width"], config["vision_cfg"]["width"] // 2, 
                       config["embed_dim"], k = 4)
        self.num_heads = 1
        head_dim = config["vision_cfg"]["width"] // self.num_heads
        self.scale = head_dim ** -0.5
        self.SPM = Sparse_Lookup()
        self._initialize_weights()

    
    def _initialize_weights(self):
        """Initialize the weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std= 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    
    def Padding_same(self, x, kernel_size):
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size // 2
        pad_top = (kernel_size - 1) // 2
        pad_bottom = kernel_size // 2
        padded_input = F.pad(x, pad=(pad_left, pad_right, pad_top, pad_bottom), mode= "constant")  # mode='replicate'
        padded_input = F.avg_pool2d(padded_input, kernel_size=kernel_size, stride=1, padding=0)
        return padded_input

    def Lookup(self, patch_feature_query,  patch_feature_support): 

        B, N, C = patch_feature_query.shape 
        B, M, C = patch_feature_support.shape 

        # Dictionary Construction
        F_Q = self.Query_Generator(patch_feature_query).reshape(B, N, self.num_heads, C // self.num_heads)
        F_K = self.Key_Generator(patch_feature_support).reshape(B, M, self.num_heads, C // self.num_heads)
        F_V = self.Value_Generator(patch_feature_support).reshape(B, M, self.num_heads, C // self.num_heads)

        # Dictionary Lookup
        attn = torch.einsum('bnkc,bmkc->bknm', F_Q, F_K) * self.scale
        attn = self.SPM(attn, adim = -1)
        x = torch.einsum('bknm,bmkc->bnkc', attn, F_V).reshape(B, N, C) 
        return x
    
    def forward(self, img_ano_features, img_good_features, mode = "train_self", gt_normal = None, gt_abnormal = None):
        if mode == "train_self":   # Training mode
            Retrived_list_ClS = []
            kernel_size_list = self.args.scale_list
            B, L, C = img_ano_features[0].shape
            B1, _, _ = img_good_features[0].shape
            H = int(np.sqrt(L))
            loss_CQC_all = 0
            loss_query_all = 0
            for i in range(len(img_ano_features)):
                img_ano_feature = img_ano_features[i].permute(0, 2, 1).view(B,-1,H,H)
                img_good_feature = img_good_features[i].permute(0, 2, 1).view(B,-1,H,H)
                for kernel_size in kernel_size_list:
                    if kernel_size !=1:
                        img_ano_feature_padding = self.Padding_same(img_ano_feature,  kernel_size=kernel_size)
                        img_good_feature_padding = self.Padding_same(img_good_feature,  kernel_size=kernel_size)
                    else:
                        img_ano_feature_padding = img_ano_feature.clone()
                        img_good_feature_padding = img_good_feature.clone()
                    Retrived_Result = self.Lookup(img_ano_feature_padding.reshape(B,-1, L).permute(0,2,1), img_good_feature_padding.reshape(B,-1, L).permute(0,2,1))
                    if kernel_size == 1:
                        Retrived_list_ClS.append(Retrived_Result.clone())
                    Retrived_Result = Retrived_Result.permute(0, 2, 1).view(B,-1,H,H)   

                    dis_all = 1 - F.cosine_similarity(img_ano_feature_padding, Retrived_Result, dim=1).unsqueeze(1)
                    if gt_normal is not None:
                        dis_n = dis_all * gt_normal
                    if gt_abnormal is not None:
                        dis_a = dis_all * gt_abnormal 
                    
                    loss_n = dis_n.reshape(dis_n.shape[0], -1)
                    loss_n = torch.sum(loss_n, dim = 1) / (torch.sum(gt_normal.reshape(gt_normal.shape[0], -1), dim = 1) + 1e-6)
                    loss_a = dis_a.reshape(dis_a.shape[0], -1)
                    loss_a = torch.sum(loss_a, dim = 1) / (torch.sum(gt_abnormal.reshape(gt_abnormal.shape[0], -1), dim = 1) + 1e-6)
                    loss_CQC_temp = loss_n - loss_a
                    loss_CQC_temp[loss_a == 0] = 0 
                    #print(loss_temp)
                    loss_CQC_temp[loss_CQC_temp <0] = 0  
                    #print(loss_temp)
                    loss_CQC = torch.sum(loss_CQC_temp) / (torch.sum(loss_CQC_temp!=0)+ 1e-6)
                    #print("hhh",loss_reg_con)
                    loss_query = torch.mean(loss_n)
                    loss_CQC_all = loss_CQC_all +  loss_CQC
                    loss_query_all = loss_query_all + loss_query
            return [loss_CQC_all, loss_query_all], Retrived_list_ClS
        
        else:   # Testing or eval mode
            anomaly_map_list = []
            Retrived_list_ClS = []
            kernel_size_list = self.args.scale_list
            B, L, C = img_ano_features[0].shape
            B1, _, _ = img_good_features[0].shape
            H = int(np.sqrt(L))
            for i in range(len(img_ano_features)):
                con_list = []
                img_ano_feature = img_ano_features[i].permute(0, 2, 1).view(B,-1,H,H)
                img_good_feature = img_good_features[i].permute(0, 2, 1).view(B1,-1,H,H)

                for kernel_size in kernel_size_list:
                    if kernel_size !=1:
                        img_ano_feature_padding = self.Padding_same(img_ano_feature,  kernel_size = kernel_size)
                        img_good_feature_padding = self.Padding_same(img_good_feature,  kernel_size=kernel_size)
                    else:
                        img_ano_feature_padding = img_ano_feature.clone()
                        img_good_feature_padding = img_good_feature.clone()
                    
                    img_good_feature_padding = img_good_feature_padding.reshape(B1,-1, L).permute(0,2,1).reshape(-1, C).unsqueeze(0)
                    Retrived_Result = self.Lookup(img_ano_feature_padding.reshape(B,-1, L).permute(0,2,1), img_good_feature_padding)
                    if kernel_size == 1:
                        Retrived_list_ClS.append(Retrived_Result.clone())
                    Retrived_Result = Retrived_Result.permute(0, 2, 1).view(B,-1,H,H)
                    con = F.cosine_similarity(img_ano_feature_padding, Retrived_Result, dim=1)
                    con_list.append(con)
                anomaly_map_list.extend(con_list)
            for i in range(len(anomaly_map_list)):
                anomaly_map = anomaly_map_list[i]
                anomaly_map_list[i] = (1 - (anomaly_map + 1)*0.5)
            return anomaly_map_list, Retrived_list_ClS
