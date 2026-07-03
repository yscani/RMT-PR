import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../mambapy/')
import time
import torch
import torch.nn as nn

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F
from tools.read_samples import read_one_need_from_seq
import yaml
from mambapy.mamba import Mamba, MambaConfig
from mambapy.mamba_lm import MambaLMConfig, MambaLM
import math


###############
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


################


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, (1, 3), 1, p=(0, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding= p,groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module): 
    # Spatial Pyramid Pooling 
    def __init__(self, c1=128, c2=128, k=(1, 5)):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, (1, 1), (1, 1))
        self.cv2 = Conv(c_ * 4, c2, (1, 1), (1, 1))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=(0, 2))

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=(64, 900), patch_size=(16, 4), stride=(16, 4), in_chans=1, embed_dim=128, norm_layer=None,
                 flatten=True):
        super().__init__()
        self.img_size = img_size  # (64, 900)
        self.patch_size = patch_size  # (16, 4)
        self.grid_size = (
        (img_size[0] - patch_size[0]) // stride[0] + 1, (img_size[1] - patch_size[1]) // stride[0] + 1)  # (4, 225)
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 900
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)  # [1, 1, 64, 900] -> [1, 128, 4, 255]
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape  # [1, 1, 64, 900]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # [1, 256, 4, 255]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC [1, 900, 256]
        x = self.norm(x)
        return x

class featureExtracter(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer=False, use_mamba=True,
                 use_patch_embed=False, use_mamba_lm=False, use_conv=False):
        super(featureExtracter, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer
        self.use_mamba = use_mamba
        self.ues_patch_embed = use_patch_embed
        self.use_mamba_lm = use_mamba_lm
        self.use_conv = use_conv

        if not self.ues_patch_embed:
            if not self.use_conv:    # do
                self.conv1 = nn.Conv2d(1, 8, kernel_size=(2,1), stride=(2,1), bias=False)
                self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,1), stride=(1,1), bias=False)
                self.conv3 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(1,1), bias=False)
                self.conv4 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(1,1), bias=False)
                self.conv5 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(1,1), bias=False)
                self.conv6 = nn.Conv2d(64, 128, kernel_size=(3,1), stride=(1,1), bias=False)
                self.conv7 = nn.Conv2d(128, 128, kernel_size=(3,1), stride=(1,1), bias=False)
                self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
                
            else:
                self.conv1 = Conv(channels, 16, k=(6, 1), s=(2, 1), p=(2, 0))
                self.conv2 = Conv(16, 32, k=(3, 1), s=(2, 1))  # 16
                self.c3_1 = C3(32, 32, 3)
                self.conv3 = Conv(32, 64, k=(3, 1), s=(2, 1))
                self.c3_2 = C3(64, 64, 3)
                self.conv4 = Conv(64, 128, k=(3, 1), s=(2, 1))
                self.c3_3 = C3(128, 128, 3)
                self.conv5 = Conv(128, 128, k=(3, 1), s=(2, 1))
                self.c3_4 = C3(128, 128, 3)
                self.sppf = SPPF(128)
        else:
            self.patch_embed = PatchEmbed()

        self.relu = nn.ReLU(inplace=True)

        """
            MHSA
        """
        if self.use_mamba:
            if self. use_mamba_lm:
                config = MambaLMConfig(d_model=256, n_layers=1, vocab_size=1000)
                self.mamba_encoder = MambaLM(config)

            else:
                config = MambaConfig(d_model=256, n_layers=1)    # do
                self.mamba_encoder = Mamba(config)


        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1), bias=False)         #
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128*900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """

        ### 
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=3600, cluster_size=32,
                                     output_dim=512, gating=True, add_batch_norm=False,
                                     is_training=True)
        
        self.net_vlad_ri = NetVLADLoupe(feature_size=512, max_samples=1800, cluster_size=32,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.net_vlad_bev = NetVLADLoupe(feature_size=512, max_samples=1800, cluster_size=32,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        ##################
        d_model = 512     #
        heads = 4
        dropout = 0.
        self.norm_1 = Norm(d_model)
        self.attn1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff1 = FeedForward(d_model, dropout=dropout)



        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

        #####
        self.w_range   = nn.Parameter(torch.tensor(1.0))
        self.w_bev  = nn.Parameter(torch.tensor(1.0))
        self.w_cross = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_l, x_l_trans):
        
        #####
        x_l_range=x_l[:,0,:,:]
        x_l_range = x_l_range.unsqueeze(1)
        x_l_BEV=x_l[:,5,:,:]
        x_l_BEV = x_l_BEV.unsqueeze(1)

        x_l_range_trans=x_l_trans[:,0,:,:]
        x_l_range_trans = x_l_range_trans.unsqueeze(1)
        x_l_BEV_trans=x_l_trans[:,5,:,:]
        x_l_BEV_trans = x_l_BEV_trans.unsqueeze(1)

        #####range

        if self.ues_patch_embed:
            out_l = self.patch_embed(x_l_range)
            out_l = out_l.unsqueeze(2).permute(0, 3, 2, 1)

        elif self.use_conv:
            out_l = self.conv1(x_l_range)
            out_l = self.conv2(out_l)
            out_l = self.c3_1(out_l)
            out_l = self.conv3(out_l)
            out_l = self.c3_2(out_l)
            out_l = self.conv4(out_l)
            out_l = self.c3_3(out_l)
            out_l = self.conv5(out_l)
            out_l = self.c3_4(out_l)
            out_l = self.sppf(out_l)

        else:   #do 
            out_l = self.relu(self.conv1(x_l_range))       
            out_l = self.relu(self.conv2(out_l))   
            out_l = self.relu(self.conv3(out_l))   
            out_l = self.relu(self.conv4(out_l))    
            out_l = self.relu(self.conv5(out_l))    
            out_l = self.relu(self.conv6(out_l))   
            out_l = self.relu(self.conv7(out_l))           
            out_l = self.relu(self.conv8(out_l))                        
            
        out_l_1 = out_l.permute(0, 1, 3, 2)  
        out_l_1 = self.relu(self.convLast1(out_l_1))  

        """Using transformer needs to decide whether batch_size first"""
        if self.use_transformer:
            out_l = out_l_1.squeeze(3)  
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)

            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        elif self.use_mamba:
            out_l = out_l_1.squeeze(3)          
            out_l = out_l.permute(0, 2, 1)      
            out_l = self.mamba_encoder(out_l)   
            out_l = out_l.permute(0, 2, 1)      
            out_l = out_l.unsqueeze(3)           

            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))    

            out_l_range = out_l
            
        else:
            out_l = torch.cat((out_l_1, out_l_1), dim=1)
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        ######### bev 
        if self.ues_patch_embed:
            out_l = self.patch_embed(x_l_BEV)
            out_l = out_l.unsqueeze(2).permute(0, 3, 2, 1)

        elif self.use_conv:
            out_l = self.conv1(x_l_BEV)
            out_l = self.conv2(out_l)
            out_l = self.c3_1(out_l)
            out_l = self.conv3(out_l)
            out_l = self.c3_2(out_l)
            out_l = self.conv4(out_l)
            out_l = self.c3_3(out_l)
            out_l = self.conv5(out_l)
            out_l = self.c3_4(out_l)
            out_l = self.sppf(out_l)

        else:   

            out_l = self.relu(self.conv1(x_l_BEV))       
            out_l = self.relu(self.conv2(out_l))   
            out_l = self.relu(self.conv3(out_l))    
            out_l = self.relu(self.conv4(out_l))   
            out_l = self.relu(self.conv5(out_l))   
            out_l = self.relu(self.conv6(out_l))   
            out_l = self.relu(self.conv7(out_l))   
            out_l = self.relu(self.conv8(out_l))                               
           

        out_l_1 = out_l.permute(0, 1, 3, 2)  
        out_l_1 = self.relu(self.convLast1(out_l_1))  

        """Using transformer needs to decide whether batch_size first"""
        if self.use_transformer:
            out_l = out_l_1.squeeze(3)  
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)

            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        elif self.use_mamba:
            out_l = out_l_1.squeeze(3)          
            out_l = out_l.permute(0, 2, 1)      
            out_l = self.mamba_encoder(out_l)   
            out_l = out_l.permute(0, 2, 1)      
            out_l = out_l.unsqueeze(3)          


            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))   

            out_l_BEV = out_l

            
        else:
            out_l = torch.cat((out_l_1, out_l_1), dim=1)
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

         #####range_trans

        if self.ues_patch_embed:
            out_l = self.patch_embed(x_l_range_trans)
            out_l = out_l.unsqueeze(2).permute(0, 3, 2, 1)

        elif self.use_conv:
            out_l = self.conv1(x_l_range_trans)
            out_l = self.conv2(out_l)
            out_l = self.c3_1(out_l)
            out_l = self.conv3(out_l)
            out_l = self.c3_2(out_l)
            out_l = self.conv4(out_l)
            out_l = self.c3_3(out_l)
            out_l = self.conv5(out_l)
            out_l = self.c3_4(out_l)
            out_l = self.sppf(out_l)

        else:   

            out_l = self.relu(self.conv1(x_l_range_trans))       
            out_l = self.relu(self.conv2(out_l))   
            out_l = self.relu(self.conv3(out_l))    
            out_l = self.relu(self.conv4(out_l))   
            out_l = self.relu(self.conv5(out_l))   
            out_l = self.relu(self.conv6(out_l))   
            out_l = self.relu(self.conv7(out_l))          
            out_l = self.relu(self.conv8(out_l))                        
            

        out_l_1 = out_l.permute(0, 1, 3, 2)  
        out_l_1 = self.relu(self.convLast1(out_l_1))  

        """Using transformer needs to decide whether batch_size first"""
        if self.use_transformer:
            out_l = out_l_1.squeeze(3)  
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)

            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        elif self.use_mamba:
            out_l = out_l_1.squeeze(3)          
            out_l = out_l.permute(0, 2, 1)      
            out_l = self.mamba_encoder(out_l)   
            out_l = out_l.permute(0, 2, 1)      
            out_l = out_l.unsqueeze(3)           


            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))    

            out_l_range_trans = out_l

            
        else:
            out_l = torch.cat((out_l_1, out_l_1), dim=1)
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        ######### bev_trans 
        if self.ues_patch_embed:
            out_l = self.patch_embed(x_l_BEV_trans)
            out_l = out_l.unsqueeze(2).permute(0, 3, 2, 1)

        elif self.use_conv:
            out_l = self.conv1(x_l_BEV_trans)
            out_l = self.conv2(out_l)
            out_l = self.c3_1(out_l)
            out_l = self.conv3(out_l)
            out_l = self.c3_2(out_l)
            out_l = self.conv4(out_l)
            out_l = self.c3_3(out_l)
            out_l = self.conv5(out_l)
            out_l = self.c3_4(out_l)
            out_l = self.sppf(out_l)

        else:   

            out_l = self.relu(self.conv1(x_l_BEV_trans))      
            out_l = self.relu(self.conv2(out_l))    
            out_l = self.relu(self.conv3(out_l))    
            out_l = self.relu(self.conv4(out_l))    
            out_l = self.relu(self.conv5(out_l))    
            out_l = self.relu(self.conv6(out_l))    
            out_l = self.relu(self.conv7(out_l))    
            out_l = self.relu(self.conv8(out_l))                               


        out_l_1 = out_l.permute(0, 1, 3, 2)  
        out_l_1 = self.relu(self.convLast1(out_l_1))  

        """Using transformer needs to decide whether batch_size first"""
        if self.use_transformer:
            out_l = out_l_1.squeeze(3) 
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)

            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        elif self.use_mamba:
            out_l = out_l_1.squeeze(3)          
            out_l = out_l.permute(0, 2, 1)      
            out_l = self.mamba_encoder(out_l)   
            out_l = out_l.permute(0, 2, 1)      #
            out_l = out_l.unsqueeze(3)          


            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))    

            out_l_BEV_trans = out_l


        else:
            out_l = torch.cat((out_l_1, out_l_1), dim=1)
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)



        ########### head 


        out_l_range = torch.cat((out_l_range, out_l_range_trans), dim=2)   
        out_l_BEV = torch.cat((out_l_BEV, out_l_BEV_trans), dim=2)   

        
        feature_ri = out_l_range.squeeze(-1)                       
        feature_bev = out_l_BEV.squeeze(-1)                     
        feature_ri = feature_ri.permute(0, 2, 1)                    
        feature_bev = feature_bev.permute(0, 2, 1)                  #

        feature_ri = F.normalize(feature_ri, dim=-1)                
        feature_bev = F.normalize(feature_bev, dim=-1)             

        feature_ri = self.norm_1(feature_ri)            
        feature_bev = self.norm_1(feature_bev)         

        feature_fuse1 = feature_bev + self.attn1(feature_bev, feature_ri, feature_ri, mask=None)  
        feature_fuse1 = self.norm_1(feature_fuse1)    
        feature_fuse1 = feature_fuse1 + self.ff1(feature_fuse1)     

        feature_fuse2 = feature_ri + self.attn1(feature_ri, feature_bev, feature_bev, mask=None)     
        feature_fuse2 = self.norm_1(feature_fuse2)                                                   
        feature_fuse2 = feature_fuse2 + self.ff1(feature_fuse2)     

        feature_fuse1_ext = feature_fuse1 + self.attn1(feature_fuse1, feature_ri, feature_ri, mask=None)       
        feature_fuse1_ext = self.norm_1(feature_fuse1_ext)                      
        feature_fuse1_ext = feature_fuse1_ext + self.ff1(feature_fuse1_ext)    

        feature_fuse2_ext = feature_fuse2 + self.attn1(feature_fuse2, feature_bev, feature_bev, mask=None)   
        feature_fuse2_ext = self.norm_1(feature_fuse2_ext)                      
        feature_fuse2_ext = feature_fuse2_ext + self.ff1(feature_fuse2_ext)   

        feature_fuse = torch.cat((feature_fuse1_ext, feature_fuse2_ext), dim=-2)   
        feature_cat_origin = torch.cat((feature_bev, feature_ri), dim=-2)          
        feature_fuse = torch.cat((feature_fuse, feature_cat_origin), dim=-1)       

        feature_fuse = feature_fuse.permute(0, 2, 1)  
        feature_com = feature_fuse.unsqueeze(3)        
        feature_com = F.normalize(feature_com, dim=1)   
        feature_com = self.net_vlad(feature_com)           
        feature_com = F.normalize(feature_com, dim=1)      

        feature_ri = feature_ri.permute(0, 2, 1)                           
        feature_ri = feature_ri.unsqueeze(-1)                               
        feature_ri_enhanced = self.net_vlad_ri(feature_ri)               
        feature_ri_enhanced = F.normalize(feature_ri_enhanced, dim=1)       

        feature_bev = feature_bev.permute(0, 2, 1)             
        feature_bev = feature_bev.unsqueeze(-1)                
        feature_bev_enhanced = self.net_vlad_bev(feature_bev)          
        feature_bev_enhanced = F.normalize(feature_bev_enhanced, dim=1)                     

        ######
        feature_ri_enhanced   = self.w_range   * feature_ri_enhanced
        feature_bev_enhanced  = self.w_bev  * feature_bev_enhanced
        feature_com           = self.w_cross * feature_com  
        
        feature_com = torch.cat((feature_ri_enhanced, feature_com), dim=1)                  
        feature_com = torch.cat((feature_com, feature_bev_enhanced), dim=1)                 

        return feature_com



