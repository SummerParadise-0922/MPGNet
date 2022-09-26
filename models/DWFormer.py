import torch
from torch import nn
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out
import math


class Global_Block(nn.Module):
    def __init__(self,dim,ratio=4) -> None:
        super(Global_Block,self).__init__()
        self.pwconv = nn.Conv2d(dim,1,kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.mlp = nn.Sequential(
            nn.Linear(dim,dim//ratio,bias=False),
            nn.ReLU(),
            nn.Linear(dim//ratio,dim,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        B,C,H,W = x.shape
        # self part
        proj = x
        # spatial prob
        spatial_pw = self.pwconv(x)
        spatial_pw = spatial_pw.reshape((B,1,-1)).permute((2,1,0))
        spatial_pw = self.softmax(spatial_pw).permute((2,1,0)).reshape((B,1,H,W))
        # matmal
        proj = proj.permute((0,2,3,1)).reshape((B,-1,C))
        spatial_pw = spatial_pw.reshape((B,1,-1))
        se_attn = torch.matmul(spatial_pw,proj).unsqueeze(-2)
        # mlp
        se_attn = self.mlp(se_attn).permute((0,3,1,2))
        return x * se_attn

class Block(nn.Module):
    def __init__(self,emb_dim,mlp_ratio,se_ratio,depth=36):
        super(Block,self).__init__()
        self.network_depth = depth
        # inter
        self.inter = nn.Sequential(
            nn.BatchNorm2d(emb_dim),
            nn.Conv2d(emb_dim,emb_dim,1),
            nn.Conv2d(emb_dim,emb_dim,kernel_size=5,stride=1,padding=2,groups=emb_dim,padding_mode='reflect'),
            nn.ReLU(True),
            Global_Block(emb_dim,se_ratio),
            nn.Conv2d(emb_dim,emb_dim,1)
        )
        # intra
        self.intra = nn.Sequential(
            nn.BatchNorm2d(emb_dim),
            nn.Conv2d(emb_dim,emb_dim*mlp_ratio,1),
            nn.ReLU(True),
            nn.Conv2d(emb_dim*mlp_ratio,emb_dim,1)
        )
        self.apply(self._init_weight)

    def _init_weight(self,m):
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = x + self.inter(x)
        x = x + self.intra(x)
        return x 



class BasicLayer(nn.Module):
    def __init__(self,layer_num,emd_dim,mlp_ratio,se_ratio) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(emd_dim,mlp_ratio,se_ratio) for _ in range(layer_num)])

    def forward(self,x):
        for blk in self.blocks:
            x = blk(x)
        return x

class DWFormer(nn.Module):
    def __init__(self,in_ch=3,emd_dims=[24,48,96,48,24],mlp_ratios=[2,4,4,2,2],layer_num = [8,8,8,6,6],se_ratio=1):
        super(DWFormer,self).__init__()
        self.downproj1 = nn.Conv2d(in_ch,emd_dims[0],kernel_size=3,stride=1,padding=1)
        self.block1 = BasicLayer(layer_num[0],emd_dims[0],mlp_ratios[0],se_ratio)

        self.downproj2 = nn.Conv2d(emd_dims[0],emd_dims[1],kernel_size=2,stride=2)
        self.block2 = BasicLayer(layer_num[1],emd_dims[1],mlp_ratios[1],se_ratio)


        self.downproj3 = nn.Conv2d(emd_dims[1],emd_dims[2],kernel_size=2,stride=2)
        self.block3 = BasicLayer(layer_num[2],emd_dims[2],mlp_ratios[2],se_ratio)


        self.upproj4 = nn.Sequential(
            nn.Conv2d(emd_dims[2],emd_dims[3]*4,1),
            nn.PixelShuffle(2)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(emd_dims[3]*2,emd_dims[3],stride=1,kernel_size=3,padding=1,padding_mode='reflect'),
            nn.ReLU()
        )
        self.block4 = BasicLayer(layer_num[3],emd_dims[3],mlp_ratios[3],se_ratio)

        self.upproj5 = nn.Sequential(
            nn.Conv2d(emd_dims[3],emd_dims[4]*4,1),
            nn.PixelShuffle(2)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(emd_dims[4]*2,emd_dims[4],stride=1,kernel_size=3,padding=1,padding_mode='reflect'),
            nn.ReLU()
        )
        self.block5 = BasicLayer(layer_num[4],emd_dims[4],mlp_ratios[4],se_ratio)

        self.out = nn.Conv2d(emd_dims[4],in_ch,kernel_size=3,stride=1,padding=1,padding_mode='reflect')

    def forward(self,x):
        skip0 = x

        x = self.downproj1(x)
        x = self.block1(x)
        skip1 = x

        x = self.downproj2(x)
        x = self.block2(x)
        skip2 = x

        x = self.downproj3(x)
        x = self.block3(x)

        x = self.upproj4(x)
        x = torch.cat([x,skip2],dim=1)
        x = self.fuse1(x)
        x = self.block4(x)

        x = self.upproj5(x)
        x = torch.cat([x,skip1],dim=1)
        x = self.fuse2(x)
        x = self.block5(x)

        x = self.out(x)
        x = x + skip0
        return x


