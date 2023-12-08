from collections import OrderedDict
from typing import Tuple, Union
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import clip
from mmaction.utils import get_root_logger
from einops import rearrange
from ..builder import BACKBONES

from torch.utils.checkpoint import checkpoint
from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import Mlp as FlashMlp

from functools import reduce, lru_cache
from operator import mul

def window_partition(x, window_size):
    """
    Args:
        x: (B, T, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    
    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_frames=8, drop_path=0.,use_flash_attn: bool = True,
                prompt=True,window_size=(32,2,2)):    
        super().__init__()
        
        
        self.use_flash_attn = use_flash_attn
        self.attn = FlashMHA(d_model, n_head, cross_attn=False, dropout=0., use_flash_attn=use_flash_attn)
        
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * 4)
        self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=QuickGELU())
        
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model, skip_connect=False)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.prompt=prompt
        
        self.window_size = window_size
        
    def forward(self, x: torch.Tensor, xt):
        
        with torch.no_grad():
            x = x + self.attn(self.ln_1(x))
            
            x = x + self.mlp(self.ln_2(x))
            
        # xt = xt + x
        # window local attention
        cls_token,windows=xt[:,:1,:],xt[:,1:,:]
        
        BT, L, C = windows.shape
        T = self.num_frames
        B = BT // self.num_frames
        H = W = int(L ** 0.5)
        window_size, _ = get_window_size((T, H, W), self.window_size, (0,0,0))
        windows = rearrange(windows, '(b t) (h w) c -> b t h w c', t=self.num_frames, h=H, w=W)
        _, Dp, Hp, Wp, _ = windows.shape
        
        shifted_win = window_partition(windows, window_size)  # B*nW, Wd*Wh*Ww, C
        
        nW=shifted_win.shape[0]//B
        nWt=T//window_size[0]
        win_p = rearrange(cls_token, '(b nWt wt) l c -> b nWt wt l c', b=B, nWt=nWt, wt=window_size[0])
        win_p = win_p.expand(-1, -1, -1, nW//nWt, -1).permute(0,1,3,2,4).reshape(-1,window_size[0], C)
        shifted_win = torch.cat([win_p, shifted_win], dim=1)
        
        shifted_attn=self.attn(self.ln_1(shifted_win))
        
        shifted_win = shifted_attn[:, window_size[0]:, :]
        win_p = shifted_win[:, :window_size[0], :] # B*nW, window_size[0] , C
        win_p = win_p.view(B, nWt, nW//nWt, -1, C) 
        win_p = rearrange(win_p, 'b nWt Wh l c -> (b nWt l) Wh c') # BT, nWh, c
        win_p=win_p.mean(dim=1)
        
        shifted_win = shifted_win.view(-1, *(window_size+(C,)))
        
        shifted_win = window_reverse(shifted_win, window_size, B, Dp, Hp, Wp) # (B, D, H, W, C)
        
        windows_attn = rearrange(shifted_win, 'b t h w c -> (b t) (h w) c')
        
        cls_token = rearrange(win_p, '(b t) n d -> (b n) t d', t=self.num_frames)
        cls_attn = self.attn(self.ln_1(cls_token))
        cls_attn = rearrange(cls_attn, '(b n) t d -> (b t) n d', n=1)
        
        xt = torch.cat([cls_attn,windows_attn],dim=1)
        xt = self.T_Adapter(xt)
        
        xt = xt + self.scale * self.S_Adapter(x)
        
        xt = xt + self.mlp(self.ln_2(xt))+ self.drop_path(self.scale * self.MLP_Adapter(xt))
        
        return x ,xt


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, scale=1., drop_path=0.1,
                use_flash_attn=True,prompt=True,window_size= (32,2,2)):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_frames=num_frames
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        
        # window_size=(8, 7, 7) if i <= 5 else (32, 2, 2)
        # window_size = (32, 2, 2) if i <= 3 else ((16, 7, 7) if 3 < i <= 7 else (4, 14, 14)),
        self.window_size = window_size
        
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i],use_flash_attn, prompt, wind_attn) for i in range(layers)])
        
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    scale,
                    num_frames,
                    dpr[i],
                    use_flash_attn,
                    prompt,
                    window_size = self.window_size
                    # window_size = (32, 2, 2) if i <= 6 else (16, 7, 7),
                )
                for i in range(layers)
            ]
        )
        
    def forward(self, x: torch.Tensor):
        xt = x
        for r in self.resblocks:
            x, xt= r(x,xt)
        
        xt = x + xt
        
        # x = rearrange(x, '(b t) n d -> b t n d',t=self.num_frames)
        # xt = rearrange(xt, '(b t) n d -> b t n d',t=self.num_frames)
        
        # out = torch.cat([x,xt],dim=1)
        
        # out = rearrange(out, 'b t n d -> (b t) n d')
        
        return xt
        # return self.resblocks(x)


@BACKBONES.register_module()
class AIM_FLASH_DUAL(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None,
                use_flash_attn=True,prompt=True,window_size= (32,2,2)):
        super().__init__()
        self.input_resolution = input_resolution
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.layers = layers
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.num_frames = num_frames
        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, width))

        self.transformer = Transformer(num_frames, width, layers, heads, scale=adapter_scale, drop_path=drop_path_rate,
                                       use_flash_attn=use_flash_attn,prompt=prompt, window_size= window_size, )

        self.ln_post = LayerNorm(width)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            ## Load OpenAI CLIP pretrained weights
            if self.layers == 12:
                clip_model, preprocess = clip.load("ViT-B/16", device="cpu")
            else:
                clip_model, preprocess = clip.load("ViT-L/14", device="cpu")
            pretrain_dict = clip_model.visual.state_dict()
            del clip_model
            del pretrain_dict['proj']
            
            swaps = [('attn.in_proj_weight', 'attn.Wqkv.weight'), ('attn.in_proj_bias', 'attn.Wqkv.bias'),
                    ('attn.out_proj.weight','attn.out_proj.weight'),('attn.out_proj.bias','attn.out_proj.bias'),
                    ('mlp.c_fc.weight','mlp.fc1.weight'),('mlp.c_fc.bias','mlp.fc1.bias'),
                    ('mlp.c_proj.weight','mlp.fc2.weight'),('mlp.c_proj.bias','mlp.fc2.bias')
                    ]
            
            out_dict={}
            for k, v in pretrain_dict.items():
                flag=True
                for sp in swaps:
                    if sp[0] in k:
                        if len(sp)==2:
                            k = k.replace(sp[0], sp[1])
                            out_dict[k] = v
                        else:
                            k2=k
                            k = k.replace(sp[0], sp[1])
                            out_dict[k] = v[:self.width]
                            k2 = k2.replace(sp[0], sp[2])
                            out_dict[k2] = v[self.width:]
                        flag=False
                if flag:
                    out_dict[k]=v
            
            msg = self.load_state_dict(out_dict, strict=False)
            
            # msg = self.load_state_dict(pretrain_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained}'")
            torch.cuda.empty_cache()
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        ## initialize S_Adapter
        for n, m in self.transformer.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize T_Adapter
        for n, m in self.transformer.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        ## initialize MLP_Adapter
        for n, m in self.transformer.named_modules():
            if 'MLP_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
        ## freeze some parameters
        for name, param in self.named_parameters():
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'Adapter' not in name:
                param.requires_grad = False
        logger = get_root_logger()
        for name, param in self.named_parameters():
            logger.info(f'{name}: {param.requires_grad}')
        num_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_total_param = sum(p.numel() for p in self.parameters())
        logger.info(
            f'Number of total parameters: {(num_total_param/1.e6):6.2f}, tunable parameters: {(num_param/1.e6):6.2f}'
        )
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed', 'temporal_embedding'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'temporal_position_bias_table'}

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        n = x.shape[1]
        x = rearrange(x, '(b t) n d -> (b n) t d', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t d -> (b t) n d', n=n)
            
        x = self.ln_pre(x)

        x = self.transformer(x)
        
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        # x = rearrange(x, '(b t) (h w) d -> b d n t',b=B)
        
        return x