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

class PatchShift(nn.Module):
    def __init__(self, inv=False, ratio=1):
        super(PatchShift, self).__init__()

        self.inv = inv

        if inv:
            print('=> Using inverse PatchShift,   tps')
        else:
            print('=> Using PatchShift,   tps')

    def forward(self, x):
        x = self.shift(x, inv=self.inv)
        return x #self.net(x)

    @staticmethod
    def shift(x, inv=False):
        B, T , H, W, c = x.size()
        feat = x
        # feat = feat.view(B, T,  H,  W, c)
        out = feat.clone()
        stride = 1
        multiplier = -1 if inv else 1
        ## Pattern C
        out[:, :, 0::3, 0::3,:] = torch.roll(feat[:, :,  0::3,0::3,:], shifts=-4*multiplier*stride, dims=1)
        out[:, :, 0::3, 1::3,:] = torch.roll(feat[:, :,  0::3,1::3,:], shifts=multiplier*stride, dims=1)
        out[:, :, 1::3, 0::3,:] = torch.roll(feat[:, :,  1::3,0::3,:], shifts=-multiplier*stride, dims=1)
        out[:, :, 0::3, 2::3,:] = torch.roll(feat[:, :,  0::3,2::3,:], shifts=2*multiplier*stride, dims=1)
        out[:, :, 2::3, 0::3,:] = torch.roll(feat[:, :,  2::3,0::3,:], shifts=-2*multiplier*stride, dims=1)
        out[:, :, 1::3, 2::3,:] = torch.roll(feat[:, :,  1::3,2::3,:], shifts=3*multiplier*stride, dims=1)
        out[:, :, 2::3, 1::3,:] = torch.roll(feat[:, :,  2::3,1::3,:], shifts=-3*multiplier*stride, dims=1)
        out[:, :, 2::3, 2::3,:] = torch.roll(feat[:, :,  2::3,2::3,:], shifts=4*multiplier*stride, dims=1) 

        # out = out.view(B, T, H, W, c)
        return out

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
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_frames=8, drop_path=0.,
                shift: bool = False, 
                shift_type: str = 'psm',
                use_flash_attn: bool = True):
        super().__init__()
        
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.use_flash_attn = use_flash_attn
        # if shift:
        #     self.attn = FlashMHA(d_model, n_head, cross_attn=True, dropout=0., use_flash_attn=use_flash_attn)
        # else:
        #     self.attn = FlashMHA(d_model, n_head, cross_attn=False, dropout=0., use_flash_attn=use_flash_attn)
        self.attn = FlashMHA(d_model, n_head, cross_attn=True, dropout=0., use_flash_attn=use_flash_attn)
        
        self.ln_1 = LayerNorm(d_model)
        
        mlp_width = int(d_model * 4)
        self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=QuickGELU())
        
        # self.mlp = nn.Sequential(OrderedDict([
        #     ("c_fc", nn.Linear(d_model, d_model * 4)),
        #     ("gelu", QuickGELU()),
        #     ("c_proj", nn.Linear(d_model * 4, d_model))
        # ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        self.d_model = d_model
        
        # self.avgpool= nn.AdaptiveAvgPool1d(1)

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        # self.S_Adapter = Adapter(d_model)
        self.S_Adapter = Adapter(d_model,skip_connect=False)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        # if num_tadapter == 2:
        #     self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.shift = shift
        self.shift_type = shift_type
        
        if self.shift:
            self.shift_op=PatchShift( inv=False)
            
        self.prompt_weight=nn.Parameter(torch.zeros(1))
        # self.prompt_weight=3e-3
    
    def forward(self, x: torch.Tensor, x_res: torch.Tensor):
        
        ## x shape [ BT, HW+1, D]
        bt, n, d = x.shape
        ## temporal adaptation
        
        class_token=x[:, :1, :] # BT, 1, D
        
        xt = rearrange(class_token, '(b t) n d -> (b n) t d', t=self.num_frames)
        
        ln_xt=self.ln_1(xt)
        
        # if self.use_flash_attn:
        # if self.shift:
        #     xt = self.T_Adapter(self.attn(x=ln_xt,x_kv=ln_xt))
        # else:
        #     xt = self.T_Adapter(self.attn(ln_xt))
        xt = self.T_Adapter(self.attn(x=ln_xt,x_kv=ln_xt))
        
        xt = rearrange(xt, '(b n) t d -> (b t) n d', n=1)
        # x = x + self.drop_path(xt)
        # x= torch.cat([x, xt], dim=0)
        
        x= torch.cat([x[:, :1, :], xt, x[:, 1:, :]], dim=1)
        
        ## prompt tuning
        if self.shift:
            xln=self.ln_1(x)
            tmp_x=xln[ :, 2:, :].clone()
            NT, L,  C = tmp_x.shape
            T = self.num_frames
            N = NT // self.num_frames
            H = W = int(L**0.5)
            tmp_x = rearrange(tmp_x, '(b t) (h w) c -> b t h w c', b=N, t = T, h=H, w=W, c=C)
            tmp_x = self.shift_op(tmp_x)
            tmp_x = rearrange(tmp_x, 'b t h w c -> (b t) c h w')
            tmp_x = tmp_x.view(NT, C, -1).permute(0, 2, 1).contiguous() # P NT C
            # tmp_x = torch.cat([xln, tmp_x], dim=1)
            
            x = x + self.attn(x=xln,x_kv=xln) + self.drop_path(self.scale * self.S_Adapter(x))
            
            x_temporal = x + self.drop_path( self.scale * self.S_Adapter(self.attn(x=self.ln_1(x),x_kv=self.ln_1(tmp_x)))) + x_res
            
            # x = x + self.S_Adapter(self.attn(x=xln,x_kv=tmp_x)) 
        else:
            ## spatial adaptation
            # x = x + self.drop_path(self.S_Adapter(self.attn(self.ln_1(x))))
            x = x + self.attn(self.ln_1(x)) + self.drop_path(self.scale * self.S_Adapter(x))
            # x = x + (1 - self.prompt_weight) * self.attn(x=self.ln_1(x), x_kv=self.ln_1(x)) + self.drop_path(self.scale * self.S_Adapter(x)) + self.drop_path( self.prompt_weight * self.attn(x=self.ln_1(x), x_kv=xt))
        
        ## joint adaptation
        # x=x[:-1,:,:]
        
        x= torch.cat([x[:, :1, :], x[:, 2:, :]], dim=1)
        
        xn = self.ln_2(x)
        x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        # x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(x))
        
        return x , x_temporal


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, scale=1., drop_path=0.1,
                shift=True,shift_type='psm',checkpoint=False,use_flash_attn=True):
        super().__init__()
        self.width = width
        self.layers = layers
        self.checkpoint=checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, scale, num_tadapter, num_frames, dpr[i]) for i in range(layers)])
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    scale,
                    num_frames,
                    dpr[i],
                    shift = shift,
                    shift_type=shift_type,
                    use_flash_attn=use_flash_attn
                )
                for i in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        
        x_res = 0.0
        for r in self.resblocks:
            if self.checkpoint and not torch.jit.is_scripting():
                x, x_res = checkpoint(r, x, x_res)
            else:
                x, x_res = r(x, x_res)
        return x
        
        # return self.resblocks(x)


@BACKBONES.register_module()
class ViT_CLIP_FLASH_RES_TUNING(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, adapter_scale=0.5,
                pretrained=None,shift=False,checkpoint=False,use_flash_attn=True):
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
        
        self.shift=shift
        self.use_flash_attn = use_flash_attn
        self.width=width

        self.transformer = Transformer(num_frames, width, layers, heads, scale=adapter_scale, drop_path=drop_path_rate,shift=shift,checkpoint=checkpoint,use_flash_attn=use_flash_attn)

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
            
            if not self.shift:
                swaps = [('attn.in_proj_weight', 'attn.Wqkv.weight'), ('attn.in_proj_bias', 'attn.Wqkv.bias'),
                    ('attn.out_proj.weight','attn.out_proj.weight'),('attn.out_proj.bias','attn.out_proj.bias'),
                    ('mlp.c_fc.weight','mlp.fc1.weight'),('mlp.c_fc.bias','mlp.fc1.bias'),
                    ('mlp.c_proj.weight','mlp.fc2.weight'),('mlp.c_proj.bias','mlp.fc2.bias')
                    ]
            else:
                swaps = [('attn.in_proj_weight', 'attn.Wq.weight','attn.Wkv.weight'), ('attn.in_proj_bias', 'attn.Wq.bias','attn.Wkv.bias'),
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
            if 'temporal_embedding' not in name and 'ln_post' not in name and 'Adapter' not in name and 'cls_head' not in name:
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

        # x = x.permute(1, 0, 2)  # NLD -> LND
        x, x_t = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x_t = self.ln_post(x_t)
        x = x[:, 0]
        x_t=x_t[:, 0]
        
        x = torch.cat([x,x_t], dim=0)
        
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # B D T H W for I3D head

        return x
