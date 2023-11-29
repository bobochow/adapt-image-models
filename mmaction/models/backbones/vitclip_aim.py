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

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


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
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, scale=1., num_tadapter=1, num_frames=8, drop_path=0.,
                 prompt=True,wind_attn=False,window_size=(32,2,2),shift_size=(0,0,0)):
        super().__init__()
        self.num_tadapter = num_tadapter
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.n_head = n_head
        self.d_model = d_model

        self.MLP_Adapter = Adapter(d_model, skip_connect=False)
        self.S_Adapter = Adapter(d_model)
        self.scale = scale
        self.T_Adapter = Adapter(d_model, skip_connect=False)
        if num_tadapter == 2:
            self.T_Adapter_in = Adapter(d_model)
        self.num_frames = num_frames
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.prompt=prompt
        self.wind_attn=wind_attn
        if self.wind_attn:
            self.window_size = window_size
            self.shift_size = shift_size
            
            assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
            assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
            assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        
    # def attention(self, x: torch.Tensor, attn_mask=None):
    #     attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
    #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def attention(
            self, x: torch.Tensor , mask=None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            
            # input: L , BT, D
            
            L, B_, C = x.shape
            
            q = (x @ self.attn.in_proj_weight[:self.d_model].T
                ) + self.attn.in_proj_bias[:self.d_model]

            k = (x @ self.attn.in_proj_weight[self.d_model:-self.d_model].T
                ) + self.attn.in_proj_bias[self.d_model:-self.d_model]
            v = (x @ self.attn.in_proj_weight[-self.d_model:].T
                ) + self.attn.in_proj_bias[-self.d_model:]
            Tx, Ty, N = q.size(0), k.size(0), q.size(1)
            q = q.view(Tx, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Tx D_head
            k = k.view(Ty, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3) # N num_heads Ty D_head
            v = v.view(Ty, N, self.attn.num_heads,
                    self.attn.head_dim).permute(1, 2, 0, 3)
            
            aff = (q @ k.transpose(-2, -1) / (self.attn.head_dim**0.5)) # (N, num_heads, Tx, Ty)
            
            # aff = aff.softmax(dim=-1)
            
            if mask is not None:
                # nW, ws[0]*ws[1]*ws[2]
                nW = mask.shape[0]
                aff = aff.view(N // nW, nW, self.n_head, Tx, Ty) + mask.unsqueeze(1).unsqueeze(0)
                aff = aff.view(-1, self.n_head, Tx, Ty)
                aff = aff.softmax(dim=-1)
            else:
                aff = aff.softmax(dim=-1)
            
            out = aff @ v  # N num_heads Tx D_head
            out = out.permute(2, 0, 1, 3).flatten(2)
            out = self.attn.out_proj(out) # N Tx D
            
            return out
    
    def forward(self, x: torch.Tensor, mask_matrix):
        if not self.wind_attn:
            ## x shape [HW+1, BT, D]
            n, bt, d = x.shape
            ## temporal adaptation
            xt = rearrange(x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            if self.num_tadapter == 2:
                xt = self.T_Adapter(self.attention(self.T_Adapter_in(self.ln_1(xt))))
            else:
                xt = self.T_Adapter(self.attention(self.ln_1(xt)))
            xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            x = x + self.drop_path(xt)
            ## spatial adaptation
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            ## joint adaptation
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
        else:
            # x shape [HW+1, BT, D]
            # window local attention
            
            cls_token,windows=x[:1,:,:],x[1:,:,:]
            
            L, BT, C = windows.shape
            T = self.num_frames
            B = BT // self.num_frames
            H = W = int(L**0.5)
            
            window_size, shift_size = get_window_size((T, H, W), self.window_size, self.shift_size)
            
            windows=rearrange(windows, '(h w) (b t) c -> b t h w c', t=self.num_frames, h=H, w=W)
            
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - T % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
            windows = F.pad(windows, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, Dp, Hp, Wp, _ = windows.shape
            
            if any(i > 0 for i in shift_size):
                shifted_win = torch.roll(windows, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                attn_mask = mask_matrix
            else:
                shifted_win = windows
                attn_mask = None
            
            shifted_win=window_partition(shifted_win, window_size) # B*nW, Wd*Wh*Ww, C
            
            shifted_win = shifted_win.permute(1, 0, 2)
            
            shifted_win=self.attention(self.ln_1(shifted_win),attn_mask) # Wd*Wh*Ww, B*nW, C
            
            shifted_win = shifted_win.permute(1, 0, 2)
            shifted_win = shifted_win.view(-1, *(window_size+(C,)))
            shifted_win = window_reverse(shifted_win, window_size, B, Dp, Hp, Wp) # B D' H' W' C
            
            # reverse cyclic shift
            if any(i > 0 for i in shift_size):
                windows_attn = torch.roll(shifted_win, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            else:
                windows_attn = shifted_win
            
            if pad_d1 >0 or pad_r > 0 or pad_b > 0:
                windows_attn = windows_attn[:, :T, :H, :W, :].contiguous()
            
            windows_attn = rearrange(windows_attn, 'b t h w c -> (h w) (b t) c')
            
            cls_token = rearrange(cls_token, 'n (b t) d -> t (b n) d', t=self.num_frames)
            cls_attn = self.attention(self.ln_1(cls_token))
            cls_attn = rearrange(cls_attn, 't (b n) d -> n (b t) d', n=1)
            xt = torch.cat([cls_attn,windows_attn],dim=0)
            
            x = x + self.drop_path(self.T_Adapter(xt))
            
            # temporal class token prompt adaptation
            # t_cls=xt[:1,:,:]
            if self.prompt:
                tcls_prompt=cls_attn
                x = torch.cat([x[:1, :, :], tcls_prompt, x[1:, :, :]], dim=0) # HW+2 BT D
            
            ## spatial adaptation
            x = x + self.S_Adapter(self.attention(self.ln_1(x)))
            
            # x = x + self.attention(self.ln_1(x)) + self.drop_path(self.scale * self.S_Adapter(x))
            
            if self.prompt:
                x = torch.cat([x[:1, :, :], x[2:, :, :]], dim=0)
            
            # x = x + self.drop_path(xt)
            
            ## joint adaptation
            xn = self.ln_2(x)
            x = x + self.mlp(xn) + self.drop_path(self.scale * self.MLP_Adapter(xn))
            
        return x


class Transformer(nn.Module):
    def __init__(self, num_frames, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, num_tadapter=1, scale=1., drop_path=0.1,
                 prompt=True,wind_attn=False,window_size= (32,2,2),not_shift=True):
        super().__init__()
        self.width = width
        self.layers = layers
        self.num_frames = num_frames
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.layers)]
        
        # window_size=(8, 7, 7) if i <= 5 else (32, 2, 2)
        
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width, 
                heads, 
                attn_mask, 
                scale, 
                num_tadapter, 
                num_frames, 
                dpr[i],
                prompt,
                wind_attn,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) or not_shift else self.shift_size,
                
                ) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        
        # calculate attention mask for SW-MSA
        L, BT, C = x.shape
        T = self.num_frames
        H = W = int((L-1)**0.5)
        window_size, shift_size = get_window_size((T,H,W), self.window_size, self.shift_size)
        # x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(T / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.resblocks:
            x = blk(x, attn_mask)
        
        # return self.resblocks(x)
        return x


@BACKBONES.register_module()
class AIM(nn.Module):
    ## ViT definition in CLIP image encoder
    def __init__(self, input_resolution: int, num_frames: int, patch_size: int, width: int, layers: int, heads: int, drop_path_rate, num_tadapter=1, adapter_scale=0.5, pretrained=None,
                 prompt=True,wind_attn=False,window_size= (32,2,2),not_shift=True):
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

        self.transformer = Transformer(num_frames, width, layers, heads, num_tadapter=num_tadapter, scale=adapter_scale, drop_path=drop_path_rate,
                                       prompt = prompt,wind_attn = wind_attn,window_size = window_size,not_shift=not_shift)

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
            msg = self.load_state_dict(pretrain_dict, strict=False)
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

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) d -> b d t',b=B,t=T)
        
        x = x.unsqueeze(-1).unsqueeze(-1)  # BDTHW for I3D head

        return x