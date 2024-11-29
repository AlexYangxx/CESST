# from fcntl import F_OFD_GETLK
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')

# depthwise-separable convolution (DSC)
class DSC(nn.Module):

    def __init__(self, nin: int) -> None:
        super(DSC, self).__init__()
        self.conv_dws = nn.Conv2d(
            nin, nin, kernel_size=1, stride=1, padding=0, groups=nin
        )
        self.norm_1 = nn.LayerNorm(nin)
        # self.bn_dws = nn.LayerNorm(nin, momentum=0.9)
        self.relu_dws = nn.ReLU(inplace=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_point = nn.Conv2d(
            nin, 1, kernel_size=1, stride=1, padding=0, groups=1
        )
        self.norm_2 = nn.LayerNorm(1)
        # self.bn_point = nn.BatchNorm2d(1, momentum=0.9)
        self.relu_point = nn.ReLU(inplace=False)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dws(x)
        out = self.norm_1(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu_dws(out)

        out = self.maxpool(out)

        out = self.conv_point(out)
        out = self.norm_2(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.relu_point(out)

        m, n, p, q = out.shape
        out = self.softmax(out.view(m, n, -1))
        out = out.view(m, n, p, q)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)

        out = out + x

        return out

#Efficient Feature Fusion(EFF)
class EFF(nn.Module):
    def __init__(self, nin: int, nout: int, num_splits: int) -> None:
        super(EFF, self).__init__()

        # assert nin % num_splits == 0
        self.conv_in = nn.Conv2d(nin, nin, 1, 1, 0, bias=False)
        self.conv_out = nn.Conv2d(nin, nout, 3, 1, 1, bias=False)

        self.nin = nin
        self.nout = nout
        self.num_splits = num_splits
        self.subspaces = nn.ModuleList(
            [DSC(int(self.nin / self.num_splits)) for i in range(self.num_splits)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        sub_feat = torch.chunk(x, self.num_splits, dim=1)
        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))
        out = torch.cat(out, dim=1)
        out = self.conv_out(out)

        return out

#Define a spatial feature transform layer
class SFT_layer(nn.Module):
    def __init__(self):
        super(SFT_layer, self).__init__()
        Relu = nn.LeakyReLU(0.2, True)

        condition_conv1 = nn.Conv2d(1,16, kernel_size=3, stride=1, padding=1)
        condition_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        condition_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        conditon_conv = [condition_conv1, Relu, condition_conv2, Relu, condition_conv3, Relu]
        self.condition_conv = nn.Sequential(*conditon_conv)

        scale_conv1 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        scale_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        scale_conv = [scale_conv1, Relu, scale_conv2, Relu]
        self.scale_conv = nn.Sequential(*scale_conv)

        sift_conv1 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)
        sift_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        sift_conv = [sift_conv1, Relu, sift_conv2, Relu]
        self.sift_conv = nn.Sequential(*sift_conv)

    def forward(self, x, texture):
        texture_condition = self.condition_conv(texture)
        scaled_feature = self.scale_conv(texture_condition) * x
        sifted_feature = scaled_feature + self.sift_conv(texture_condition)

        return sifted_feature
    
#define a residual coordiante attention block
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class residual_coordinate_Attention_block(nn.Module):
    def __init__(self, F_in, F_out, act=nn.LeakyReLU(0.2, True), use_bias=nn.InstanceNorm2d, reduction=31):
        super(residual_coordinate_Attention_block, self).__init__()

        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(F_in, F_in, kernel_size=3, stride=1, padding=1, bias=use_bias))
            # if bn: modules_body.append(nn.InstanceNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        # modules_body.append(SpatialAttention(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.relu = nn.ReLU(inplace=True)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, F_in // reduction)

        self.conv2 = nn.Conv2d(F_in, F_in, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.conv1 = nn.Conv2d(F_in, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.InstanceNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, F_in, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.conv_w = nn.Conv2d(mip, F_in, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.conv_out = nn.Conv2d(F_in, F_out, kernel_size=3, stride=1, padding=1, bias=use_bias)
        

    def forward(self, x):
        
        f = self.body(x)
        n,c,h,w = f.size()
        f = self.conv2(f)
        f = self.relu(f)
        f_h = self.pool_h(f)
        f_w = self.pool_w(f).permute(0, 1, 3, 2)

        y = torch.cat([f_h, f_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        f_h, f_w = torch.split(y, [h, w], dim=2)
        f_w = f_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(f_h).sigmoid()
        a_w = self.conv_w(f_w).sigmoid()

        out = f * a_w * a_h
        res = x + out
        res = self.conv_out(res)

        return res

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.conv4 = conv(n_feat, n_feat, kernel_size, bias=bias)
       
    def forward(self, x, rgb):
        x1 = self.conv1(x)
        img = self.conv2(x) + self.conv3(rgb)
        # img = self.img_out(x)
        x2 = torch.sigmoid(self.conv4(img))
        x1 = x1*x2
        x1 = x1+x
        
        return x1, img

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0., stride=False):
        super().__init__()
        self.stride = stride
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input dim={self.dim}, out dim={self.out_dim}"

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]


class RGB_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        x = x.reshape(b,h*w,c)
        # x_2 = x_2.reshape(b,h*w,c)
        # x_3 = x_3.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class Shuffle_MSA(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, shuffle=False, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., relative_pos_embedding=False):
        super().__init__()
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shuffle = shuffle
        # self.conv = nn.Conv2d(dim, dim, 1, 1, (0,5), bias=False)
        # dim1_pad = np.pad(ori_image, ((50, 50), (0, 0), (0, 0)), 'constant', constant_values=0)

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x):
        # x = self.conv(x)
        # x_pad = np.pad(x, ((0,0), (0, 0), (0, 5), (0, 5)), 'constant', constant_values=0)
        b, c, h, w = x.shape
        if h % self.ws !=0:
            pad_h = self.ws - h % self.ws
            pad_w = self.ws - w % self.ws
            x_pad = F.pad(x, [0, pad_w, 0, pad_h])
            b, c, h, w = x_pad.shape
            qkv = self.to_qkv(x_pad)
 
        # if h % self.ws != 0:
        #     pad = self.ws - h % self.ws
        #     qkv = nn.Conv2d(c, c, 1, 1, (0,pad), bias=False)       

        if self.shuffle:
            q, k, v = rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        else:
            q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        if self.shuffle:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
        else:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class RGB_MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                RGB_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x_1 = x.permute(0, 2, 3, 1)
        # x_2 = x_2.permute(0, 2, 3, 1)
        # x_3 = x_3.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x_1 = attn(x_1) + x_1
            x = ff(x_1) + x_1
        out = x.permute(0, 3, 1, 2)
        return out

class Shuffle_MSAB(nn.Module):
    def __init__(self, dim, num_heads, num_blocks, window_size=1, shuffle=False, qkv_bias=False, qk_scale=None, act_layer=nn.ReLU6, stride=False, relative_pos_embedding=False):
        super().__init__()
        
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                Shuffle_MSA(dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, qkv_bias=qkv_bias, qk_scale=qk_scale, relative_pos_embedding=relative_pos_embedding),
                nn.Conv2d(dim, dim, window_size, 1, window_size//2, groups=dim, bias=qkv_bias),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
        print("input dim={}, output dim={}, stride={}, expand={}, num_heads={}".format(dim, dim, stride, shuffle, num_heads))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """       
        for (attention, local, forward) in self.blocks:
            # x = x.permute(0, 2, 3, 1)
            b,c,h,w = x.shape
            attention = attention(x)
            attention = F.interpolate(attention, size=(h,w), mode='bilinear', align_corners=False) 
            x = attention + x
            x_1 = local(x)
            x_1 = x_1 + x
            x_1 = x_1.permute(0, 2, 3, 1)
            x = forward(x_1) + x_1
            x = x.permute(0, 3, 1, 2)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        # self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)

    def forward(self, x):
        # x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input dim={self.dim}, out dim={self.out_dim}"

class Shuffle_StageModule(nn.Module):
    def __init__(self, dim, heads, num_blocks, window_size=1, shuffle=True, qkv_bias=False, qk_scale=None, relative_pos_embedding=False):
        super().__init__()
        self.layers = 2

        num = self.layers // 2
        self.layers = nn.ModuleList([])
        for idx in range(num):
            the_last = (idx==num-1)
            self.layers.append(nn.ModuleList([
                Shuffle_MSAB(dim=dim, num_heads=heads, num_blocks=num_blocks, window_size=window_size, shuffle=False, qkv_bias=qkv_bias, qk_scale=qk_scale, relative_pos_embedding=relative_pos_embedding),
                Shuffle_MSAB(dim=dim, num_heads=heads, num_blocks=num_blocks, window_size=window_size, shuffle=shuffle, qkv_bias=qkv_bias, qk_scale=qk_scale, relative_pos_embedding=relative_pos_embedding)
            ]))

    def forward(self, x):
        # if self.patch_partition:
        #     x = self.patch_partition(x)
            
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x

class Spa_Spec_Block(nn.Module):
    def __init__(self, dim, dim_head, num_blocks, window_size, heads, shuffle, qkv_bias, qk_scale, relative_pos_embedding):
        super(Spa_Spec_Block, self).__init__()

        self.spec_attention = RGB_MSAB(dim=dim, num_blocks=num_blocks, dim_head=dim_head, heads=heads)
        self.spa_attention = Shuffle_StageModule(num_blocks=num_blocks, dim=dim, heads=heads, window_size=window_size, shuffle=shuffle, 
                                                qkv_bias=qkv_bias, qk_scale=qk_scale, relative_pos_embedding=relative_pos_embedding)

    def forward(self, x):
        fea_spec = self.spec_attention(x)
        fea_spa = self.spa_attention(x)
        out = fea_spec + fea_spa

        return out

class SPa_Spec_MST(nn.Module):
    def __init__(self, in_dim=31, out_dim=31, dim=31, stage=2, num_blocks=[2,4,4], window_size=8, shuffle=True, qkv_bias=False, qk_scale=None, relative_pos_embedding=True):
        super(SPa_Spec_MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                Spa_Spec_Block(
                    dim=dim_stage, dim_head=dim, num_blocks=num_blocks[i], heads=dim_stage // dim, window_size=window_size, shuffle=shuffle,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, relative_pos_embedding=relative_pos_embedding),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        # self.bottleneck = nn.ModuleList([])
        self.bottleneck = Spa_Spec_Block(
                    dim=dim_stage, dim_head=dim, num_blocks=num_blocks[-1], heads=dim_stage // dim, window_size=window_size, shuffle=shuffle,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, relative_pos_embedding=relative_pos_embedding)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                Spa_Spec_Block(
                    dim=dim_stage // 2, dim_head=dim, num_blocks=num_blocks[stage - 1 - i], heads=(dim_stage // 2) // dim, window_size=window_size, shuffle=shuffle,
                    qkv_bias=qkv_bias, qk_scale=qk_scale, relative_pos_embedding=relative_pos_embedding)
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # Embedding
        # x_1, x_2, x_3 = x.split([31,31,31],dim=1)
        fea = self.embedding(x)
        # Encoder
        fea_encoder = []
        for (Spa_Spec_Block, FeaDownSample) in self.encoder_layers:
            fea = Spa_Spec_Block(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
        # Bottleneck
        fea = self.bottleneck(fea)
        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)
        # Mapping
        out = self.mapping(fea) + x

        return out

class RGB_MSA_2(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_1, x_2, x_3):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_1.shape
        x_1 = x_1.reshape(b,h*w,c)
        x_2 = x_2.reshape(b,h*w,c)
        x_3 = x_3.reshape(b,h*w,c)
        q_inp = self.to_q(x_1)
        k_inp = self.to_k(x_2)
        v_inp = self.to_v(x_3)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class RGB_fusion(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                RGB_MSA_2(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x_r, x_g):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x_r = x_r.permute(0, 2, 3, 1)
        x_g = x_g.permute(0, 2, 3, 1)
        # x_b = x_b.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x_1 = attn(x_r, x_g, x_g) + x_r
            x = ff(x_1) + x_1
        out = x.permute(0, 3, 1, 2)
        return out
    
    
class CESST(nn.Module):
    def __init__(self, in_channels=1, out_channels=31, n_feat=31, stage=1, num_splits=31):
        super(CESST, self).__init__()
        # self.LatentFormer = LatentFormer(in_channels=1, out_channels=1, n_feat=31)
        self.stage = stage
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        modules_body = [SPa_Spec_MST(dim=31, stage=2, num_blocks=[1,1,1], window_size=7, shuffle=True, qkv_bias=False, qk_scale=None, relative_pos_embedding=True) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.fusion = residual_coordinate_Attention_block(F_in=n_feat*3, F_out=n_feat)
        self.channel_learning = RGB_fusion(dim=31, dim_head=31, heads=1, num_blocks=1)
        self.sam = SAM(n_feat, kernel_size=3, bias=False)
        self.concate = nn.Conv2d(n_feat*2, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv = nn.Conv2d(n_feat*2, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)
        self.conv_res = nn.Conv2d(3, out_channels, kernel_size=3, padding=(3 - 1) // 2,bias=False)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
        input_3 = F.interpolate(x, scale_factor= 1/4, mode='bilinear',align_corners=True) 
        input_2 = F.interpolate(x, scale_factor= 1/2, mode='bilinear',align_corners=True)

        b_2, c_2, h_inp_2, w_inp_2 = input_2.shape
        hb_2, wb_2 = 8, 8
        pad_h_2 = (hb_2 - h_inp_2 % hb_2) % hb_2
        pad_w_2 = (wb_2 - w_inp_2 % wb_2) % wb_2
        input_2 = F.pad(input_2, [0, pad_w_2, 0, pad_h_2], mode='reflect')

        b_3, c_3, h_inp_3, w_inp_3 = input_3.shape
        hb_3, wb_3 = 8, 8
        pad_h_3 = (hb_3 - h_inp_3 % hb_3) % hb_3
        pad_w_3 = (wb_3 - w_inp_3 % wb_3) % wb_3
        input_3 = F.pad(input_3, [0, pad_w_3, 0, pad_h_3], mode='reflect')

        R_1, G_1, B_1 = (x[:,0,:,:])[:,np.newaxis,:,:], (x[:,1,:,:])[:,np.newaxis,:,:], (x[:,2,:,:])[:,np.newaxis,:,:]
        R_2, G_2, B_2 = (input_2[:,0,:,:])[:,np.newaxis,:,:], (input_2[:,1,:,:])[:,np.newaxis,:,:], (input_2[:,2,:,:])[:,np.newaxis,:,:]
        R_3, G_3, B_3 = (input_3[:,0,:,:])[:,np.newaxis,:,:], (input_3[:,1,:,:])[:,np.newaxis,:,:], (input_3[:,2,:,:])[:,np.newaxis,:,:]
        
        h_middle_R_1 = self.conv_in(R_1)
        h_middle_G_1 = self.conv_in(G_1)
        h_middle_B_1 = self.conv_in(B_1)
        h_middle_R_2 = self.conv_in(R_2)
        h_middle_G_2 = self.conv_in(G_2)
        h_middle_B_2 = self.conv_in(B_2)
        h_middle_R_3 = self.conv_in(R_3)
        h_middle_G_3 = self.conv_in(G_3)
        h_middle_B_3 = self.conv_in(B_3)
     
        #compute the 3rd branch:
        # x_3 = torch.cat([h_middle_R_3, h_middle_G_3, h_middle_B_3], 1)
        h_R_3 = self.body(h_middle_R_3)
        h_G_3 = self.body(h_middle_G_3)
        h_B_3 = self.body(h_middle_B_3)
        # h_R_3, h_G_3, h_B_3 = x_3.split([31, 31, 31], dim=1)
        h_R_3 = self.conv_out(h_R_3)
        h_R_3 += h_middle_R_3
        h_G_3 = self.conv_out(h_G_3)
        h_G_3 += h_middle_G_3
        h_B_3 = self.conv_out(h_B_3)
        h_B_3 += h_middle_B_3
        h_R_3_rb = self.channel_learning(h_R_3, h_B_3)
        h_R_3_rg = self.channel_learning(h_R_3, h_G_3)
        h_R_3 = self.conv(torch.cat([h_R_3_rb, h_R_3_rg], dim=1))
        h_G_3_gr = self.channel_learning(h_G_3, h_R_3)
        h_G_3_gb = self.channel_learning(h_G_3, h_B_3)
        h_G_3 = self.conv(torch.cat([h_G_3_gb, h_G_3_gr], dim=1))
        h_B_3_br = self.channel_learning(h_B_3, h_R_3)
        h_B_3_bg = self.channel_learning(h_B_3, h_G_3)
        h_B_3 = self.conv(torch.cat([h_B_3_br, h_B_3_bg], dim=1))
        h_3 = torch.cat([h_R_3, h_G_3, h_B_3], 1)
        h_3 = self.fusion(h_3)
        h_3_feature_up, h_3_img = self.sam(h_3, input_3)
        h_3_feature_up = F.interpolate(h_3_feature_up, scale_factor= 2, mode='bilinear',align_corners=True) 

        #compute the 2rd branch:
        h_inp_2_new = h_inp_2 + pad_h_2
        w_inp_2_new = w_inp_2 + pad_w_2
        h_3_feature_up = h_3_feature_up[:, :, :h_inp_2_new, :w_inp_2_new]
        h_middle_R_2 =  self.concate(torch.cat([h_middle_R_2, h_3_feature_up], 1))
        h_middle_G_2 =  self.concate(torch.cat([h_middle_G_2, h_3_feature_up], 1))
        h_middle_B_2 =  self.concate(torch.cat([h_middle_B_2, h_3_feature_up], 1))
        # x_2 = torch.cat([h_middle_R_2, h_middle_G_2, h_middle_B_2], 1)
        h_R_2 = self.body(h_middle_R_2)
        h_G_2 = self.body(h_middle_G_2)
        h_B_2 = self.body(h_middle_B_2)
        # h_R_2, h_G_2, h_B_2 = x_2.split([31, 31, 31], dim=1)
        h_R_2 = self.conv_out(h_R_2)
        h_R_2 += h_middle_R_2
        h_G_2 = self.conv_out(h_G_2)
        h_G_2 += h_middle_G_2
        h_B_2 = self.conv_out(h_B_2)
        h_B_2 += h_middle_B_2
        h_R_2_rb = self.channel_learning(h_R_2, h_B_2)
        h_R_2_rg = self.channel_learning(h_R_2, h_G_2)
        h_R_2 = self.conv(torch.cat([h_R_2_rb, h_R_2_rg], dim=1))
        h_G_2_gr = self.channel_learning(h_G_2, h_R_2)
        h_G_2_gb = self.channel_learning(h_G_2, h_B_2)
        h_G_2 = self.conv(torch.cat([h_G_2_gb, h_G_2_gr], dim=1))
        h_B_2_br = self.channel_learning(h_B_2, h_R_2)
        h_B_2_bg = self.channel_learning(h_B_2, h_G_2)
        h_B_2 = self.conv(torch.cat([h_B_2_br, h_B_2_bg], dim=1))
        h_2 = torch.cat([h_R_2, h_G_2, h_B_2], 1)
        h_2 = self.fusion(h_2)
        h_2_feature_up, h_2_img = self.sam(h_2, input_2)
        h_2_feature_up = F.interpolate(h_2_feature_up, scale_factor= 2, mode='bilinear',align_corners=True) 

        #compute the 1st branch:
        h_inp_1_new = h_inp + pad_h
        w_inp_1_new = w_inp + pad_w
        h_2_feature_up = h_2_feature_up[:, :, :h_inp_1_new, :w_inp_1_new]
        h_middle_R_1 =  self.concate(torch.cat([h_middle_R_1, h_2_feature_up], 1))
        h_middle_G_1 =  self.concate(torch.cat([h_middle_G_1, h_2_feature_up], 1))
        h_middle_B_1 =  self.concate(torch.cat([h_middle_B_1, h_2_feature_up], 1))
        # x_1 = torch.cat([h_middle_R_1, h_middle_G_1, h_middle_B_1], 1)
        h_R_1 = self.body(h_middle_R_1)
        h_G_1 = self.body(h_middle_G_1)
        h_B_1 = self.body(h_middle_B_1)
        # h_R_1, h_G_1, h_B_1 = x_1.split([31, 31, 31], dim=1)
        h_R_1 = self.conv_out(h_R_1)
        h_R_1 += h_middle_R_1
        h_G_1 = self.conv_out(h_G_1)
        h_G_1 += h_middle_G_1
        h_B_1 = self.conv_out(h_B_1)
        h_B_1 += h_middle_B_1
        h_R_2_rb = self.channel_learning(h_R_2, h_B_2)
        h_R_2_rg = self.channel_learning(h_R_2, h_G_2)
        h_R_2 = self.conv(torch.cat([h_R_2_rb, h_R_2_rg], dim=1))
        h_G_2_gr = self.channel_learning(h_G_2, h_R_2)
        h_G_2_gb = self.channel_learning(h_G_2, h_B_2)
        h_G_2 = self.conv(torch.cat([h_G_2_gb, h_G_2_gr], dim=1))
        h_B_2_br = self.channel_learning(h_B_2, h_R_2)
        h_B_2_bg = self.channel_learning(h_B_2, h_G_2)
        h_B_2 = self.conv(torch.cat([h_B_2_br, h_B_2_bg], dim=1))
        h_1 = torch.cat([h_R_1, h_G_1, h_B_1], 1)
        h_1 = self.fusion(h_1)
        h_1_img = h_1 + self.conv_res(x)

        return h_1_img[:, :, :h_inp, :w_inp], h_2_img[:, :, :h_inp//2, :w_inp//2], h_3_img[:, :, :h_inp//4, :w_inp//4]














