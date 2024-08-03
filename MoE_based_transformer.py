# Modified from: https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/models/simplevit.py
# and  from: https://gist.github.com/ruvnet/0928768dd1e4af8816e31dde0a0205d5

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from moe import MoELayer

# from moe import MoE

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

# THIS IS REPLACED BY THE MoE in the Moe_Based Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MoE_based_Transformer(nn.Module):
    # def __init__(self, dim, depth, heads, dim_head, moe_input_size, moe_output_size, moe_num_experts, moe_hidden_size, moe_noisy_gating, moe_k):
    def __init__(self, dim, depth, heads, dim_head, moe_input_size, moe_output_dim, moe_num_experts, moe_hidden_dim, moe_noisy_gating, num_experts_per_tok):
        super(MoE_based_Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.num_experts_per_tok = num_experts_per_tok ##### MODIFICATION
        self.moe_layer = MoELayer(dim, moe_hidden_dim, moe_output_dim, moe_num_experts) ##### MODIFICATION

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head), 
                # MoE(input_size=dim, output_size=dim, num_experts=moe_num_experts, hidden_size=moe_hidden_size, noisy_gating=moe_noisy_gating, k=moe_k)
                MoELayer(dim, hidden_dim=moe_hidden_dim, output_dim=dim, num_experts=moe_num_experts)
            ]))
    def forward(self, x):
        for attn, moe_layer in self.layers:

            x = attn(x) + x
            x = self.moe_layer(x, self.num_experts_per_tok) + x
        return x
    
    
class MoE_ViT_classifier(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                moe_input_size, moe_output_dim, moe_num_experts, moe_hidden_dim, moe_noisy_gating, moe_k, num_experts_per_tok, # MoE parameters
                pool = 'cls', channels = 3, dim_head = 64, emb_dropout = 0.):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        print('num_patches', num_patches)
        print('patch_dim', patch_dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )        

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = MoE_based_Transformer(dim, depth, heads, dim_head, moe_input_size, moe_output_dim, moe_num_experts, moe_hidden_dim, moe_noisy_gating, num_experts_per_tok)
        
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)  