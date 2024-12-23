import torch
from torch import nn
from transformers.activations import ACT2FN

from einops import rearrange
from einops.layers.torch import Rearrange
from ap_mae.configuration_ap_mae import APMAEConfig

class FeedForward(nn.Module):
    def __init__(self, config: APMAEConfig, dim, hidden_dim, dropout = 0.):
        super().__init__()
        
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.net = nn.Sequential(
            nn.LayerNorm(dim, eps=config.layer_norm_eps),
            nn.Linear(dim, hidden_dim),
            ACT2FN[config.hidden_act],
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, config: APMAEConfig, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = config.qkv_bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, config: APMAEConfig, dim, layers, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(config, dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(config, dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class APMAEEncoder(nn.Module):
    def __init__(
        self,
        *,
        config: APMAEConfig,
    ):
        super().__init__()
        self.config = config
        self.max_length = config.max_length
        self.patch_size = config.patch_size
        self.encoder_dim = config.encoder_dim
        self.encoder_layers = config.encoder_layers
        self.encoder_heads = config.encoder_heads
        self.encoder_dim_head = config.encoder_dim_head
        self.encoder_mlp_dim = config.encoder_mlp_dim
        self.encoder_dropout = config.encoder_dropout
        self.encoder_emb_dropout = config.encoder_emb_dropout
        self.encoder_pool = config.encoder_pool

        self.pixel_values_per_patch = self.patch_size**2
        self.grid_size = self.max_length // self.patch_size
        self.num_masked = int(config.mask_ratio * sum(range(self.grid_size+1)))
        self.num_patches = self.grid_size**2

        self.to_patches = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1 = self.patch_size,
            p2 = self.patch_size
        )
        self.patches_to_embedding = nn.Sequential(
            nn.LayerNorm(self.pixel_values_per_patch, eps=config.layer_norm_eps),
            nn.Linear(self.pixel_values_per_patch, self.encoder_dim),
            nn.LayerNorm(self.encoder_dim, eps=config.layer_norm_eps),
        )
        # newer vitmae model uses sin-cos initialized which might be better
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.encoder_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dim))

        self.transformer = Transformer(config, self.encoder_dim, self.encoder_layers, self.encoder_heads, self.encoder_dim_head, self.encoder_mlp_dim, self.encoder_dropout)

        self.pool = self.encoder_pool
    
    def forward(self, attn_weights, return_patch_info=True, masked=True):
        """
            attn_weights: batch_s x 1 x n x n tensor of attention weights.

            dim 1 is a leftover from mimicking vitmae, was color channels.
            using it as one channel works fine, but removable in future
        """
        device = attn_weights.device

        # get patches
        patches = self.to_patches(attn_weights)
        # patch to encoder tokens and add positions
        tokens = self.patches_to_embedding(patches)
        if self.pool == "cls":
            tokens += self.pos_embedding[:, 1:(self.num_patches + 1)]
        elif self.pool == "mean":
            tokens += self.pos_embedding.to(device, dtype=tokens.dtype)

        # apply our masking procedure, include tril patches only
        bottom_left_ids = torch.tril_indices(self.grid_size, self.grid_size, device = device)
        id_matrix = torch.arange(0,self.num_patches, device = device).reshape(self.grid_size, self.grid_size)


        sampleable_ids = id_matrix[bottom_left_ids[0], bottom_left_ids[1]]
        # indices for random patch masks across the whole batch
        rand_indices = torch.rand(attn_weights.shape[0], len(sampleable_ids), device = device).argsort(dim = -1)

        shuffled_patch_ids = sampleable_ids[rand_indices]
        unmasked_indices = shuffled_patch_ids[:,self.num_masked:]

        # high performance gather implementation for the random mask indexing.
        unmasked_gather_indices = unmasked_indices.unsqueeze(dim=-1).repeat(1,1,tokens.shape[-1])
        if masked:
            sel_tokens = tokens.gather(1, unmasked_gather_indices)
        else:
            sel_tokens = tokens
        if return_patch_info:
            masked_indices = shuffled_patch_ids[:,:self.num_masked]
            masked_gather_indices = masked_indices.unsqueeze(dim=-1).repeat(1,1,patches.shape[-1])
            masked_patches = patches.gather(1, masked_gather_indices)

        # add cls tokens for each batch after masking
        cls_token = self.cls_token + self.pos_embedding[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, sel_tokens), dim=1)

        # we now have our prepared embeddings to run through encoder
        # shape(batches, cls_tokens_patch+selected_patches, encoder_dim)
        encoded_tokens = self.transformer(tokens)

        if return_patch_info:
            return encoded_tokens, unmasked_indices, masked_indices, masked_patches
        else:
            return encoded_tokens
        
    def encode(self, attn_weights):
        """
            attn_weights: batch_s x 1 x n x n tensor of attention weights.

            dim 1 is a leftover from mimicking vitmae, was color channels.
            using it as one channel works fine, but removable in future
        """
        return self.forward(attn_weights, return_patch_info=False, masked = False)

