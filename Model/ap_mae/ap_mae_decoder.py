import torch
from torch import nn
from transformers.activations import ACT2FN
import torch.nn.functional as F
from einops import repeat, rearrange
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

class APMAEDecoder(nn.Module):
    def __init__(
        self,
        *,
        config: APMAEConfig,
    ):
        super().__init__()
        self.config = config
        self.max_length = config.max_length
        self.patch_size = config.patch_size
        self.mask_ratio = config.mask_ratio
        self.decoder_dim = config.decoder_dim
        self.decoder_mlp_dim = config.decoder_mlp_dim
        self.decoder_layers = config.decoder_layers
        self.decoder_heads = config.decoder_heads
        self.decoder_dim_head = config.decoder_dim_head
        self.encoder_dim = config.encoder_dim

        self.grid_size = self.max_length // self.patch_size
        self.num_patches = self.grid_size**2
        self.num_masked = int(self.mask_ratio * sum(range(self.grid_size+1)))
        self.pixel_values_per_patch = self.patch_size**2

        if self.encoder_dim != self.decoder_dim:
            self.enc_to_dec = nn.Linear(self.encoder_dim, self.decoder_dim) 
        else:
            self.enc_to_dec = nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(self.decoder_dim))
        # newer vitmae model uses sin-cos initialized which might be better
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.decoder_dim)
        )
        self.to_pixels = nn.Linear(
            self.decoder_dim, self.pixel_values_per_patch
        )

        self.transformer = Transformer(
            config, 
            dim = self.decoder_dim, 
            layers = self.decoder_layers, 
            heads = self.decoder_heads, 
            dim_head = self.decoder_dim_head, 
            mlp_dim = self.decoder_mlp_dim
        )
    
    def forward(
            self, 
            encoded_tokens, 
            unmasked_indices, 
            masked_indices, 
            masked_patches, 
            visualizing=False
        ):
        batch_size = encoded_tokens.shape[0]
        batch_range = torch.arange(batch_size, device = encoded_tokens.device)[:, None]

        # project encoder to decoder dimensions, if they are not equal the 
        # paper says you can get away with a smaller dimension for decoder

        emb_decoder_tokens = self.enc_to_dec(encoded_tokens)
        unmasked_decoder_tokens = emb_decoder_tokens[:,1:,:]

        # repeat mask tokens for number of batches and masked patches
        mask_tokens = repeat(
            self.mask_token, 'd -> b n d', b = batch_size, n = self.num_masked
        )

        # concat to get back to original shape number of patches
        decoder_tokens = torch.zeros(
            batch_size, 
            self.num_patches, 
            self.decoder_dim, 
            device=emb_decoder_tokens.device
        )

        # high performance scatter implementation to restore patch locations.
        unmasked_gather_indices = unmasked_indices.unsqueeze(dim=-1).repeat(1,1,decoder_tokens.shape[-1])
        masked_gather_indices = unmasked_indices.unsqueeze(dim=-1).repeat(1,1,decoder_tokens.shape[-1])
        decoder_tokens = decoder_tokens.scatter(1,unmasked_gather_indices, unmasked_decoder_tokens)
        decoder_tokens = decoder_tokens.scatter(1,masked_gather_indices, mask_tokens)

        # add the cls tokens back in front
        decoder_tokens = torch.cat(
            (emb_decoder_tokens[:,:1,:], decoder_tokens), 
            dim=1
        )
        # apply the encoder positional embedding all at once
        decoder_tokens = decoder_tokens + self.pos_embedding

        # decoder embedded inputs are ready to run through decoder
        # shape is now (batches, 1_cls_patch+all_patches, decoder_dim)
        decoded_tokens = self.transformer(decoder_tokens)

        # project to pixel values
        pred_pixel_values = self.to_pixels(decoded_tokens)
        # split out cls tokens patch
        patches = pred_pixel_values[:,1:,:]
        # select the predicted values for masked patches
        pred_patches = patches[batch_range, masked_indices]
        # calculate reconstruction loss against true masked patches
        recon_loss = F.mse_loss(pred_patches, masked_patches)

        if visualizing:
            return recon_loss, pred_patches, masked_indices, masked_patches, unmasked_indices
        else:
            return recon_loss