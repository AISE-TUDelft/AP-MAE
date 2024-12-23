from transformers.configuration_utils import PretrainedConfig

class APMAEConfig(PretrainedConfig):
    model_type = "ap_mae"

    def __init__(
        self,
        max_length = 256,
        patch_size = 32,
        mask_ratio = 0.5,

        decoder_dim = 512,
        decoder_layers = 8,
        decoder_heads = 8,
        decoder_dim_head = 64,
        decoder_mlp_dim = 2048,

        encoder_dim = 512,
        encoder_layers = 24,
        encoder_heads = 16,
        encoder_dim_head = 64,
        encoder_mlp_dim = 2048,
        
        encoder_dropout = 0.,
        encoder_emb_dropout = 0.,
        encoder_pool = 'cls',

        hidden_act="gelu",
        layer_norm_eps=1e-12,
        qkv_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.decoder_dim = decoder_dim
        self.decoder_mlp_dim = decoder_mlp_dim
        self.decoder_layers = decoder_layers
        self.decoder_heads = decoder_heads
        self.decoder_dim_head = decoder_dim_head

        self.encoder_dim = encoder_dim
        self.encoder_layers = encoder_layers
        self.encoder_heads = encoder_heads
        self.encoder_dim_head = encoder_dim_head
        self.encoder_mlp_dim = encoder_mlp_dim
        self.encoder_dropout = encoder_dropout
        self.encoder_emb_dropout = encoder_emb_dropout
        self.encoder_pool = encoder_pool

        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
