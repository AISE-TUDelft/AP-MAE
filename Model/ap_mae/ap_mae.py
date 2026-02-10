import torch
import torch.nn as nn
from ap_mae.configuration_ap_mae import APMAEConfig
from ap_mae.ap_mae_encoder import APMAEEncoder
from ap_mae.ap_mae_decoder import APMAEDecoder
from transformers.modeling_utils import PreTrainedModel

class APMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.

    APMAE is designed to train on nxn attention weight matrices.
    """

    config_class = APMAEConfig
    base_model_prefix = "ap_mae"
    main_input_name = "attn_weights"
    supports_gradient_checkpointing = False


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class APMAE(APMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = APMAEEncoder(config=config)
        self.decoder = APMAEDecoder(config=config)

        self.post_init()
    def forward(self, attn_weights, visualizing=False):
        tokens, unmasked_indices, masked_indices, masked_patches = self.encoder(attn_weights)

        if self.encoder.pos_embedding.device != self.decoder.pos_embedding.device:
            decoder_device = self.decoder.pos_embedding.device
            encoded_tokens = encoded_tokens.to(decoder_device)
            masked_indices = masked_indices.to(decoder_device)
            unmasked_indices = unmasked_indices.to(decoder_device)
            masked_patches = masked_patches.to(decoder_device)

        return self.decoder(
            tokens,
            unmasked_indices,
            masked_indices,
            masked_patches,
            visualizing=visualizing
        )
    def to(self, *args, encoder_device=None, decoder_device=None, **kwargs):
        """
        Optional simple device map to split up encoder and decoder components.
        encoder_device : torch device for encoder component
        decoder_device : torch device for decoder component

        same effect as original .to() if left unspecified or set equal.
        """
        if encoder_device is None and decoder_device is None:
            print(f"APMAE loaded with .to(", *args, ")", flush=True)
            return super().to(*args, **kwargs)
        if encoder_device == decoder_device:
            print(f"APMAE loaded with .to({encoder_device},", *args, ")", flush=True)
            return super().to(encoder_device, *args, **kwargs)
        if encoder_device is None:
            print(f"APMAE model warning: decoder device set but encoder device left unspecified : None defaults to cpu")
        if decoder_device is None:
            print(f"APMAE model warning: encoder device set but decoder device left unspecified : None defaults to cpu")
        
        self.encoder = self.encoder.to(encoder_device, *args, **kwargs)
        self.decoder = self.decoder.to(decoder_device, *args, **kwargs)
        print("simple APMAE .to() device map used", flush=True)
        print(f"encoder device: {self.encoder.pos_embedding.device}", flush=True)
        print(f"decoder device: {self.decoder.pos_embedding.device}", flush=True)
        return self