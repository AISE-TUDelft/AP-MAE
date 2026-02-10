This work is based on forking https://github.com/lucidrains/vit-pytorch

We base our ap_mae_encoder.py on vit.py
We base our ap_mae_decoder.py on mae.py

The combined ap_mae.py model is based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py

The configuration_ap_mae.py template is based on
https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/configuration_vit_mae.py

Usage:
from ap_mae import APMAE, APMAEConfig
config = APMAEConfig(
    % optionally set custom model parameters
)
model = APMAE(config)
