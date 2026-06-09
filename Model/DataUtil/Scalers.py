import numpy as np
import torch
from torchvision.transforms import Normalize


def select_scaler(config):
    if config.attention_scaler is None:
        return no_scaler
    if config.attention_scaler == "log":
        return log_scaler
    if config.attention_scaler == "log_standardize":
        return log_standardize_scaler
    if config.attention_scaler == "log_normalize":
        return log_normalize_scaler


def no_scaler(attentions, config):
    return attentions


def log_scaler(attentions, config):
    attentions = torch.log(attentions)
    attentions = torch.nan_to_num(
        attentions,
        nan=1 / np.log(config.max_length),
        posinf=1 / np.log(config.max_length),
        neginf=1 / np.log(config.max_length),
    )
    return attentions


def log_standardize_scaler(attentions, config):
    attentions = torch.log(attentions)
    attentions = torch.nan_to_num(
        attentions,
        nan=1 / np.log(config.max_length),
        posinf=1 / np.log(config.max_length),
        neginf=1 / np.log(config.max_length),
    )
    return Normalize(1 / (np.log(config.max_length)), (2))(attentions)


def log_normalize_scaler(attentions, config):
    attentions = torch.log(attentions)
    img_batch_size = len(attentions)
    img_channels = len(attentions[0])
    img_size = len(attentions[0][0])
    _attentions = attentions.clone()
    _attentions[torch.isneginf(_attentions)] = 0
    _attentions = torch.reshape(_attentions, (img_batch_size, img_size**2))
    _mins = torch.min(_attentions, dim=-1).values
    _mins_b_imssq = torch.repeat_interleave(_mins, img_size**2, dim=-1)
    _mins_b_c_ims_ims = _mins_b_imssq.reshape(
        img_batch_size, img_channels, img_size, img_size
    )
    scaled = (attentions - _mins_b_c_ims_ims) / (-1 * _mins_b_c_ims_ims)
    scaled[torch.isneginf(scaled)] = 0
    return scaled

