import torch
import numpy as np
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
    attentions = torch.nan_to_num(attentions, nan=1/np.log(config.max_length), posinf=1/np.log(config.max_length), neginf=1/np.log(config.max_length))

    return attentions

def log_standardize_scaler(attentions, config):
    attentions = torch.log(attentions)
    attentions = torch.nan_to_num(attentions, nan=1/np.log(config.max_length), posinf=1/np.log(config.max_length), neginf=1/np.log(config.max_length))
    return Normalize(1/(np.log(config.max_length)),  (2))(attentions)

def log_normalize_scaler(attentions, config):
    """
    Key point : individually normalizes each attention matrix in attentions according to its min and max value.

    Args:
        attn_heads : batch of attention head weight matrices
                     expected shape : (batch_size, channels, seq_len, seq_len)
    
    Returns:
        scaled : torch.Tensor of same shape, log normalized to 0-1 scale.
    """
    # attn always 1 in 'top left' from first token to first token attention
    # attn always 0 in triu, range 0 to 1 in tril, log(attn) range : -inf to 0

    # apply log scaling
    attentions = torch.log(attentions)

    # construct min values tensor for each individual attention
    img_batch_size = len(attentions)
    img_channels = len(attentions[0])
    img_size = len(attentions[0][0])
    # use a clone to ignore triu neginf values, want to use min values in tril
    _attentions = attentions.clone()
    _attentions[torch.isneginf(_attentions)] = 0
    _attentions = torch.reshape(_attentions, (img_batch_size, img_size**2))
    _mins = torch.min(_attentions, dim=-1).values
    # repeat the min values to full attn matrix size for tensor math
    _mins_b_imssq = torch.repeat_interleave(_mins, img_size**2, dim=-1)
    _mins_b_c_ims_ims = _mins_b_imssq.reshape(
        img_batch_size, img_channels, img_size, img_size
    )

    # normalize the original log scaled values using the min values tensor
    #   norm = (x - min) / (max - min)
    # we leave out the max as it is zeros, top left is always log(1) = 0
    scaled = (attentions - _mins_b_c_ims_ims) / (-1 * _mins_b_c_ims_ims)

    # now we still need to filter out neginf values
    scaled[torch.isneginf(scaled)] = 0

    return scaled
