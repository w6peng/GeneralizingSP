import torch
import warnings
import numpy as np
from lib.utils import bits_transfer

def add_noise_quantize_layer(prob,mask,layer):
    w = layer
    c_max = w.abs().max().item()

    if c_max > 0:
        step = c_max / float(2 ** (8 - 1) - 1)
        labels = torch.round(w / step)
        pred = int_tensor_to_binary_with_sign(labels)
        pred = torch.tensor(pred)
        shape = pred.shape
        pred = pred.reshape(-1)

        mask_length = pred.numel()
        repeats = mask_length // len(mask)
        duplicated_array = np.tile(mask, repeats)[:mask_length]
        mask = torch.IntTensor(duplicated_array)

        noise_mask = torch.rand(mask_length)
        noise_mask = (noise_mask < prob)

        noise_weight_binary = pred.int() ^ (noise_mask.int() & (~mask))
        noise_weight_binary = noise_weight_binary.to(torch.int8)
        noise_weight_binary = noise_weight_binary.reshape(shape)

        noise_weight = tensor_to_integers(noise_weight_binary)

        w_q = noise_weight * step
        noise_weight = w_q

    return noise_weight

def add_noise_float_layer(prob,mask,layer):
    pred = float2bit(layer, num_e_bits=8, num_m_bits=23, bias=127.)
    shape = pred.shape
    pred = pred.reshape(-1)
    mask_length = pred.numel()
    repeats = mask_length // len(mask)
    duplicated_array = np.tile(mask, repeats)[:mask_length]
    mask = torch.IntTensor(duplicated_array)

    noise_mask = torch.rand(mask_length)
    noise_mask = (noise_mask < prob)

    noise_weight_binary = pred.int() ^ (noise_mask.int() & (~mask))
    noise_weight_binary = noise_weight_binary.reshape(shape)
    noise_weight = bit2float(noise_weight_binary)
    return noise_weight