import random
import json

import torch
import numpy as np
import cupy as cp

import struct


def add_noise_float_layer(data, mask, noise, prob, bitwidth):
    org_dtype = data.dtype
    mask = cp.asarray(mask, dtype=cp.uint32)
    # mask_2 = cp.asarray(mask_2, dtype=cp.uint32)
    data = cp.asarray(data.cpu(), dtype=cp.float32)
    data_shape = data.shape

    data = data.reshape(-1)
    float_to_int = cp.RawKernel(r'''
        extern "C" __global__
        void float_to_int(const float * x, unsigned int * y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            memcpy(&(y[tid]), &(x[tid]), sizeof(float));
        }
    ''', 'float_to_int')
    x_p = cp.zeros(data.shape, dtype=cp.uint32)
    float_to_int(data.shape, (1, ), (data, x_p))

    noise_mask_arr = cp.random.rand(data.size, bitwidth)
    noise_mask_arr = (noise_mask_arr < prob).astype(cp.uint32)
    noise_mask = cp.sum(noise_mask_arr * (2 ** cp.arange(bitwidth-1,-1,-1)), axis=1)
    noise_mask = noise_mask.astype(cp.uint32)
    
    # x_p_2 = x_p ^ (noise_mask & (~mask_2))
    x_p = x_p ^ (noise_mask & (~mask))

    int_to_float = cp.RawKernel(r'''
        extern "C" __global__
        void int_to_float(const unsigned int * x, float * y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            memcpy(&(y[tid]), &(x[tid]), sizeof(float));
        }
    ''', 'int_to_float')
    output = cp.zeros(data.shape, dtype=cp.float32)
    int_to_float(data.shape, (1, ), (x_p, output))
    output = output.reshape(data_shape)
    output = cp.asnumpy(output)
    output = torch.as_tensor(output, dtype=org_dtype)

    # output_2 = cp.zeros(data.shape, dtype=cp.float32)
    # int_to_float(data.shape, (1, ), (x_p_2, output_2))
    # output_2 = output_2.reshape(data_shape)
    # output_2 = cp.asnumpy(output_2)
    # output_2 = torch.as_tensor(output_2, dtype=torch.float32)
    return output

def add_noise_BCH_float_layer(data, mask, noise, prob, bitwidth, code):
    org_dtype = data.dtype
    n, k, t = code
    data = cp.asarray(data.cpu(), dtype=cp.float32)
    data_shape = data.shape
    data = data.reshape(-1)
    total = data.size
    
    mask_pos = []
    for i in range(bitwidth):
        if mask & 0x1 == 1:
            mask_pos.append(i)
        mask = mask >> 1
    mask_pos = sorted(mask_pos, reverse=True)
    unmask_pos = [i for i in range(bitwidth) if i not in mask_pos]
    unmask_pos = sorted(unmask_pos, reverse=True)
    mask_pos = cp.asarray(mask_pos, dtype=cp.int32)
    unmask_pos = cp.asarray(unmask_pos, dtype=cp.int32)
    mask_bit = mask_pos.size

    float_to_int = cp.RawKernel(r'''
        extern "C" __global__
        void float_to_int(const float * x, unsigned int * y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            memcpy(&(y[tid]), &(x[tid]), sizeof(float));
        }
    ''', 'float_to_int')
    x_p = cp.zeros(data.shape, dtype=cp.uint32)
    float_to_int(data.shape, (1, ), (data, x_p))

    # bch
    bch_noise = cp.random.rand((total*mask_bit - 1) // k + 1, n)
    bch_noise = (bch_noise < prob).astype(cp.int32) 
    error_mask = cp.sum(bch_noise, axis=1) > t
    bch_de = cp.ones(((total*mask_bit - 1) // k + 1, k), dtype=cp.int32)
    bch_de[error_mask] = cp.zeros_like(bch_de[error_mask])
    # bch_de = cp.zeros(((total*mask_bit - 1) // k + 1, k), dtype=cp.int32)
    # bch_de[error_mask] = bch_noise[error_mask][:,:k]

    bch_noise_mask_arr = cp.ones((total*mask_bit), dtype=cp.int32)
    # bch_noise_mask_arr = cp.zeros((total*mask_bit), dtype=cp.int32)
    bch_noise_mask_arr = bch_de.reshape(-1)[:total*mask_bit]
    bch_noise_mask_arr = bch_noise_mask_arr.reshape((total, mask_bit))
    bch_noise_mask = cp.sum(bch_noise_mask_arr * (2 ** mask_pos), axis=1)
    bch_noise_mask += cp.sum(2 ** unmask_pos)
    bch_noise_mask = bch_noise_mask.astype(cp.uint32)

    noise_mask_arr = cp.random.rand(total, bitwidth-mask_bit) 
    noise_mask_arr = (noise_mask_arr < prob).astype(cp.int32)
    noise_mask = cp.sum(noise_mask_arr * (2 ** unmask_pos), axis=1)
    
    noise_mask = noise_mask.astype(cp.uint32)

    x_p = x_p ^ noise_mask
    x_p = x_p & bch_noise_mask
    # x_p = x_p ^ bch_noise_mask

    int_to_float = cp.RawKernel(r'''
        extern "C" __global__
        void int_to_float(const unsigned int * x, float * y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            memcpy(&(y[tid]), &(x[tid]), sizeof(float));
        }
    ''', 'int_to_float')
    output = cp.zeros(data.shape, dtype=cp.float32)
    int_to_float(data.shape, (1, ), (x_p, output))
    output = output.reshape(data_shape)
    output = cp.asnumpy(output)
    output = torch.as_tensor(output, dtype=org_dtype)

    return output

def add_noise_float_model(model, important_mask, noise, prob, mask_index, bitwidth=32, is_pruned=False, code=None):
    layer_mask_dict = {n: m for (n, m) in zip(mask_index, important_mask)}

    for i, layer in enumerate(model.modules()):
        if i not in mask_index:
            continue
        mask = layer_mask_dict[i]
        if i < 115:
            prob = 1e-6
        else:
            prob = 1e-2
        if code == None:
            noisy_weight_data = add_noise_float_layer(layer.weight.data, mask, noise, prob, bitwidth)
        else:
            noisy_weight_data = add_noise_BCH_float_layer(layer.weight.data, mask, noise, prob, bitwidth, code)
        layer.weight.data = noisy_weight_data.cuda()


def add_noise_float_model_weight_data(model, mask_index, data_1, data_2, alpha):
    for i, layer in enumerate(model.modules()):
        if i not in mask_index:
            continue
        weight_1 = data_1[i]
        if data_2 is None:
            weight_2 = layer.weight.data
        else:
            weight_2 = data_2[i]

        layer.weight.data = alpha * weight_1 + (1 - alpha) * weight_2
