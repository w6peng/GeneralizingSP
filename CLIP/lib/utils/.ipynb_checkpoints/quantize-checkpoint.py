import torch
import torch.nn as nn

import numpy as np
import cupy as cp
import random

def reconstruct_weight_from_quan_result(step, labels):
    weight = labels * step
    return weight

def quick_quantize_model(model, quantize_index, quantize_bits, quantize_bias=False):
    assert len(quantize_index) == len(quantize_bits), \
        'You should provide the same number of bit setting as layer list!'
    quantize_layer_bit_dict = {n: b for n,b in zip(quantize_index, quantize_bits)}

    for i, layer in enumerate(model.modules()):
        if i not in quantize_index:
            continue
        n_bit = quantize_layer_bit_dict[i]
        if n_bit < 0:
            continue
        if type(n_bit) == list:  # given both the bit of weight and bias
            assert len(n_bit) == 2
            assert hasattr(layer, 'weight')
            assert hasattr(layer, 'bias')
        else:
            n_bit = [n_bit, n_bit]  # using same setting for W and b

        if hasattr(layer, 'weight'):
            w = layer.weight.data
            c_max = w.abs().max().item() 
            if c_max > 0:
                step = c_max / float(2 ** (n_bit[0] - 1) - 1)
                labels = torch.round(w / step)

                w_q = labels * step
                layer.weight.data = w_q
        
        if hasattr(layer, 'bias') and quantize_bias:
            w = layer.bias.data
            c_max = w.abs().max().item() 
            if c_max > 0:
                step = c_max / float(2 ** (n_bit[1] - 1) - 1)
                labels = torch.round(w / step)

                w_q = labels * step
                layer.bias.data = w_q

def quantize_model(model, quantize_index, quantize_bits, quantize_bias=False, is_pruned=False):
    assert len(quantize_index) == len(quantize_bits), \
        'You should provide the same number of bit setting as layer list!'
    quantize_layer_bit_dict = {n: b for n,b in zip(quantize_index, quantize_bits)}
    centroid_label_dict = {}

    for i, layer in enumerate(model.modules()):
        if i not in quantize_index:
            continue
        this_cl_list = []
        n_bit = quantize_layer_bit_dict[i]
        if n_bit < 0:
            continue
        if type(n_bit) == list:  # given both the bit of weight and bias
            assert len(n_bit) == 2
            assert hasattr(layer, 'weight')
            assert hasattr(layer, 'bias')
        else:
            n_bit = [n_bit, n_bit]  # using same setting for W and b

        if hasattr(layer, 'weight'):
            w = layer.weight.data
            if is_pruned:
                nz_mask = w.ne(0)
                print('*** pruned density: {:.4f}'.format(torch.sum(nz_mask) / w.numel()))
                ori_shape = w.size()
                w = w[nz_mask]
            c_max = w.abs().max().item() 
            if c_max > 0:
                step = c_max / float(2 ** (n_bit[0] - 1) - 1)
                labels = torch.round(w / step)

                # labels += 2 ** (n_bit[0] - 1) - 1
                # centroids = np.arange(-c_max, c_max, step)
                # centroids = torch.from_numpy(centroids).cuda().view(1, -1)

                if is_pruned:
                    full_labels = labels.new(ori_shape).zero_() - 1
                    full_labels[nz_mask] = labels
                    labels = full_labels
                this_cl_list.append([step, labels])
                w_q = reconstruct_weight_from_quan_result(step, labels)
                layer.weight.data = w_q
        
        if hasattr(layer, 'bias') and quantize_bias:
            w = layer.bias.data
            c_max = w.abs().max().item() 
            if c_max > 0:
                step = c_max / float(2 ** (n_bit[1] - 1) - 1)
                labels = torch.round(w / step)

                # labels += 2 ** (n_bit[0] - 1) - 1
                # centroids = np.arange(-c_max, c_max, step)
                # centroids = torch.from_numpy(centroids).cuda().view(1, -1)
                this_cl_list.append([step, labels])
                w_q = reconstruct_weight_from_quan_result(step, labels)
                layer.bias.data = w_q

        centroid_label_dict[i] = this_cl_list
    return centroid_label_dict

def add_noise_quantize_layer(step, labels, mask, noise, prob, bitwidth):
    mask = cp.asarray(mask, dtype=cp.int64)
    noisy_labels = cp.asarray(labels.cpu(), dtype=cp.int64)
    labels_shape = noisy_labels.shape

    bits_weight = cp.arange(bitwidth-1,-1,-1)
    # bits_weight_sign = bits_weight == (bitwidth - 1)
    bits_weight = 2 ** bits_weight
    # bits_weight[bits_weight_sign] = -1
    noise_mask_arr = cp.random.rand(noisy_labels.size, bitwidth)
    noise_mask_arr = (noise_mask_arr < prob).astype(cp.int64)
    noise_mask = cp.sum(noise_mask_arr * bits_weight, axis=1)
    noise_mask = noise_mask.astype(cp.int64)

    noisy_labels = noisy_labels.reshape(-1)
    # noisy_labels = add_noise(noisy_labels, noise_mask)
    noisy_labels = noisy_labels ^ (noise_mask & (~mask))
    noisy_labels = noisy_labels.reshape(labels_shape)
    noisy_labels = noisy_labels.astype(cp.int32)
    noisy_labels = cp.asnumpy(noisy_labels)
    noisy_labels = torch.from_numpy(noisy_labels).float().cuda()

    noisy_weight = reconstruct_weight_from_quan_result(step, noisy_labels)
    return noisy_weight

def add_noise_BCH_quantize_layer_cupy(step, labels, mask, noise, prob, bitwidth, code):
    n, k ,t = code
    
    mask_pos = []
    for i in range(bitwidth):
        if mask & 0x1 == 1:
            mask_pos.append(i)
        mask = mask >> 1
    unmask_pos = [i for i in range(bitwidth) if i not in mask_pos]
    mask_pos = cp.asarray(mask_pos, dtype=cp.int32)
    unmask_pos = cp.asarray(unmask_pos, dtype=cp.int32)
    mask_bit = mask_pos.size

    noisy_labels = cp.asarray(labels.cpu(), dtype=cp.int8)
    labels_shape = labels.shape
    noisy_labels = noisy_labels.reshape(-1)
    total = noisy_labels.size

    # bch
    bch_noise = cp.random.rand((total*mask_bit - 1) // k + 1, n)
    bch_noise = (bch_noise < prob).astype(cp.int32) 
    error_mask = cp.sum(bch_noise, axis=1) > t
    bch_de = cp.zeros(((total*mask_bit - 1) // k + 1, k), dtype=cp.int32)
    bch_de[error_mask] = bch_noise[error_mask][:,:k]

    bch_noise_mask_arr = cp.zeros((total*mask_bit), dtype=cp.int32)
    bch_noise_mask_arr = bch_de.reshape(-1)[:total*mask_bit]
    bch_noise_mask_arr = bch_noise_mask_arr.reshape((total, mask_bit))
    bch_noise_mask = cp.sum(bch_noise_mask_arr * (2 ** mask_pos), axis=1)

    noise_mask_arr = cp.random.rand(total, bitwidth-mask_bit) 
    noise_mask_arr = (noise_mask_arr < prob).astype(cp.int32)
    noise_mask = cp.sum(noise_mask_arr * (2 ** unmask_pos), axis=1)
    
    noise_mask = bch_noise_mask ^ noise_mask
    noise_mask = noise_mask.astype(cp.int8)

    noisy_labels = noisy_labels ^ noise_mask
    noisy_labels = noisy_labels.reshape(labels_shape)
    noisy_labels = noisy_labels.astype(cp.int32)
    noisy_labels = cp.asnumpy(noisy_labels)
    noisy_labels = torch.from_numpy(noisy_labels).float().cuda()

    noisy_weight = reconstruct_weight_from_quan_result(step, noisy_labels)
    return noisy_weight

def add_noise_quantize_model(model, important_mask, noise, prob, quantize_index, quantize_bits, centroid_label_dict, is_pruned=False, code=None):
    quantize_layer_mask_dict = {n: m for (n, m) in zip(quantize_index, important_mask)}
    quantize_layer_bit_dict = {n: b for n, b in zip(quantize_index, quantize_bits)}
    for i, layer in enumerate(model.modules()):
        if i not in quantize_index:
            continue
        this_cl_list = centroid_label_dict[i]
        mask = quantize_layer_mask_dict[i]
        n_bit = quantize_layer_bit_dict[i]
        if n_bit < 0:  # if -1, do not quantize
            continue
        if i < 115:
            prob = 1e-6
        else:
            prob = 1e-2
        if type(n_bit) == list:  # given both the bit of weight and bias
            assert len(n_bit) == 2
            assert hasattr(layer, 'weight')
            assert hasattr(layer, 'bias')
        else:
            n_bit = [n_bit, n_bit]  # using same setting for W and b
        if code == None:
            noisy_weight_data = add_noise_quantize_layer(this_cl_list[0][0], this_cl_list[0][1], mask, noise, prob, n_bit[0])
        else:
            noisy_weight_data = add_noise_BCH_quantize_layer_cupy(this_cl_list[0][0], this_cl_list[0][1], mask, noise, prob, n_bit[0], code)
        layer.weight.data = torch.as_tensor(noisy_weight_data, dtype=layer.weight.data.dtype)



#%%
