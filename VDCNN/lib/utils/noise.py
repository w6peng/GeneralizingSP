# General imports
import random
import json
import torch
import numpy as np
import cupy as cp
import struct

def add_noise_float_layer(data, mask, noise, prob, bitwidth):
    # Note: Originally commented-out code has been removed for readability. Original code can be viewed in the "Deep Reinforcement Learning For Robust Networks" repo
    '''
    Adds noise to a floating-point representation of weights in a layer. An ideal ECC is used in this implementation.
    
    Arguments:
    - data: (torch.Tensor)
        - the model to add noise to
    - mask: (int)
        - bit mask for diff layers
    - noise: (str)
        - specifies the type of noise to apply
        - Note: This variable is not used in the function
    - prob: (float)
        - BER at which to add noise
    - bitwidth: (int)
        - length of weight string (i.e. bitwidth of 32 for 32-bit binary weight)
    
    Return:
    - output: (torch.Tensor)
        - layer with added weight
    '''
    
    # Turn mask and data objects into arrays
    mask = cp.asarray(mask, dtype=cp.uint32)
    data = cp.asarray(data.cpu(), dtype=cp.float32)
    data_shape = data.shape

    # Flatten data into 1d array, create same-sized array of zeros, 
    # Create & apply float_to_int function to data-length array of zeros
    # float_to_int: copy data in memory for float value x to memory for int value y
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
    
    # Creates an array of size (data, bitwidth) of random values and keeps prob proportion of them
    # (comparing # of them < prob is same as keeping prob proportion according to LLN)
    # Performs column-wise sum of noise_mask * 2^([bitwidth - 1, ..., 0])
    # Creates noise_mask which is integer representation of noise_mask_array as binary string
    # Note: noise values uniformly distributed between 0-1
    noise_mask_arr = cp.random.rand(data.size, bitwidth)
    noise_mask_arr = (noise_mask_arr < prob).astype(cp.uint32)
    noise_mask = cp.sum(noise_mask_arr * (2 ** cp.arange(bitwidth-1,-1,-1)), axis=1)
    noise_mask = noise_mask.astype(cp.uint32)

    # x_p_2 = x_p ^ (noise_mask & (~mask_2))
    # x_p XOR (noise_mask AND (NOT mask))
    # NOTing the mask gives us location where noise can be added, and ANDing this mask with the noise_mask gives all locations where the generated noise is "effective"
    # XORing this resulting "effective_noise" mask with the data applies the noise
    # In this regard, all protected bits are immune from any noise
    x_p = x_p ^ (noise_mask & (~mask))

    # int_to_float: Copy value in memory for int value x to memory for float value y
    int_to_float = cp.RawKernel(r'''
        extern "C" __global__
        void int_to_float(const unsigned int * x, float * y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            memcpy(&(y[tid]), &(x[tid]), sizeof(float));
        }
    ''', 'int_to_float')
    
    # Create data-sized array of zeros, take int-valued x_p and convert to float-valued output
    # Transforms output array to original input size, and converts back to tensor
    output = cp.zeros(data.shape, dtype=cp.float32)
    int_to_float(data.shape, (1, ), (x_p, output))
    output = output.reshape(data_shape)
    output = cp.asnumpy(output)
    output = torch.as_tensor(output, dtype=torch.float32)

    return output


def add_noise_BCH_float_layer(data, mask, noise, prob, bitwidth, code):
    # Note: Originally commented-out code has been removed for readability. Original code can be viewed in the "Deep Reinforcement Learning For Robust Networks" repo
    '''
    Adds noise to a floating-point representation of weights in a layer. A BCH ECC is used in this implementation.
    
    Arguments:
    - data: (torch.Tensor)
        - the model to add noise to
    - mask: (int)
        - bit mask for diff layers
    - noise: (str)
        - specifies the type of noise to apply
        - Note: This variable is not used in the function
    - prob: (float)
        - BER at which to add noise
    - bitwidth: (int)
        - length of weight string (i.e. bitwidth of 32 for 32-bit binary weight)
    - code: (set)
        - BCH code to be used in weight protection
    
    Return:
    - output: (torch.Tensor)
        - layer with added weight
    '''
    
    #n: code length, k: # of data bits, t: max number of correctable bit errors
    # data_shape: shape of original data (model), total: length of 1d data array
    # Turns data object into 1d array
    n, k, t = code
    data = cp.asarray(data.cpu(), dtype=cp.float32)
    data_shape = data.shape
    data = data.reshape(-1)
    total = data.size # total number of weights in layer
    
    # Creates list of mask indices where binary representation of mask has bit = 1
    # Ex: 12 = [00001100] = [2,3]
    mask_pos = []
    for i in range(bitwidth):
        if mask & 0x1 == 1: # True when LSB is 1
            mask_pos.append(i)
        mask = mask >> 1
        
    # Create list of non-masked bits, reverse sort both mask/unmask lists, and convert to array
    mask_pos = sorted(mask_pos, reverse=True)
    unmask_pos = [i for i in range(bitwidth) if i not in mask_pos]
    unmask_pos = sorted(unmask_pos, reverse=True)
    mask_pos = cp.asarray(mask_pos, dtype=cp.int32)
    unmask_pos = cp.asarray(unmask_pos, dtype=cp.int32)
    mask_bit = mask_pos.size

    # float_to_int: copy data in memory for float value x to memory for int value y
    # create data-length array of zeros & use to store float-valued data as int-valued data
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
    # Creates an array of random values and keeps prob proportion of them
    # (comparing # of them < prob is same as keeping prob proportion according to LLN)
    # Size of noise:
    #     [int(((# vals in data) * (# vals being protected) - 1) / (# data bits + 1)), code length]
    # Create error_mask as number of instances where too many errors occur (more than t)
    # Create array of int(1) vals of size 
    #     [int(((# vals in data) * (# vals being protected) - 1) / (# data bits + 1)), # data bits]
    # Sets prior array at index error_mask to be 0-valued
    bch_noise = cp.random.rand((total*mask_bit - 1) // k + 1, n) # num bch codewords * n
    bch_noise = (bch_noise < prob).astype(cp.int32) 
    error_mask = cp.sum(bch_noise, axis=1) > t
    bch_de = cp.ones(((total*mask_bit - 1) // k + 1, k), dtype=cp.int32) # nulls weights if more than t errors
    bch_de[error_mask] = cp.zeros_like(bch_de[error_mask])

    # Creates bch_noise_mask_arr to be 1-valued array of size s = (# vals in data) * (# vals being protected)
    # Sets array to first s values of 1d bch_de array
    # Reshapes to shape (# vals in data, # vals being protected)
    # Creates bch_noise_mask as column-wise sum of bch_noise_mask_arr * (2^(vals being protected))
    # Adds sum of 2^(vals not being protected) to bch_noise_mask
    # Typecasts bch_noise_mask vals as ints
    bch_noise_mask_arr = cp.ones((total*mask_bit), dtype=cp.int32)
    bch_noise_mask_arr = bch_de.reshape(-1)[:total*mask_bit]
    bch_noise_mask_arr = bch_noise_mask_arr.reshape((total, mask_bit))
    bch_noise_mask = cp.sum(bch_noise_mask_arr * (2 ** mask_pos), axis=1)
    bch_noise_mask += cp.sum(2 ** unmask_pos)
    bch_noise_mask = bch_noise_mask.astype(cp.uint32)

    # Create noise mask array for number of unprotected bits and trim noise to fit BER
    # Performs column-wise sum of noise_mask * 2^([bitwidth - 1, ..., 0])
    # Create noise_mask as int representation of binary array
    noise_mask_arr = cp.random.rand(total, bitwidth-mask_bit) 
    noise_mask_arr = (noise_mask_arr < prob).astype(cp.int32)
    noise_mask = cp.sum(noise_mask_arr * (2 ** unmask_pos), axis=1)
    noise_mask = noise_mask.astype(cp.uint32)

    # x_p (original data) XOR noise_mask: adds all noise
    # x_p AND bch_noise_mask: nulls weights if too many errors
    x_p = x_p ^ noise_mask
    x_p = x_p & bch_noise_mask

    # int_to_float: Copy value in memory for int value x to memory for float value y
    int_to_float = cp.RawKernel(r'''
        extern "C" __global__
        void int_to_float(const unsigned int * x, float * y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            memcpy(&(y[tid]), &(x[tid]), sizeof(float));
        }
    ''', 'int_to_float')
    
    # Create data-sized array of zeros, take int-valued x_p and convert to float-valued output
    # Transforms output array to original input size, and converts back to tensor
    output = cp.zeros(data.shape, dtype=cp.float32)
    int_to_float(data.shape, (1, ), (x_p, output))
    output = output.reshape(data_shape)
    output = cp.asnumpy(output)
    output = torch.as_tensor(output, dtype=torch.float32)

    return output

def add_noise_float_model(model, important_mask, noise, prob, mask_index, bitwidth=32, is_pruned=False, code=None):
    # Note: A chunk of the noise code was commented out & has been removed for readability. Original code can be viewed in the "Deep Reinforcement Learning For Robust Networks" repo
    '''
    Adds noise to floating-point represented weights layer-by-layer to all layers containing trainable weights in network.
    
    Parameters:
    - model: (torch.nn.parallel.data_parallel.DataParallel)
        - PyTorch model containing trained weights to add noise to
    - important_mask: (list)
        - list of bit masks for weights
    - noise: (str)
        - string specifier for how noise is to be added. default from ibs_env.py is 'Random'
    - mask_index: (list)
        - list of indices at which the masks are to applied
    - bitwidth=32: (int)
        - number of bits used in weight representation. default to 32 for our 32-bit IEEE-754 Floating Point
    - is_pruned=False (bool)
        - specifies is the model is pruned (if unnecessary weights have been removed)
    - code=None (str)
        - specifies code type of ECC (None = ideal, (8191, 6722, 115) = floating-point BCH)
    '''

    layer_mask_dict = {n: m for (n, m) in zip(mask_index, important_mask)}
    for i, layer in enumerate(model.modules()):
        if i not in mask_index: # skips layers that don't have a mask provided
            continue
        mask = layer_mask_dict[i]
        if code == None: # ideal ECC
            noisy_weight_data = add_noise_float_layer(layer.weight.data, mask, noise, prob, bitwidth)
        else: # BCH ECC
            noisy_weight_data = add_noise_BCH_float_layer(layer.weight.data, mask, noise, prob, bitwidth, code)
        layer.weight.data = noisy_weight_data.cuda() # replaces weights with noisy weights


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
