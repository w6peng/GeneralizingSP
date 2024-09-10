# General imports
import os
import csv
import math
import json
import cupy as cp
import torch
from torch.autograd import Variable

# VDCNN imports
from tqdm import tqdm # Progress bar
from torch.utils.data import Dataset
import lmdb
import numpy as np
import torch.nn.functional as F

# Source (likely): https://github.com/ansleliu/LightNet/blob/master/scripts/model_measure.py

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def norm_sigmoid(x, a=1):
    return (1 / (1 + math.exp(-a*x)) - 0.5) / math.fabs(0.5 - 1 / (1 + math.exp(-a)))

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).type(dtype)

def export_results(filename, data, field_names=None):
    '''
    Appends the results of a trial to a CSV file
    
    Args:
    - filename (str): name of file
    - data (list/dict): dictionary (or list containing dicts) of data to append
    - field_names (list): column header names for the csv file
    '''
    # Ensure data is wrapped properly
    if type(data) != list:
        assert type(data) == dict, "Underlying data must be in dict format"
        data = [data]
    
    # If no field_names (header) passed, infer order from data
    if not field_names:
        field_names = data[0].keys()
        
    # Check file existence for writing header
    file_existed = os.path.isfile(filename)

    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)

        if not file_existed:
            writer.writeheader()

        writer.writerows(data)

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters:
        parent_dir : (str)
              Path of the directory containing all experiment runs.

    Returns:
        parent_dir/run_dir
            Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

def get_num_gen(gen):
    return sum(1 for x in gen)

def is_leaf(model):
    return get_num_gen(model.children()) == 0

def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name

def get_layer_param(model):
    import operator
    import functools
    # Gives num weights + biases in layer
    return sum([functools.reduce(operator.mul, i.size(), 1) for i in model.parameters()])

def measure_layer(layer, x, arch):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)
    
    #layer.in_h: output
    #layer.in_w: ??
    
    # ops_conv
    if type_name in ['Conv1d']:
        # Note: Template copied from Conv2d with changes made for dimension
        out_h = int((x.size()[1] + layer.padding[0] - layer.kernel_size[0]) / layer.stride[0] + 1)
        layer.in_h = x.size()[1]        
        out_w = int((x.size()[2] / layer.stride[0] + 1))
        layer.in_w = x.size()[2]
        layer.out_h = out_h
        layer.out_w = out_w
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[0] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)
        layer.flops = delta_ops
        layer.params = delta_params
        
    elif type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        layer.in_h = x.size()[2]
        layer.in_w = x.size()[3]
        layer.out_h = out_h
        layer.out_w = out_w
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)
        layer.flops = delta_ops
        layer.params = delta_params

    # ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel() / x.size(0)
        delta_params = get_layer_param(layer)

    # ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    # ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        if layer.bias is not None:
            bias_ops = layer.bias.numel()
        else:
            bias_ops = 0
        layer.in_h = x.size()[1]
        layer.in_w = 1
        delta_ops = weight_ops + bias_ops
        delta_params = get_layer_param(layer)
        layer.flops = delta_ops
        layer.params = delta_params

    # ops_nothing
    elif type_name in ['BatchNorm1d', 'BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'MaxPool1d', 'AdaptiveMaxPool1d']:
        delta_params = get_layer_param(layer)

    # unknown layer type
    else:
        delta_params = get_layer_param(layer)

    count_ops += delta_ops
    count_params += delta_params

    return delta_ops, delta_params

def measure_model(model, c, H, W, arch, batch_size=4):
    print('==> Measuring Model')
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    
    # Want to be of shape of input channel in network
    data = torch.zeros(batch_size, c, H, W).cuda() # batch size, channels, height, width

    def should_measure(x):
        return is_leaf(x)

    def modify_forward(model):
        for child in model.children(): # All layers
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x, arch)
                        #x = x.long()
                        return m.old_forward(x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)
                
    modify_forward(model)
    data = torch.unsqueeze(data,dim=3)
    data = data.squeeze()
    data = data.to(device='cuda', dtype=torch.long)
    model.forward(data)    
    restore_forward(model)

    return count_ops, count_params

def measure_bit_prob_layer_float(data, bitwidth):
    data = cp.asarray(data.cpu(), dtype=cp.float32)
    data_shape = data.shape

    data = data.reshape(-1)
    # float_to_int: copy data in memory for float value x to memory for int value y
    float_to_int = cp.RawKernel(r'''
        extern "C" __global__
        void float_to_int(const float * x, unsigned int * y) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            memcpy(&(y[tid]), &(x[tid]), sizeof(float));
        }
    ''', 'float_to_int')
    x_p = cp.zeros(data.shape, dtype=cp.uint32)
    float_to_int(data.shape, (1, ), (data, x_p))

    bit_prob = []
    for i in range(bitwidth):
        mask = 1 << (bitwidth - i - 1)
        mask = cp.asarray(mask, dtype=cp.uint32)
        d = x_p & mask
        p = cp.mean(d != 0)
        bit_prob.append(p)

    return bit_prob

def measure_bit_probability_float(model, layer_idx, bitwidth=32):
    bit_prob = {}
    for i, layer in enumerate(model.modules()):
        if i not in layer_idx:
            continue
        bit_prob[i] = measure_bit_prob_layer_float(layer.weight.data, bitwidth)

    return bit_prob

def measure_bit_prob_layer_quan(data, bitwidth=8):
    # Create flattened 1d array out of data
    data = cp.asarray(data.cpu(), dtype=cp.int8)
    data_shape = data.shape
    data = data.reshape(-1) 
    
    bit_prob = []
    for i in range(bitwidth):
        mask = 1 << (bitwidth - i - 1)
        mask = cp.asarray(mask, dtype=cp.int8)
        d = data & mask
        p = cp.mean(d != 0)
        bit_prob.append(p)

    return bit_prob

# Old function def takes 4 arguments but 5 were given in call, so created dummy variable to host 5th argument
# Note: altered function def instead of function call in case unlisted variable is actually important for something not yet understood...
#def measure_bit_probability_quan(model, layer_idx, quantize_bits, centroid_label_dict):
def measure_bit_probability_quan(model, layer_idx, bitwidth, quantize_bits, centroid_label_dict):
    bit_prob = {}
    quantize_layer_bit_dict = {n: b for n, b in zip(layer_idx, quantize_bits)}
    for i, layer in enumerate(model.modules()):
        if i not in layer_idx:
            continue
        this_cl_list = centroid_label_dict[i]
        n_bit = quantize_layer_bit_dict[i]
        bit_prob[i] = measure_bit_prob_layer_quan(this_cl_list[0][1], n_bit)
    return bit_prob


def best_policy_list(policy, protection, bit=None):
    if protection == 'bitmask':
        return best_bitmask_list(policy)
    elif protection == 'topbits':
        return best_topbits_list(policy, bit)
    
def best_bitmask_list(best_policy):
    # For each array in best_policy, turn list of numbers into string-based layer mask for noise addition stage
    return [''.join([str(int(item)) for item in policy.tolist()]) for policy in best_policy]

def best_topbits_list(best_policy, bit):
    # Creates layer mask based off of best_policy for use in noise addition stage
    best_policy_list = []
    for policy in best_policy:
        li = ['1' for index in range(policy)] + ['0' for index in range(bit - policy)]
        li = ''.join(li)
        best_policy_list.append(li)
    return best_policy_list
        
#VDCNN functions
class TupleLoader(Dataset):

    def __init__(self, path=""):
        self.path = path

        self.env = lmdb.open(path, max_readers=4, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def __getitem__(self, i):
        xtxt = list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), np.int32)
        lab = list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), np.int32)[0]
        return xtxt, lab
    
def list_from_bytes(string, dtype=np.int32):
    return np.frombuffer(string, dtype=dtype)