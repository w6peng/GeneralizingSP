# General imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os

# Interal imports
from lib.utils.quantize import add_noise_quantize_model, quantize_model
from lib.utils.noise import add_noise_float_model

# VDCNN imports
from torch.utils.data import DataLoader, Dataset
from lib.net import VDCNN
import lmdb
from lib.utils import TupleLoader

class Eval:

    def __init__(self, model, test_dataset, bit, n_layers, representation):
        self.model = model
        self.state_dict = model.state_dict()
        self.test_dataset = test_dataset
        self.bit = bit

        self.layer_types = [nn.Conv1d, nn.Linear] 
        self.build_index()
        if representation == 'fixed':
            self.quantize_bits = [bit] * len(self.layer_idx)
            self.centroid_label_dict = quantize_model(self.model, self.layer_idx, self.quantize_bits)

    def build_index(self):
        self.layer_idx = []
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.layer_types:
                self.layer_idx.append(i)

    def eval(self):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for data in self.test_dataset:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
    
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
    
                total += inputs.size(0)
                correct += (preds == labels).sum().item()
    
        acc = correct / total
        return acc

    def eval_noisy_model(self, epoch, mask, noise, prob, code):
        acc_arr = []
        for i in range(epoch):
            self.model.load_state_dict(self.state_dict)
            if self.bit == 32:
                add_noise_float_model(self.model, mask, noise, prob, self.layer_idx, code=code)
            else:
                add_noise_quantize_model(self.model, mask, noise, prob, self.layer_idx, self.quantize_bits, self.centroid_label_dict, code=code)
            acc = self.eval()
            acc_arr.append(acc)
        acc_total = sum(acc_arr) / epoch
        return acc_total


def convert_mask(mask):
    mask_result = []
    for m in mask:
        m_result = 0
        for bit in m:
            m_result <<= 1
            m_result += int(bit)
        mask_result.append(m_result)
    return mask_result
    
def output(values):
    for (name,value) in values:
        print('{}: {}'.format(name, value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robust Neural Network Noise Evaluator')
    
    parser.add_argument('--cycles', default='10', type=int, help='number of cycles to run evaluation')
    parser.add_argument('--representation', default=None, type=str, help='binary representation of weights', choices=['fixed', 'float'])
    parser.add_argument('--bit', default=None, type=int, help='number of bits in weight representation')
    parser.add_argument('--n_layers', default=15, type=int, help='number of layers being subject to noise')
    parser.add_argument('--save', default=None, type=str, help='pass file to save accuracy statistics to')
    parser.add_argument('--dataset',default=None, type=str, help='specifies the dataset to apply noise to', choices=['ag_news', 'sogou_news', 'yelp_polarity'])
    args = parser.parse_args()

    seed = 7
    bit = 32
    lr = 1e-3
    epoch = 100
    #batch_size = 64
    batch_size = 4
    num_workers = 16
    acc_list, ber_list, med_list = [], [], []
    df_stats = pd.DataFrame(columns=['BER'])
    df_stats.set_index('BER', inplace=True)
    #prob_list = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    prob_list = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

    # Set seed so that actions taken will be reproducible 
    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create columns of dataframe
    for i in range(15):
        df_stats['layer_{}'.format(i)] = np.nan
    # Add BERs to df
    df_stats.BER = prob_list
    
    # Set bit arg based on representation
    if args.representation == 'float':
        # Default set to 32 since the floating point representation doesn't have a variable bit length (for our purposes)
        bit = 32
    elif args.representation == 'fixed':
        # Check if bit arg is given for edge cases, but default to 8 since it's all we use
        bit = args.bit if args.bit is not None else 8
    
    # setup model and load parameters
    if args.dataset == 'ag_news':
        n_classes = 4
    elif args.dataset == 'sogou_news':
        n_classes = 5
    elif args.dataset == 'yelp_polarity':
        n_classes = 2
    
    model = VDCNN(n_classes=n_classes, num_embedding=69, embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=True)
    model.load_state_dict(torch.load(os.path.join('results', 'models', 'vdcnn', args.dataset, 'model_epoch_100')))

    model = torch.nn.DataParallel(model).cuda()

    # load test data
    te_path = os.path.join('datasets', args.dataset, 'vdcnn', 'test.lmdb')
    test_dataset = DataLoader(TupleLoader(te_path), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    e = Eval(model, test_dataset, bit, args.n_layers, args.representation)
    
    for index, prob in enumerate(prob_list):
        acc_list_internal = []
        
        print('Evaluating weight at: ', prob)
        # Create a noise mask based on position. The i-th layer is left unprotected and all other layers are protected
        for i in range(len(e.layer_idx)):
            layer_acc = []
            print('\tLayer {} of {}'.format(i+1, len(e.layer_idx)))
            # Create mask
            mask = list()
            if i == 0:
                mask.append(''.join(['0' for i in range(bit)]))
                for _ in range(1, len(e.layer_idx)):
                    mask.append(''.join(['1' for i in range(bit)]))
            elif i == len(e.layer_idx) - 1:
                for _ in range(len(e.layer_idx) - 1):
                    mask.append(''.join(['1' for i in range(bit)]))
                mask.append(''.join(['0' for i in range(bit)]))
            else:
                for _ in range(i):
                    mask.append(''.join(['1' for i in range(bit)]))
                mask.append(''.join(['0' for i in range(bit)]))
                for _ in range(i + 1, len(e.layer_idx)):
                    mask.append(''.join(['0' for i in range(bit)]))
        
            mask = convert_mask(mask)
            noise = 'Random'
            code = None
            
            for j in range(1,args.cycles + 1):
                print('\t\tCycle {} of {}'.format(j, args.cycles))
                accuracy = e.eval_noisy_model(epoch, mask, noise, prob, code)
                layer_acc.append(accuracy)

            # Gather statistics
            avg_acc = sum(layer_acc) / len(layer_acc)
            if args.cycles > 1: # Only need statistics if more than one cycle is performed
                min_acc = min(layer_acc)
                max_acc = max(layer_acc)
                range_acc = max_acc - min_acc
                med_acc = np.median(layer_acc)

            # Add stats to statistics dataframe
            df_stats.loc[prob, 'layer_{}'.format(i)] = str(layer_acc)
        
            output([('Layer', i+1), ('BER', prob), ('Avg', avg_acc)])
            if args.cycles > 1: 
                output([('Min', min_acc), ('Max', max_acc), ('Range', range_acc), ('Median Accuracy', med_acc)])
                med_list.append(med_acc)
                
            acc_list_internal.append(layer_acc)
            
                # Save the results to a specified file
            if args.save is not None:
                df_stats.to_csv(args.save)
            
        acc_list.append(avg_acc)
        ber_list.append(prob)


    output([('Average Accuracy', acc_list), ('BER', ber_list)])

    if args.cycles > 1:
        output([('Median Accuracy', med_list)])
    
    # Save the results to a specified file
    if args.save is not None:
        df_stats.to_csv(args.save)
