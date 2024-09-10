# General imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import os

# Internal imports
from lib.utils.quantize import add_noise_quantize_model, quantize_model
from lib.utils.noise import add_noise_float_model

# VDCNN imports
from torch.utils.data import DataLoader, Dataset
from lib.net import VDCNN
import lmdb
from lib.utils.utils import TupleLoader

class Eval:

    def __init__(self, model, test_dataset, bit, n_layers, representation):
        self.model = model
        self.state_dict = model.state_dict()
        self.test_dataset = test_dataset
        self.bit = bit

        # Select layers based on arg. VDCNN has 3 linear, 12 conv1d, 11 batchnorm1d layers.
        if n_layers == 3:
            self.layer_types = [nn.Linear]
        elif n_layers == 12:
            self.layer_types = [nn.Conv1d]
        elif n_layers == 15:
            self.layer_types = [nn.Linear, nn.Conv1d] 
        elif n_layers == 26:
            self.layer_types = [nn.Linear, nn.Conv1d, nn.BatchNorm1d]
            
        self.build_index()
        if representation == 'fixed':
            self.quantize_bits = [bit] * len(self.layer_idx)
            self.centroid_label_dict = quantize_model(self.model, self.layer_idx, self.quantize_bits)

    def build_index(self):
        # Create list of indexes for layers subject to noise addition
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
    
def make_baseline(bit, ratio, n_layer):
    '''
    Creates baseline protection mask based
    
    Parameters:
    - bit (int):
        - number of bits in representation
    - ratio (float):
        - preserve ratio for calculating number of bits to protect
    - n_layer (int):
        - number of layers in model
        
    Returns:
    - mask (list):
        - list of baseline protection masks
    '''
    assert ratio >= 0.0, 'preserve ratio for baseline protection must be greater than or equal to 0'
    assert ratio <= 1.0, 'preserve ratio for baseline protection must be less than or equal to 1'
    
    num_protected = int(bit * ratio)
    
    mask = [''.join(['1' for idx in range(num_protected)]) + ''.join(['0' for idx in range(bit-num_protected)])]
    mask *= n_layer
    return mask
    
def output(values):
    '''
    Prints results in an easily-viewed manner for all values passed in
    
    Parameters:
    - values (list tuples):
        - a list of tuple values for what output to print. for (name, value), Name: Value is printed.
    '''
    for (name,value) in values:
        print('{}: {}'.format(name, value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robust Neural Network Noise Evaluator')
    
    parser.add_argument('--cycles', default='10', type=int, help='number of cycles to run evaluation')
    parser.add_argument('--protection', default=None, type=str, help='specifies type of protection technique for mask selection', choices={'topbits', 'bitmask', 'baseline'})
    parser.add_argument('--preserve_ratio', default=0.25, type=float, help='preserve ratio to use when computing baseline performance (default: 0.25)')
    parser.add_argument('--code', default=None, type=str, help='type of ecc in place (bch or ideal shannon)', choices={'bch', 'ideal'})
    parser.add_argument('--representation', default=None, type=str, help='binary representation of weights', choices={'fixed', 'float'})
    parser.add_argument('--bit', default=None, type=int, help='number of bits in weight representation')
    parser.add_argument('--n_layers', default=15, type=int, help='number of layers being subject to noise')
    parser.add_argument('--save', default=None, type=str, help='pass file to save accuracy statistics to')
    parser.add_argument('--ber', default='range', type=str, help='whether to add noise at standard BER (1e-2) or range', choices={'range', 'standard'})
    parser.add_argument('--dataset', default=None, type=str, help='dataset to add noise to', choices={'ag_news', 'sogou_news', 'yelp_polarity'})
    parser.add_argument('--dict', default=None, type=str, help='dataset for which mask to apply (no need to enter if same as dataset arg)', choices={'ag_news', 'sogou_news', 'yelp_polarity'})
    
    args = parser.parse_args()
    
    # Set dict to use if none given
    if args.dict == None:
        args.dict = args.dataset
    
    print('\n\nCurrently testing {}_{}_{}: {} mask on {} model'.format(args.protection, args.representation, args.code, args.dict if args.protection != 'baseline' else 'baseline', args.dataset))
    
    # Set bit arg based on representation
    if args.representation == 'float':
        # Default set to 32 since the floating point representation doesn't have a variable bit length (for our purposes)
        bit = 32
    elif args.representation == 'fixed':
        # Check if bit arg is given for edge cases, but default to 8 since it's all we use
        bit = args.bit if args.bit is not None else 8

    # Variable creation
    seed = 7
    lr = 1e-3
    epoch = 100
    #batch_size = 64
    batch_size = 4
    num_workers = 16
    acc_list, ber_list, med_list = [], [], []
    df_stats = pd.DataFrame(columns=['BER', 'Accuracy'])
    df_stats.set_index('BER', inplace=True)
    mask_dict = {}
    
    if args.ber == 'range':
        prob_list = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
        #prob_list = [1e-9, 3e-9, 5e-9, 7e-9, 9e-9, 1e-8, 3e-8, 5e-8, 7e-8, 9e-8, 1e-7, 3e-7, 5e-7, 7e-7, 9e-7, 1e-6, 3e-6, 5e-6, 7e-6, 9e-6, 1e-5]
    elif args.ber == 'standard':
        prob_list = [1e-2]
    
    mask_dict['ag_news'] = {  
        ('bitmask', 'float', 'ideal') : \
            ['11110111101000000010000000011010', '11010111000100010100000001010001', '11111111001100001000000000000000', '11111111000100000010000001000000', '11111111100101000110111000000000', '11111111101000000010010000001000', '11111111000100000010010000000000', '11111111100100000010111000000000', '11111111100100000010011000000000', '11111111100100000010010000000000', '11111111100100000010011000000000', '11111111100100000010011000000000', '01010111000100001000000000000000', '01010111000100001000100000000000', '01111111010100001110000011010001'], 
        ('bitmask', 'float', 'bch') : \
            ['11111111100001100000010000001110', '11111111100101001100100000000111', '11111111100001101100000000000111', '11111111100111000000000000000110', '11111111100111001100000010000001', '11111111100011101100010000000011', '11111111100111001100000010000111', '11111111100011001100000010000011', '11111111100011001100000000000011', '11111111100011001100000010000111', '11111111100011001100000010000011', '11111111100011001100000000000001', '01001111000011000000000000000000', '01011101000001000000001000000000', '11111111101111000000110000100010'],
        ('bitmask', 'fixed', 'ideal') : \
            ['11110000', '11110000', '11100000', '11110000', '11110000', '11100000', '11010000', '11100000', '11000000', '11000000', '11000000', '11000000', '00000000', '00000000', '00000010'],  
        ('bitmask', 'fixed', 'bch') : \
            ['11001001', '11101101', '11101000', '11101000', '11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '01100000', '00000000', '00000000', '10101101'],
        ('topbits', 'float', 'ideal') : \
            ['11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000', '11111110000000000000000000000000', '11111111000000000000000000000000', '11111111000000000000000000000000'], 
        ('topbits', 'float', 'bch') : \
        ['11111111110000000000000000000000', '11111111110000000000000000000000', '11111111111000000000000000000000', '11111111110000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111100000000000000000000000', '11111111110000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111110000000000000000000000', '11111111000000000000000000000000', '11111110000000000000000000000000', '11100000000000000000000000000000'],
        ('topbits', 'fixed', 'ideal') : \
            ['11111000', '11111000', '11111000', '11111000', '11111000', '11111000', '11111000', '11111000', '11111000', '11111000', '11111000', '11111000', '11000000', '00000000', '00000000'],
        ('topbits', 'fixed', 'bch') : \
            ['11111110', '11111100', '11111110', '11111000', '11111110', '11111110', '11110000', '11111000', '11111000', '11100000', '11100000', '11100000', '10000000', '00000000', '11111100'],
    }
    
    mask_dict['sogou_news'] = {
        ('bitmask', 'float', 'ideal') : \
            #['11101111100110110000000011100000', '11111111110101110101110110000111', '11001111110000110001100110000001', '11111111110101110011100110010011', '11111111110000000101110110011011', '11111111100001010001101110010001', '11111111110100000011000111000000', '11111111100000000001100100000000', '11111111100110000110100100000000', '11110111100110000111000111000000', '11110111100000000101000000000000', '11100111100000000000000000000000', '11100111100000000000000001000000', '11000111100000000000000001000000', '11100111100010000010000110000000'],
            ['11111111000000000001000000000000', '11111111001000100000010000000001', '11111111000001001000000000000001', '11111111001101001000000000000001', '11111111000001001000000010000001', '11111111000100001000000010000001', '11111111000100001000000010000001', '11111111000000001000000010000001', '11101111000100001001000010000001', '11111111000000000000000010001001', '11111111000000001000000010000001', '11101111000000000000000010000001', '11011110000101000000000000000000', '11011110000000000000000010000000', '11011110000100010101001010001001'],
        ('bitmask', 'float', 'bch') : \
            ['11110111001110100000111011011011', '11111111001010100001110011001011', '11111111000000001000000000000001', '11111111010000000100000000000000', '11111111100011000101110000001000', '11111111000010100000000000000001', '11111111000101000000001010000001', '11111111001000100000000000000000', '11111111100011100100001000001000', '11111111001010100000000010000010', '11111111000010000000000000000000', '11111111000000100000000000000000', '11110111000000000000000000000000', '11111111000000000000000010000000', '11110111000000000000010010000010'],
        ('bitmask', 'fixed', 'ideal') : \
            ['11100100', '11000100', '11100100', '11000101', '11100100', '11100110', '11000101', '11100100', '11100110', '11000100', '11000110', '10000000', '00000000', '00000000', '10000000'],
        ('bitmask', 'fixed', 'bch') : \
            ['11010110', '11110100', '11110000', '11110100', '11110000', '11100000', '11110000', '11110000', '11100000', '11110100', '11110000', '01100000', '00000000', '00000000', '00010000'], 
        ('topbits', 'float', 'ideal') : \
            ['11111111110000000000000000000000', '11111111111000000000000000000000', '11111111110000000000000000000000', '11111111110000000000000000000000', '11111111111000000000000000000000', '11111111100000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111100000000000000000000000', '11111111111000000000000000000000', '11111111000000000000000000000000', '11111110000000000000000000000000', '11111111000000000000000000000000', '11111111000000000000000000000000'],
        ('topbits', 'float', 'bch') : \
            ['11111111110000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111111100000000000000000000', '11111111111110000000000000000000', '11111111110000000000000000000000', '11111111111100000000000000000000', '11111111110000000000000000000000', '11111111110000000000000000000000', '11111111111100000000000000000000', '11111111111100000000000000000000', '11111111111100000000000000000000', '11111111000000000000000000000000', '11111100000000000000000000000000', '11111111111100000000000000000000'],
        ('topbits', 'fixed', 'ideal') : \
            ['11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '10000000', '10000000', '11111110'],
        ('topbits', 'fixed', 'bch') : \
            ['11111100', '11111100', '11111100', '11111100', '11111100', '11111100', '11111100', '11111100', '11111100', '11111100', '11111100', '11111100', '10000000', '11000000', '11111000']
    }
    
    mask_dict['yelp_polarity'] = {
        ('bitmask', 'float', 'ideal') : \
            ['11101111111010101101011010000000', '11101111111001101101101001110000', '11101111111000100111101001000100', '11101111111000100101000000010000', '11101111111000100100001011010100', '11101111111000100101011001000000', '11101111110000100001000000010000', '11101111111001100100001000000000', '11101111111010100101001000000000', '11101111110001100101001000000000', '11101111111011101101001010000000', '11101111111011101111001010000000', '01100111100000000000000000001000', '01000101101000010000000000000000', '01101111111000000001000100001001'],
        ('bitmask', 'float', 'bch') : \
            ['11111111000100001000101000000110', '11111111011000000001110010010101', '11101111110000000001100000000111', '11111111001000001000100000000111', '11111111110000000000101000000101', '11111111011000000001001000000101', '11111111001000001100100000000111', '11111111110000000000101000000101', '11111111000000001001001000000101', '11111111000000000000100000000000', '11111111000000000000001000000100', '11111110001000000000001000000000', '01111110000000000001001000000000', '11111110000000000000000000000000', '11111111000001000001000010000000'],
        ('bitmask', 'fixed', 'ideal') : \
            ['11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '11100000', '11000000', '00000000', '00000000', '10000000'],
        ('bitmask', 'fixed', 'bch') : \
            ['11101000', '11001000', '11101000', '11101000', '11101000', '11101000', '11101000', '11101000', '11101000', '01101000', '11101000', '01101000', '00000000', '00000000', '01101000'], 
        ('topbits', 'float', 'ideal') : \
            ['11111111111000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111100000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111111000000000000000000000', '11111111110000000000000000000000', '11111111100000000000000000000000', '11111111111000000000000000000000', '11111111110000000000000000000000', '11111110000000000000000000000000', '11111111100000000000000000000000', '11111111100000000000000000000000'],
        ('topbits', 'float', 'bch') : \
            ['11111111111110000000000000000000', '11111111111110000000000000000000', '11111111111100000000000000000000', '11111111100000000000000000000000', '11111111110000000000000000000000', '11111111111110000000000000000000', '11111111111110000000000000000000', '11111111110000000000000000000000', '11111111111100000000000000000000', '11111111111110000000000000000000', '11111111111110000000000000000000', '11111111111100000000000000000000', '11111110000000000000000000000000', '11111111000000000000000000000000', '11111111000000000000000000000000'],
        ('topbits', 'fixed', 'ideal') : \
            ['11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '00000000', '00000000', '11000000'],
        ('topbits', 'fixed', 'bch') : \
            ['11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '11111111', '00000000', '00000000', '11110000']
    }
    
    # Set seed so that actions taken will be reproducible 
    if seed > 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Apply noise for each BER 
    print("Testing auxiliary models...")
    for prob in prob_list:
        # Create fresh list for every ber
        acc_list_internal = []
        
        # Completes noise for specified number of cycles & provides statistics
        for index in range(1,args.cycles + 1):
            print('Cycle {} of {}...'.format(index, args.cycles))
            # setup model and load parameters
            if args.dataset == 'ag_news':
                n_classes = 4
            elif args.dataset == 'sogou_news':
                n_classes = 5
            elif args.dataset == 'yelp_polarity':
                n_classes = 2
            model = VDCNN(n_classes=n_classes, num_embedding=69, embedding_dim=16, depth=9, n_fc_neurons=2048, shortcut=True)
            #model.load_state_dict(torch.load(os.path.join('results', 'models', 'vdcnn', args.dataset, 'model_epoch_100')))
            model.load_state_dict(torch.load(os.path.join('results', 'models', 'vdcnn', 'ag_news_aux', 'ag_news_0.01_0.25_0.9')))

            model = torch.nn.DataParallel(model).cuda()

            # load test data
            te_path = os.path.join('datasets', args.dataset, 'vdcnn', 'test.lmdb')
            test_dataset = DataLoader(TupleLoader(te_path), batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
            e = Eval(model, test_dataset, bit, args.n_layers, args.representation)
            '''
                mask: a list of bitmask for each layer
                        each bit mask represent as a integer 
                        (1,1,1,1,0,1,1,0) -> 246
                code: None represent ideal code approaching shannon capacity,
                        BCH code (n, k, t): (8191, 6722, 115)
            '''
            
            # Set protection mask from mask dict
            #if all([args.protection, args.representation, args.code]) # Ensures all variables are not None
            if args.protection is not None and args.representation is not None and args.code is not None:
                if args.protection == 'baseline':
                    mask = make_baseline(bit, args.preserve_ratio, args.n_layers)
                else:
                    mask = mask_dict[args.dict][(args.protection, args.representation, args.code)]                    
            else:
                # Used for full-network noise addition w/o protection. For layerwise protection, use noise_prob_layer.py
                if args.representation == 'float':
                    mask = ['00000000000000000000000000000000'] * len(e.layer_idx)
                elif args.representation == 'fixed':
                    # Creates a 0-full mask of length bit for each layer
                    mask = [''.join(['0' for i in range(bit)])] * len(e.layer_idx)
                    
            # Set code type being used
            if args.code == 'ideal' or args.code is None:
                code = None
            elif args.code == 'bch':
                if args.representation == 'float':
                    code = (8191, 6722, 115)
                elif args.representation == 'fixed':
                    code = (8191, 6787, 110)
                
            # Set parameters & apply noise
            mask = convert_mask(mask)
            noise = 'Random'
            accuracy = e.eval_noisy_model(epoch, mask, noise, prob, code)
            acc_list_internal.append(accuracy)

            # Gather statistics
            avg_acc = sum(acc_list_internal) / len(acc_list_internal)
            if args.cycles > 1: # Only need statistics if more than one cycle is performed
                min_acc = min(acc_list_internal)
                max_acc = max(acc_list_internal)
                range_acc = max_acc - min_acc
                med_acc = np.median(acc_list_internal)

        # Add results to statistics dataframe
        df_stats.loc[prob, 'Accuracy'] = str(acc_list_internal)

        # Output average. If multiple runs, then output more in-depth statistics
        output([('BER', prob), ('Avg', avg_acc)])
        if args.cycles > 1: 
            output([('Min', min_acc), ('Max', max_acc), ('Range', range_acc), ('Median Accuracy', med_acc)])
            med_list.append(med_acc)

        # Only create summary stats if there are multiple probs to summarize
        if len(prob_list) > 1:
            # Add ber results to running lists
            acc_list.append(avg_acc)
            ber_list.append(prob)

    # Only outputs summary stats if there are multiple values to report. Else the stats were printed out already
    if len(prob_list) > 1:
        output([('Average Accuracy', acc_list), ('BER', ber_list)])

        # If multiple runs, then output the median values
        if args.cycles > 1:
            output([('Median Accuracy', med_list)])
    
    # Save the results to a specified file
    if args.save is not None:
        df_stats.to_csv(args.save)