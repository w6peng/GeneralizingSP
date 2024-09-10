# General imports
import math
from copy import deepcopy
import os

# Torch imports
import torch
import torch.nn as nn
#from torchvision import datasets, transforms 

# VDCNN imports
from lib.net import VDCNN
from lib.datasets import load_datasets, AgNews, SoguNews
import numpy as np
from sklearn import utils, metrics
from lib.utils import TupleLoader
from torch.utils.data import DataLoader, Dataset
#from torchtext.datasets import AG_NEWS
import lmdb


class ImportantBitsEnv:

    def __init__(self, model, data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=8, is_model_pruned=True, code=None):
        self.layer_types = [nn.Linear, nn.Conv1d] # Layer types of VDCNN model that have trainable weights

        self.epoch = args.finetune_epoch
        self.cur_index = 0
        self.noise = 'Random'
        self.prob = 1e-2 # For BER of 0.01 in original paper
        self.strategy = []
        self.code = code
        self.alpha = 0.2
        self.gamma = 3.0 # 7.5
        
        self.val_size = args.val_size
        self.num_classes, self.dataset = self._get_dataset(data, data_root, args.arch)
        #self.dataset = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
        #                       shuffle=False, num_workers=num_workers, pin_memory=True)
        self.model = model
        self.model_for_measure = deepcopy(model)
        self.save_states()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.compress_ratio = compress_ratio # preserve_ratio in rl_noise files
        self.is_model_pruned = is_model_pruned
        self.last_action = None
        self.bitwidth = bitwidth
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.n_actions = args.n_actions
        self.data = data

        self.best_reward = -math.inf

        # build indexs
        self._build_index()
        self._get_weight_size()
        self.n_layer = len(self.layer_idx)

        self.acc_origin = self._get_performance()
        self._build_state_embedding(args.arch)

        self.reset()
        print('=> original acc: {:.5f}'.format(self.acc_origin))
        print('=> original #param: {:.4f}, model size: {:.4f} MB'.format(sum(self.wsize_list) * 1. / 1e6,
                                                                         sum(self.wsize_list) * self.bitwidth * 1. / 8e6))

    def reset(self):
        self.load_states()
        self.cur_index = 0
        self.strategy = []
        observation = self.layer_embedding[0].copy()
        return observation

    def step(self, action):
        raise NotImplementedError

    def load_states(self):
        self.model.load_state_dict(self.state_dict)

    def save_states(self):
        self.state_dict = deepcopy(self.model.state_dict())

    def reward(self, acc, w_size_ratio=None):
        raise NotImplementedError

    def _is_final_layer(self):
        return self.cur_index == len(self.layer_idx) - 1

    def _final_action_wall(self):
        raise NotImplementedError

    def _cur_weight(self):
        raise NotImplementedError

    def _org_weight(self):
        # Calculates original weight as the (# params * # bits per param)
        org_weight = 0.
        org_weight += sum(self.wsize_list) * self.bitwidth
        return org_weight

    def _get_dataset(self, data, data_root, arch):
        print("get dataset")
        if data == 'ag_news':
            num_classes = 4
        elif data == 'sogou_news':
            num_classes = 5
        elif data == 'yelp_polarity':
            num_classes = 2
        else:
            raise NotImplementedError('data [%s] is not found'.format(data))

        te_path = os.path.join('datasets', data, 'vdcnn', 'test.lmdb')
        dataset = DataLoader(TupleLoader(te_path), batch_size=128, shuffle=False, num_workers=4, pin_memory=False)

        return num_classes, dataset

    def _get_performance(self):
        # print("get the performance of current model")
        self.model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for data in self.dataset:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                is_nan, _ = torch.max(torch.isnan(outputs), 1)
                is_nan = ~is_nan
                is_nan.byte()

                total += inputs.size(0)
                correct += (preds == labels).sum().item()

                # print(self.criterion(outputs, labels))
        acc = correct / total
        return acc

    def _build_index(self):
        self.layer_idx = []
        self.layer_type_list = []
        self.bound_list = []
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.layer_types:
                self.layer_idx.append(i)
                self.layer_type_list.append(type(m))
                self.bound_list.append((self.min_bit, self.max_bit))
        print('=> Final bound list: {}'.format(self.bound_list))

    def _get_weight_size(self):
        # get the param size for each layers to prune, size expressed in number of params
        self.wsize_list = []
        for i, m in enumerate(self.model.modules()):
            if i in self.layer_idx:
                if not self.is_model_pruned:
                    self.wsize_list.append(m.weight.data.numel())
                else:  # the model is pruned, only consider non-zeros items
                    self.wsize_list.append(torch.sum(m.weight.data.ne(0)))
        self.wsize_dict = {i: s for i, s in zip(self.layer_idx, self.wsize_list)}

    def _build_state_embedding(self):
        raise NotImplementedError