import math

import torch.nn as nn
import numpy as np

from lib.utils.utils import measure_model, prGreen, prCyan 
from lib.utils.quantize import add_noise_quantize_model, quantize_model
from lib.utils.noise import add_noise_float_model
from .ibs_env import ImportantBitsEnv

class ImportantBitsFirstNEnv(ImportantBitsEnv):

    def __init__(self, model, preprocess, data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=8, is_model_pruned=False, code=None):
        self.episode = 0
        super(ImportantBitsFirstNEnv, self).__init__(model, preprocess,data, data_root, compress_ratio, args, batch_size, num_workers, bitwidth, is_model_pruned, code)

        self.beta = 1.0
        self.n_actions = 1 

    def reset(self):
        self.load_states()
        self.cur_index = 0
        self.episode += 1
        # self.beta = min(max((self.episode-500.0)/500.0, 0.0), 0.95) + 0.05
        self.beta = 1.0
        self.strategy = []
        observation = self.layer_embedding[0].copy()
        return observation

    def reward(self, acc, w_size_ratio=None):
        size_loss = 0.0
        r = w_size_ratio / self.compress_ratio
        delta = w_size_ratio - self.compress_ratio
        if w_size_ratio > self.compress_ratio:
            # size_loss = math.log(r) * 1.0
            size_loss = math.log(1.0 + delta*32) / 2.0 * self.beta
            # size_loss = (r - 1.0) * 0.5 + 0.0 
            # size_loss = - self.beta * delta * 10
            # size_loss = (16 * self.beta * delta) ** 2
            # size_loss = (16 * delta) ** 2
        else:
            # size_loss = (3 * self.beta * delta) ** 2
            # size_loss = (4 * delta) ** 2
            size_loss = - delta * 5 * self.beta
            # size_loss = 0.0

        acc_loss = acc - self.acc_origin

        # return (acc_loss - self.gamma * size_loss) * self.alpha
        # acc_loss = acc - self.acc_origin
        # if self.data == 'MNIST':
        #     acc_threshold = -0.005
        # elif self.data == 'Cifar10':
        #     acc_threshold = -0.05
        # elif self.data == 'ImageNet':
        #     acc_threshold = -0.1
        # else:
        #     acc_threshold = 0

        # if acc_loss < acc_threshold:
        #     acc_loss = 0.5 * acc_loss - 0.5 * (1 + acc_threshold)
        # else:
        #     acc_loss = 0.5 / acc_threshold  * acc_loss

        # if w_size_ratio is not None:
        #     return (acc_loss + 1. / w_size_ratio) * self.alpha
        return acc_loss  * self.alpha

    def _final_action_wall(self):
        target = self.compress_ratio * self._org_weight()
        min_weight = 0
        for i, n_bit in enumerate(self.strategy):
            min_weight += self.wsize_list[i] * self.min_bit
        while min_weight < self._cur_weight() and target < self._cur_weight():
            for i, n_bit in enumerate(reversed(self.strategy)):
                if n_bit > self.min_bit:
                    self.strategy[-(i+1)] -= 1
                if target >= self._cur_weight():
                    break
        print('=> Final action list: {}'.format(self.strategy))

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_index
        # limit the action to certain range
        action = float(action)
        min_bit, max_bit = self.bound_list[self.cur_index]
        lbound, rbound = min_bit - 0.5, max_bit + 0.5  # same stride length for each bit
        action = (rbound - lbound) * action + lbound
        action = min(max(int(np.round(action, 0)), min_bit), max_bit)
        action = int(action)
        self.last_action = action
        return action  # not constrained here

    def _cur_weight(self):
        cur_weight = 0.
        # quantized
        for i, n_bit in enumerate(self.strategy):
            cur_weight += n_bit * self.wsize_list[i]
        return cur_weight

    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        #measure_model(self.model_for_measure, 3, 224, 224)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        for i, ind in enumerate(self.layer_idx):
            m = module_list[ind+1]
            this_state = []
            if isinstance(m, nn.Conv2d):
                # print('layer {}: {}'.format(i, int(m.in_channels == m.groups)))
                this_state.append([int(m.in_channels == m.groups)])  # layer type, 1 for conv_dw
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([1])  # input feature_map_size
            elif isinstance(m, nn.Linear):
                this_state.append([0.])  # layer type, 0 for fc
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([2])  # input feature_map_size
            elif isinstance(m, nn.Embedding):
                this_state.append([2.])  # layer type, 2 for embedding
                this_state.append([m.num_embeddings])  # in channels (vocab size)
                this_state.append([m.embedding_dim])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([0.])  # input feature_map_size (not applicable)
            elif isinstance(m, nn.LayerNorm):
                this_state.append([3.])  # layer type, 3 for LayerNorm
                this_state.append([m.normalized_shape[0]])  # in channels
                this_state.append([m.normalized_shape[0]])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([0.])  # input feature_map_size (not applicable)

            this_state.append([i])  # index
            this_state.append([float(self.max_bit)])  # bits
            layer_embedding.append(np.hstack(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding


class ImportantBitsFloatFirstNEnv(ImportantBitsFirstNEnv):

    def __init__(self, model, preprocess , data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=32, is_model_pruned=False, code=None):
        super(ImportantBitsFloatFirstNEnv, self).__init__(model, preprocess, data, data_root, compress_ratio, args, batch_size, num_workers, bitwidth, is_model_pruned, code)

    def reward(self, acc, w_size_ratio=None):
        size_loss = 0.0
        r = w_size_ratio / self.compress_ratio
        # delta = w_size_ratio - self.compress_ratio
        # if w_size_ratio > self.compress_ratio:
        #     # size_loss = math.log(1.0 + delta*64) * self.beta
        #     size_loss = 2 * (sigmoid(delta, a=5) - 0.5)
        # else:
        #     size_loss = 2 * (sigmoid(-delta, a=1) - 0.5) 

        acc_loss = acc - self.acc_origin

        # return (acc_loss - self.gamma * size_loss) * self.alpha
        return acc_loss * self.alpha


    def step(self, action):
        action = self._action_wall(action)

        self.strategy.append(action)

        if self._is_final_layer():
            self._final_action_wall()
            assert len(self.strategy) == len(self.layer_idx)
            w_size = self._cur_weight()
            w_size_ratio = w_size / self._org_weight()

            # add noise
            mask = []
            for i, a in enumerate(self.strategy):
                m = ((1 << a) - 1) << (self.bitwidth - a)
                mask.append(m)
            acc = 0.0
            for _ in range(self.epoch):
                add_noise_float_model(self.model, mask, self.noise, self.prob, self.layer_idx, code=self.code)
                acc += self.clip_eval_imagenetv2()
            acc = acc / self.epoch
            reward = self.reward(acc, w_size_ratio)
            
            info_set = {'w_ratio': w_size_ratio, 'accuracy': acc, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    self.best_reward, acc, w_size_ratio))
            else:
                prCyan('Policy: reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    reward, acc, w_size_ratio))

            obs = self.layer_embedding[self.cur_index, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set

        w_size = self._cur_weight()
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_index += 1  # the index of next layer
        self.layer_embedding[self.cur_index][-1] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_index, :].copy()
        return obs, reward, done, info_set


class ImportantBitsQuanFirstNEnv(ImportantBitsFirstNEnv):

    def __init__(self, model, preprocess, data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=8, is_model_pruned=False, code=None):
        super(ImportantBitsQuanFirstNEnv, self).__init__(model, preprocess, data, data_root, compress_ratio, args, batch_size, num_workers, bitwidth, is_model_pruned, code)
        self.quantize_bits = [bitwidth] * self.n_layer
        self.centroid_label_dict = quantize_model(self.model, self.layer_idx, self.quantize_bits)

    def step(self, action):
        action = self._action_wall(action)

        self.strategy.append(action)

        if self._is_final_layer():
            self._final_action_wall()
            assert len(self.strategy) == len(self.layer_idx)
            w_size = self._cur_weight()
            w_size_ratio = w_size / self._org_weight()

            # add noise
            mask = []
            for i, a in enumerate(self.strategy):
                m = ((1 << a) - 1) << (self.max_bit - a)
                mask.append(m)
            acc = 0.0
            for _ in range(self.epoch):
                add_noise_quantize_model(self.model, mask, self.noise, self.prob, self.layer_idx, self.quantize_bits, self.centroid_label_dict, code=self.code)
                acc += self.clip_eval_imagenetv2()
            acc = acc / self.epoch
            reward = self.reward(acc, w_size_ratio)
            
            info_set = {'w_ratio': w_size_ratio, 'accuracy': acc, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    self.best_reward, acc, w_size_ratio))
            else:
                prCyan('Policy: reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    reward, acc, w_size_ratio))

            obs = self.layer_embedding[self.cur_index, :].copy()  # actually the same as the last state
            done = True
            return obs, reward, done, info_set

        w_size = self._cur_weight()
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_index += 1  # the index of next layer
        self.layer_embedding[self.cur_index][-1] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_index, :].copy()
        return obs, reward, done, info_set

