import math

import torch.nn as nn
import numpy as np

from lib.utils.utils import measure_model, prGreen, prCyan, norm_sigmoid, measure_bit_probability_quan, measure_bit_probability_float  
from lib.utils.quantize import add_noise_quantize_model, quantize_model
from lib.utils.noise import add_noise_float_model
from .ibs_env import ImportantBitsEnv

class ImportantBitsSizeEnv(ImportantBitsEnv):

    def __init__(self, model, preprocess,data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=8, is_model_pruned=False, code=None):
        self.episode = 0
        super(ImportantBitsSizeEnv, self).__init__(model,preprocess, data, data_root, compress_ratio, args, batch_size, num_workers, bitwidth, is_model_pruned, code)
        self.threshold = 0.5
        self.beta = 0
        
        self._get_hint()
        #self.mask_hint = mask_hint
        self.n_actions = self.max_bit - self.min_bit 

    def reset(self):
        self.load_states()
        self.cur_index = 0
        self.episode += 1
        self.beta = min(max((self.episode-750.0)/750.0, 0.1), 1.0)
        # self.beta = 2.0
        self.strategy = []
        observation = self.layer_embedding[0].copy()
        return observation

    def reward(self, acc, w_size_ratio=None):
        size_loss = 0.0
        r = w_size_ratio / self.compress_ratio
        delta = w_size_ratio - self.compress_ratio
        print("w_size_ratio is " + str(w_size_ratio))
        print("delta is " + str(delta))
        if w_size_ratio > self.compress_ratio:
            # size_loss = math.log(r) * 1.0
            # size_loss = math.log(1.0 + delta*32) / 2.0 * self.beta
            size_loss = math.log(1.0 + delta*64) * self.beta
            # size_loss = (r - 1.0) * 0.5 + 0.0 
            # size_loss = - self.beta * delta * 10
            # size_loss = (16 * self.beta * delta) ** 2
            # size_loss = (16 * delta) ** 2
        else:
            # size_loss = (3 * self.beta * delta) ** 2
            # size_loss = (4 * delta) ** 2
            # size_loss = - delta * 5 * self.beta
            size_loss = - delta * 10 * self.beta
            # size_loss = 0.0

        acc_loss = acc - self.acc_origin

        return (acc_loss - self.gamma * size_loss) * self.alpha

    def _get_hint(self):
        maximum = (1 << self.bitwidth) - 1 
        performance = []
        for i in range(self.bitwidth):
            j = self.bitwidth - i - 1
            mask = [maximum ^ (1 << j)] * len(self.layer_idx)
            self.load_states()

            self._add_noise(mask)

            performance.append(self.clip_eval_imagenetv2())

            # add_noise_float_model(model, mask, self.noise, self.prob, self.layer_idx, code=code)
        mask_hint_ind = np.argsort(performance)[::-1]
        self.mask_hint = [0] * (self.max_bit - self.min_bit)
        num_bit = int(self.compress_ratio * self.bitwidth)
        
        for ind in mask_hint_ind:
            if sum(self.mask_hint) > num_bit - 1:
                break
            if ind < self.min_bit or ind >= self.max_bit:
                continue
            self.mask_hint[ind-self.min_bit] = 1    
        print(mask_hint_ind, self.mask_hint)
        self.mask_hint = [1, 1, 1, 1, 1, 1, 0, 0]
        
        self.load_states()

    def _add_noise(self, mask):
        raise NotImplementedError

    def _strategy_tostring(self):
        strategy_string = []
        for i, action in enumerate(self.strategy):
            action_string = ''
            for _ in range(self.min_bit):
                action_string += '1'
            for bit in action:
                action_string += str(int(bit))
            for _ in range(self.bitwidth - self.max_bit):
                action_string += '0'
            strategy_string.append(action_string)
        strategy_string = ', '.join(strategy_string)
        return strategy_string

    def _final_action_wall(self):
        # select action by threshold.
        for i in range(len(self.strategy)):
            self.strategy[i] = (self.strategy[i] >= self.threshold).astype(float)

        # select action by threshold and sequential reduction
        # num_bits = []
        # sorted_index_arr = []
        # for i in range(len(self.strategy)):
        #     sorted_index = self.strategy[i].argsort()[::-1]
        #     sorted_index_arr.append(sorted_index)
        #     num_bits.append(np.sum(self.strategy[i] > self.threshold))
        #     one_index, zero_index = np.split(sorted_index, [num_bits[-1]])
        #     self.strategy[i][one_index] = 1 
        #     self.strategy[i][zero_index] = 0 

        # target = self.compress_ratio * self._org_weight()
        # flag = False
        # while target >= self._cur_weight():
        #     for i, n_bit in enumerate(num_bits):
        #         if n_bit < self.max_bit - 1:
        #             num_bits[i] += 1
        #             one_index = sorted_index_arr[i][num_bits[i]]
        #             self.strategy[i][one_index] = 1
        #             if target < self._cur_weight():
        #                 self.strategy[i][one_index] = 0
        #                 flag = True
        #                 break
        #     if flag:
        #         break

        # while target < self._cur_weight():
        #     for i, n_bit in enumerate(reversed(num_bits)):
        #         if n_bit > self.min_bit:
        #             num_bits[-(i+1)] -= 1
        #             zero_index = sorted_index_arr[-(i+1)][num_bits[-(i+1)]]
        #             self.strategy[-(i+1)][zero_index] = 0
        #         if target >= self._cur_weight():
        #             break
        print('=> Final action list: {}'.format(self._strategy_tostring()))

    def _cur_weight(self):
        cur_weight = 0.
        # quantized
        for i, action in enumerate(self.strategy):
            cur_weight += action.sum() * self.wsize_list[i]
        cur_weight += sum(self.wsize_list) * self.min_bit
        return cur_weight

    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        #measure_model(self.model_for_measure, 3, 224, 224)
        # build the static part of the state embedding
        bit_prob = self.get_bit_prob()
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
            a= []
            for i in range(self.min_bit, self.max_bit):
                x = bit_prob[ind][i].tolist()
                a.append(x)
            #this_state.append(bit_prob[ind][self.min_bit: self.max_bit])
            this_state.append(a)
            this_state.append([1.] * (self.max_bit - self.min_bit))  # bits

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


class ImportantBitsQuanSizeEnv(ImportantBitsSizeEnv):

    def __init__(self, model,preprocess, data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=8, is_model_pruned=False, code=None):
        super(ImportantBitsQuanSizeEnv, self).__init__(model,preprocess, data, data_root, compress_ratio, args, batch_size, num_workers, bitwidth, is_model_pruned, code)
        # self.threshold = 1 - compress_ratio

        self.quantize_bits = [bitwidth] * self.n_layer
        self.centroid_label_dict = quantize_model(self.model, self.layer_idx, self.quantize_bits)

    def reward(self, acc, w_size_ratio=None):
        size_loss = 0.0
        r = w_size_ratio / self.compress_ratio
        delta = w_size_ratio - self.compress_ratio
        if w_size_ratio > self.compress_ratio:
            # size_loss = math.log(1.0 + delta*64) * self.beta
            size_loss = norm_sigmoid(delta, a=2)
        else:
            size_loss = norm_sigmoid(delta, a=1) / 2.0 

        acc_loss = acc - self.acc_origin
        acc_loss = norm_sigmoid(acc_loss, a=2)

        return (acc_loss - self.gamma * size_loss * self. beta) * self.alpha

    def step(self, action):
        self.strategy.append(action)

        if self._is_final_layer():
            self._final_action_wall()
            assert len(self.strategy) == len(self.layer_idx)
            w_size = self._cur_weight()
            w_size_ratio = w_size / self._org_weight()

            # add noise
            mask = []
            for i, mask_arr in enumerate(self.strategy):
                m = 0
                for elem in mask_arr:
                    m = m << 1
                    m += int(elem) 
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
        self.layer_embedding[self.cur_index][-self.n_actions:] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_index, :].copy()
        return obs, reward, done, info_set

    def get_bit_prob(self):
        self.quantize_bits = [self.bitwidth] * self.n_layer
        self.centroid_label_dict = quantize_model(self.model, self.layer_idx, self.quantize_bits)
        return measure_bit_probability_quan(self.model, self.layer_idx, self.bitwidth, self.quantize_bits, self.centroid_label_dict) 

    def _add_noise(self, mask):
        add_noise_quantize_model(self.model, mask, self.noise, self.prob, self.layer_idx, self.quantize_bits, self.centroid_label_dict, code=self.code)


class ImportantBitsFloatSizeEnv(ImportantBitsSizeEnv):

    def __init__(self, model, preprocess,data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=32, is_model_pruned=False, code=None):
        super(ImportantBitsFloatSizeEnv, self).__init__(model,preprocess, data, data_root, compress_ratio, args, batch_size, num_workers, bitwidth, is_model_pruned, code)

    def reward(self, acc, w_size_ratio=None):
        size_loss = 0.0
        r = w_size_ratio / self.compress_ratio
        delta = w_size_ratio - self.compress_ratio
        if w_size_ratio > self.compress_ratio:
            # size_loss = math.log(1.0 + delta*64) * self.beta
            #size_loss = norm_sigmoid(delta, a=2)
            size_loss = delta
        else:
            #size_loss = norm_sigmoid(delta, a=2) / 1.5 
            size_loss = - delta * 0.5

        acc_loss = acc - self.acc_origin

        return (acc_loss - self.gamma * size_loss * self.beta) * self.alpha
        # return acc_loss * self.alpha

    def step(self, action):
        self.strategy.append(action)

        if self._is_final_layer():
            self._final_action_wall()
            assert len(self.strategy) == len(self.layer_idx)
            w_size = self._cur_weight()
            w_size_ratio = w_size / self._org_weight()


            # add noise
            mask = []
            for i, mask_arr in enumerate(self.strategy):
                m = 0
                for _ in range(self.min_bit):
                    m = m << 1
                    m += 1
                for elem in mask_arr:
                    m = m << 1
                    m += int(elem) 
                m = m << int(self.bitwidth - self.max_bit)
                mask.append(m)
            acc_total = 0.0
            for _ in range(self.epoch):
                add_noise_float_model(self.model, mask, self.noise, self.prob, self.layer_idx, code=self.code)
                acc = self.clip_eval_imagenetv2()
                acc_total += acc
            acc = acc_total / self.epoch
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
        self.layer_embedding[self.cur_index][-self.n_actions:] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_index, :].copy()
        return obs, reward, done, info_set

    def get_bit_prob(self):
        return measure_bit_probability_float(self.model, self.layer_idx , self.bitwidth) 

    def _add_noise(self, mask):
        add_noise_float_model(self.model, mask, self.noise, self.prob, self.layer_idx, code=self.code)

