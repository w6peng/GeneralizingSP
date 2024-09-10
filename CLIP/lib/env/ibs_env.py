import math
from copy import deepcopy

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch
import clip
from tqdm.notebook import tqdm
from pkg_resources import packaging
from imagenetv2_pytorch import ImageNetV2Dataset
from lib.utils.utils import read_list_from_json, accuracy


class ImportantBitsEnv:

    def __init__(self, model, preprocess, data, data_root, compress_ratio, args, batch_size=64, num_workers=16, bitwidth=8, is_model_pruned=False, code=None):
        self.layer_types = [torch.nn.modules.conv.Conv2d, clip.model.LayerNorm,
                            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
                            torch.nn.modules.linear.Linear, torch.nn.modules.sparse.Embedding]

        self.epoch = args.finetune_epoch
        self.cur_index = 0
        self.noise = 'Random'
        self.prob = 1e-2
        self.strategy = []
        self.code = code
        self.alpha = 0.2
        self.gamma = 3.0 #7.5
        self.preprocess = preprocess

        self.num_classes, self.dataset = self._get_dataset(data, data_root)
        self.dataset = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers, pin_memory=True)
        self.model = model.module if isinstance(model, torch.nn.DataParallel) else model
        self.model_for_measure = deepcopy(model)
        self.save_states()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.compress_ratio = compress_ratio
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

        self.classnames = read_list_from_json("imagenet_classes")
        self.templates = read_list_from_json("imagenet_templates")

        self.zeroshot_weights =self.zeroshot_classifier()

        self.acc_origin = self.clip_eval_imagenetv2()
        self._build_state_embedding()

        self.reset()
        print('=> original acc: {:.3f}%'.format(self.acc_origin))
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
        org_weight = 0.
        org_weight += sum(self.wsize_list) * self.bitwidth
        return org_weight

    def _get_dataset(self, data, data_root):
        print("get dataset")
        if data == 'ImageNetV2':
            num_classes = 1000
            dataset = ImageNetV2Dataset(transform=self.preprocess)
            print('ImageNetV2')
            print(len(ImageNetV2Dataset()))
        else:
            raise NotImplementedError('data [%s] is not found'.format(data))

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

    def clip_eval_imagenetv2(self):
        #make image and target to device .to(device)
        # text_conder with noise, add the foloowing line
        self.zeroshot_weights = self.zeroshot_classifier()
        
        with torch.no_grad():
            top1, top5, n = 0., 0., 0.
            for i, (images, target) in enumerate(tqdm(self.dataset, disable = True)):
                images = images.cuda()
                target = target.cuda()
                # predict
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ self.zeroshot_weights
                # measure accuracy
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)
        top1 = (top1 / n)
        top5 = (top5 / n)

        return top5


    def _build_index(self):
        self.layer_idx = []
        self.layer_type_list = []
        self.bound_list = []
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.layer_types:
                self.layer_idx.append(i)
                self.layer_type_list.append(type(m))
                self.bound_list.append((self.min_bit, self.max_bit))
        print('=> Final layer index: {}'.format(self.layer_idx))
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

    def zeroshot_classifier(self):
        #remember make the weights to device use .to(device)

        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(self.classnames, disable = True):
                texts = [template.format(classname) for template in self.templates] #format with class
                texts = clip.tokenize(texts).cuda() #tokenize
                class_embeddings = self.model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        return zeroshot_weights

    def _build_state_embedding(self):
        raise NotImplementedError


