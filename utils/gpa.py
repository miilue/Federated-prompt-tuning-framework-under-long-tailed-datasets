import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from functools import partial
from torch.nn.modules.loss import _Loss
import random



class Gradient_adaptive_Prompt_Adjuster(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_sigmoid = True
        self.weight = []
        self.num_classes = args.num_classes

        # self.vis_grad = vis_grad
        self.varphi = args.varphi
        self.device = args.device
        self.sigma = 12.0
        self.mu = 0.8
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes) * 100)
        self.register_buffer('pn_diff', torch.zeros(self.num_classes))
        self.pos_grad = self.pos_grad.to(args.device)
        self.neg_grad = self.neg_grad.to(args.device)
        self.pos_neg = self.pos_neg.to(args.device)
        self.pn_diff = self.pn_diff.to(args.device)
        self.pos_neg_head_list = []
        self.pos_neg_mid_list = []
        self.pos_neg_tail_list = []

        def _func(x, sigma, mu):
            return 1 / (1 + torch.exp(-1 * (x - mu) * sigma))

        self.map_func = partial(_func, sigma=self.sigma, mu=self.mu)

    def forward(self, cls_score, label):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        self.target = expand_label(cls_score, label)

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * self.target + neg_w * (1 - self.target)
        self.weight = weight

        cls_loss = F.binary_cross_entropy_with_logits(cls_score, self.target, reduction='none')
        hook_handle = cls_score.register_hook(self.hook_func)
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        return cls_loss

    def get_weight(self, cls_score):
        mu = torch.mean(self.pos_neg)
        sigma = torch.std(self.pos_neg)
        neg_w = self.map_func(self.pos_neg)
        pos_w = self.varphi * (1 - neg_w) + 1
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

    def hook_func(self, grad):
        target_temp = self.target.detach()
        grad_temp = grad.detach()
        grad_temp *= self.weight
        grad_temp = torch.abs(grad_temp)

        pos_grad = torch.sum(grad_temp * target_temp, dim=0)
        neg_grad = torch.sum(grad_temp * (1 - target_temp), dim=0)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def reset_pos_neg_list(self):
        print('pos_neg_head_list:', self.pos_neg_head_list)
        print('pos_neg_mid_list:', self.pos_neg_mid_list)
        print('pos_neg_tail_list:', self.pos_neg_tail_list)
        self.pos_neg_head_list = []
        self.pos_neg_mid_list = []
        self.pos_neg_tail_list = []

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    def standardization(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma


