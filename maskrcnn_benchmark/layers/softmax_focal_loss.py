import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None, balance_index=-1, smooth=None, size_average=True):
        super(SoftmaxFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_classes, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_classes
            self.alpha = torch.FloatTensor(alpha).view(self.num_classes, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_classes, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')

        if self.smooth < 0 or self.smooth > 1.0:
            raise ValueError('smooth value should be in [0,1]')

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(-1, 1)

        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != inputs.device:
            alpha = alpha.to(inputs.device)

        logits = F.softmax(inputs, dim=-1)
        idx = targets.cpu().long()

        one_hot_key = torch.FloatTensor(targets.size(0), self.num_classes).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)

        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logits).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
