"""import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, margin=0.3, scale=32, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.m = margin
        self.s = scale
        self.easy_margin = easy_margin

    def forward(self, input, target):

        # make a one-hot index
        index = input.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.bool()

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = input[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m  - sin_t * sin_m

        if self.easy_margin:
            cond = F.relu(cos_t)
            keep = cos_t
        else:
            cond_v = cos_t - math.cos(math.pi - self.m)
            cond = F.relu(cond_v)
            keep = cos_t - math.sin(math.pi - self.m) * self.m

        cos_t_add_m = torch.where(cond.bool(), cos_t_add_m, keep)

        output = input * 1.0 #size=(B,Classnum)
        output[index] = cos_t_add_m
        output = self.s * output

        #return F.cross_entropy(output, target)
        loss = F.cross_entropy(output, target)
        
        if torch.isnan(loss):
            print("Loss is NaN. Input:", input, "Target:", target)
            return torch.tensor(0.0)
        
        clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return loss
        
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
        
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features=2048, out_features=702, s=30.0, m=0.50, bias=False):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.weight = Parameter(torch.Tensor(out_features, in_features).cuda())
        if bias:
            self.bias = Parameter(torch.Tensor(out_features).cuda())
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, labels):
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        cosine = F.linear(F.normalize(x), F.normalize(torch.transpose(self.weight, 0, 1)))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        loss = F.cross_entropy(output, labels, reduction='mean')
        return loss

import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim=2048, s=32.0, m=0.3):
        super(ArcFaceLoss, self).__init__()
        self.feat_dim = feat_dim
        self._num_classes = num_classes
        self.s = s
        self.m = m
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.weight = Parameter(torch.Tensor(num_classes, feat_dim).cuda())
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))

    def forward(self, features, targets):
        # get cos(theta)
        #print(features.shape)
        #print(self.weight.shape)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight).t())
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t.to('cuda')
            #self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self.s
        loss = F.cross_entropy(pred_class_logits, targets)
        return loss.mean()"""

import math
import torch
import torch.nn.functional as F
from torch import nn


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.1, scale=16, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.m = margin
        self.s = scale
        self.easy_margin = easy_margin

    def forward(self, input, target):

        # make a one-hot index
        index = input.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.bool()

        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        cos_t = input[index]
        sin_t = torch.sqrt(1.0 - cos_t * cos_t)
        cos_t_add_m = cos_t * cos_m  - sin_t * sin_m

        if self.easy_margin:
            cond = F.relu(cos_t)
            keep = cos_t
        else:
            cond_v = cos_t - math.cos(math.pi - self.m)
            cond = F.relu(cond_v)
            keep = cos_t - math.sin(math.pi - self.m) * self.m

        cos_t_add_m = torch.where(cond.bool(), cos_t_add_m, keep)

        output = input * 1.0 #size=(B,Classnum)
        output[index] = cos_t_add_m
        output = self.s * output

        return F.cross_entropy(output, target)