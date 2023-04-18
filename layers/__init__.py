import torch.nn.functional as F
import torch
import pdb
from .triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLoss
from .arcface_loss import ArcFaceLoss
from .circle_loss import CircleLoss
from .cosface_loss import CosFaceLoss
from .contrastive_loss import ContrastiveLoss
from .center_loss import CenterLoss
from .centroid_triplet import CentroidTripletLoss




def make_loss(cfg):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(margin=cfg.SOLVER.MARGIN)  # triplet loss
    cross_entropy = CrossEntropyLoss(num_classes=cfg.SOLVER.CLASSNUM,epsilon=cfg.SOLVER.SMOOTH)
    arc_face = ArcFaceLoss(cfg.SOLVER.MARGIN)
    circlel = CircleLoss(cfg.SOLVER.MARGIN)
    contra = ContrastiveLoss(num_classes=cfg.SOLVER.CLASSNUM)
    cosf = CosFaceLoss(cfg.SOLVER.MARGIN)
    cent = CenterLoss(num_classes=cfg.SOLVER.CLASSNUM)
    centroid = CentroidTripletLoss(cfg.SOLVER.MARGIN)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return cosf(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            #loss_id = cross_entropy(score, target) + triplet(feat, target)[0]
            loss_id = cosf(score, target) + cross_entropy(score, target)
            # cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) #+PairwiseConfusion(score)/100.0
            return loss_id
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func
