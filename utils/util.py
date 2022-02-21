from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
import random
import os
import argparse
from torch.optim.lr_scheduler import _LRScheduler


def get_stamp(args):
    basic_stamp = 'lr:{lr} ep:{epochs} wd:{weight_decay} gamma:{lr_decay_gamma} bs:{batch_size} warm:{warm} seed:{seed}'.format(
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        lr_decay_gamma=args.lr_decay_gamma,
        batch_size=args.batch_size,
        warm=args.warm,
        seed=args.seed)

    dataset_stamp = 'nlbl:{num_labeled_classes} nulbl:{num_unlabeled_classes}'.format(
        num_labeled_classes=args.num_labeled_classes,
        num_unlabeled_classes=args.num_unlabeled_classes)

    model_stamp = 'ft {feature_extractor} knn:{knn_graph_k} dist:{knn_dist} temp:{nce_temperature} wtak:{wta_k} innce:{instance_nce} stnce:{structure_nce} recon:{reconstruct} modul:{modularity} bce:{bce}'.format(
        feature_extractor=args.feature_extractor,
        knn_graph_k=args.knn_graph_k,
        knn_dist=args.knn_dist,
        nce_temperature=args.nce_temperature,
        wta_k=args.wta_k,
        instance_nce=args.instance_nce,
        structure_nce=args.structure_nce,
        reconstruct=args.reconstruct,
        modularity=args.modularity,
        bce=args.bce)

    stamp = '{basic}--{dataset}--{model}'.format(basic=basic_stamp,
                                                 dataset=dataset_stamp,
                                                 model=model_stamp)
    return stamp


def dgl_2_adj(g):
    adj = g.adj()
    return adj.to_dense()


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def cluster_pred_2_gt(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    _, col_idx = linear_assignment(w.max() - w)
    return col_idx

def pred_2_gt_proj_acc(proj, y_true, y_pred):
    proj_pred = proj[y_pred]
    return accuracy_score(y_true, proj_pred)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BCE(nn.Module):
    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(
            simi), 'Wrong input size:{0},{1},{2}'.format(
                str(len(prob1)), str(len(prob2)), str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        #dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')