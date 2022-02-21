import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck
import dgl
from dgl.nn import GCN2Conv
from utils.util import dgl_2_adj, PairEnum

eps = np.finfo(float).eps


class Air(nn.Module):
    def __init__(self, args):
        super(Air, self).__init__()

        self.args = args

        if self.args.feature_extractor == 'resnet18':
            self.feature_extractor = models.resnet18()
        elif self.args.feature_extractor == 'resnet34':
            self.feature_extractor = models.resnet34()

        n_features = self.feature_extractor.fc.in_features

        self.feature_extractor.fc = nn.Identity()

        self.graph_encoder = GCN2Conv(n_features, n_features, 1)

        # self.label_head = nn.Linear(n_features, args.n_labeled_classes)
        # self.unlabel_head = nn.Linear(n_features, args.n_unlabeled_classes)
        self.head = nn.Linear(
            n_features, args.num_labeled_classes + args.num_unlabeled_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):

        z, adj, adj_reconstructed = None, None, None

        feature = self.feature_extractor(x)

        if self.args.modularity or self.args.structure_nce or self.args.reconstruct or self.args.bce:
            dgl_adj = dgl.knn_graph(feature,
                                    k=self.args.knn_graph_k,
                                    dist=self.args.knn_dist)
            adj = dgl_2_adj(dgl_adj).to(self.args.device)

        if self.args.structure_nce or self.args.reconstruct or self.args.bce:
            z = self.graph_encoder(dgl_adj, feature, feature)

        if self.args.reconstruct or self.args.bce:
            adj_reconstructed = torch.sigmoid(z @ z.t())

        # label_logits = self.label_head(feature)
        # unlabel_logits = self.unlabel_head(feature)

        final_feature = feature

        logits = self.head(final_feature)

        return logits, feature, z, adj, adj_reconstructed


def NCE(x, t):
    logits = x.mm(x.t()) / t
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    exp_logits = torch.exp(logits) + eps
    log_prob = -torch.log(
        exp_logits / torch.sum(exp_logits, dim=1, keepdim=True))
    # return -F.log_softmax(logits, dim=1).diag().mean()
    return torch.mean(log_prob)


def modularity_loss(x, adj):

    n = adj.shape[0]
    D = adj.sum(1) + eps
    D_sqrt_inv = 1.0 / torch.sqrt(D)

    I = torch.eye(n).to(x.device)
    L = I - D_sqrt_inv.reshape(1, -1) * adj * D_sqrt_inv.reshape(-1, 1)
    return torch.trace(x.t() @ L @ x) / x.size(0)


def WTA(x, topk=5):
    rank_feat = x.detach()
    rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
    rank_idx1, rank_idx2 = PairEnum(rank_idx)
    rank_idx1, rank_idx2 = rank_idx1[:, :topk], rank_idx2[:, :topk]
    rank_idx1, _ = rank_idx1.sort(dim=1)
    rank_idx2, _ = rank_idx2.sort(dim=1)

    rank_diff = rank_idx1 - rank_idx2
    rank_diff = rank_diff.abs().sum(1)
    s = torch.ones_like(rank_diff).float().to(x.device)
    s[rank_diff > 0] = 0
    return s
