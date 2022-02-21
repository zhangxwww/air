import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler, SGD
from sklearn.metrics import accuracy_score
# from data.cifarloader import CIFAR100Loader
from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from utils.util import cluster_acc, seed_torch
from torchvision.models import resnet18, resnet34, resnet50
from utils.util import WarmUpLR, get_stamp

from data.cifarloader import CIFAR100LoaderMix, CIFAR100Loader, StageCIFAR100Loader, PARTITION_CONFIG
from models.model import Air, NCE, modularity_loss, WTA

import argparse

parser = argparse.ArgumentParser(description='Air')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--lr_decay_gamma', type=float, default=0.2)
parser.add_argument('--lr_decay_step', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_labeled_classes', default=50, type=int)
parser.add_argument('--num_unlabeled_classes', default=50, type=int)
parser.add_argument('--dataset_root', type=str, default='./dataset')
parser.add_argument('--exp_root', type=str, default='./data/experiments/')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--warm', default=1, type=int)

parser.add_argument('--feature_extractor',
                    default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50'])

parser.add_argument('--knn_graph_k', default=5, type=int)
parser.add_argument('--knn_dist', default='cosine', type=str)

parser.add_argument('--nce_temperature', default=0.1, type=float)
parser.add_argument('--wta_k', default=5, type=int)

parser.add_argument('--instance_nce', action='store_true', default=False)
parser.add_argument('--instance_nce_lambda', default=1.0, type=float)

parser.add_argument('--structure_nce', action='store_true', default=False)
parser.add_argument('--structure_nce_lambda', default=1.0, type=float)

parser.add_argument('--reconstruct', action='store_true', default=False)
parser.add_argument('--reconstruct_lambda', default=1.0, type=float)

parser.add_argument('--modularity', action='store_true', default=False)
parser.add_argument('--modularity_lambda', default=1.0, type=float)

parser.add_argument('--bce', action='store_true', default=False)
parser.add_argument('--bce_lambda', default=1.0, type=float)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.device = device
seed_torch(args.seed)

stamp = get_stamp(args)

# train_loader = CIFAR100LoaderMix(root=args.dataset_root,
#                                  batch_size=args.batch_size,
#                                  split='train',
#                                  aug='once',
#                                  shuffle=True,
#                                  labeled_list=range(args.num_labeled_classes),
#                                  unlabeled_list=range(args.num_labeled_classes,
#                                                       100),
#                                  num_workers=args.workers)

# test_loader_labeled = CIFAR100Loader(root=args.dataset_root,
#                                      batch_size=args.batch_size,
#                                      split='test',
#                                      aug=None,
#                                      shuffle=False,
#                                      target_list=range(
#                                          args.num_labeled_classes),
#                                      num_workers=args.workers)

# test_loader_unlabeled = CIFAR100Loader(
#     root=args.dataset_root,
#     batch_size=args.batch_size,
#     split='test',
#     aug=None,
#     shuffle=False,
#     target_list=range(args.num_labeled_classes,
#                       args.num_labeled_classes + args.num_unlabeled_classes),
#     num_workers=args.workers)

train_loader, _ = StageCIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True, partition_config=PARTITION_CONFIG['stage 1'])

test_loader_labeled, _ = StageCIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, partition_config=PARTITION_CONFIG['stage 1'])

if args.feature_extractor == 'resnet18':
    model = resnet18(pretrained=False)
elif args.feature_extractor == 'resnet34':
    model = resnet34(pretrained=False)
elif args.feature_extractor == 'resnet50':
    model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 100)
# model = Air(args)
model.cuda()

optimizer = SGD(model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=args.lr_decay_gamma)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

iter_per_epoch = len(train_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

print(stamp)
writer = SummaryWriter(comment='_debug')

# nl = args.num_labeled_classes
nl = 40

for epoch in range(args.epochs):
    # train
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        # for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        label_mask = target < nl

        optimizer.zero_grad()
        # output, feature, z, adj, adj_reconstructed = model(data)
        output = model(data)

        # loss_ce = F.cross_entropy(output[label_mask, :nl], target[label_mask])
        loss_ce = F.cross_entropy(output, target)

        # loss_nce_instance = (NCE(feature, args.nce_temperature) if args.instance_nce else torch.tensor(0)) * args.instance_nce_lambda

        # loss_nce_structure = (NCE(z, args.nce_temperature) if args.structure_nce else torch.tensor(0)) * args.structure_nce_lambda

        # loss_reconstruct = (F.binary_cross_entropy(adj_reconstructed, adj) if args.reconstruct else torch.tensor(0)) * args.reconstruct_lambda

        # loss_modularity = (modularity_loss(output, adj) if args.modularity else torch.tensor(0)) * args.modularity_lambda

        # loss_bce = (F.binary_cross_entropy(adj_reconstructed.reshape(-1), WTA(output, args.wta_k)) if args.bce else torch.tensor(0)) * args.bce_lambda

        # loss = loss_ce + loss_nce_instance + loss_nce_structure + loss_reconstruct + loss_modularity + loss_bce

        loss = loss_ce

        loss.backward()
        optimizer.step()

        # pred = output[label_mask, :nl].max(1, keepdim=True)[1]
        pred = output.max(1, keepdim=True)[1]
        # train_acc = accuracy_score(target[label_mask].cpu().numpy(), pred.cpu().numpy())
        train_acc = accuracy_score(target.cpu().numpy(), pred.cpu().numpy())

        writer.add_scalar('train loss', loss.item(),
                          epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('train loss nce instance', loss_nce_instance.item(),
        #                   epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('train loss nce structure',
        #                   loss_nce_structure.item(),
        #                   epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('train loss reconstruct', loss_reconstruct.item(),
        #                   epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('train loss modularity', loss_modularity.item(),
        #                   epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('train loss bce', loss_bce.item(),
        #                   epoch * len(train_loader) + batch_idx)

        if epoch < args.warm:
            warmup_scheduler.step()
    if epoch >= args.warm:
        exp_lr_scheduler.step()

    # eval
    model.eval()
    with torch.no_grad():
        total_correct = 0
        for batch_idx, (data, target, _) in enumerate(test_loader_labeled):
            # for batch_idx, (data, target) in enumerate(test_loader):

            label_mask = target < nl

            data, target = data.cuda(), target.cuda()
            # output, _, _, _, _ = model(data)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            # correct = accuracy_score(target[label_mask].cpu().numpy(),
            #                          pred[label_mask].cpu().numpy(),
            #                          normalize=False)
            correct = accuracy_score(target.cpu().numpy(), pred.cpu().numpy(), normalize=False)
            total_correct += correct
        writer.add_scalar('test-acc', total_correct / len(test_loader_labeled), epoch)

        # preds = np.array([], dtype=np.int64)
        # targets = np.array([], dtype=np.int64)
        # for batch_idx, (data, target, _) in enumerate(test_loader_unlabeled):
        #     data, target = data.to(args.device), target.to(args.device)
        #     output, _, _, _, _ = model(data)

        #     label_mask = target < nl

        #     targets = np.append(targets, target[~label_mask].cpu().numpy())
        #     output = output[~label_mask, nl:]

        #     pred = output.max(1)[1] + nl

        #     preds = np.append(preds, pred.cpu().numpy())

        # u_acc = cluster_acc(targets, preds)
        # writer.add_scalar('cluster-acc', u_acc, epoch)
