import enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler, optimizer
from torch.utils.data import DataLoader
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import BCE, PairEnum, accuracy, cluster_acc, AverageMeter, seed_torch, cluster_pred_2_gt, pred_2_gt_proj_acc
from utils import ramps
from models.resnet import ResNet, BasicBlock
from models.baseline import Baseline_v3
from data.cifarloader import CIFAR100Loader, CIFAR100Data
# from data.rotationloader import DataLoader, GenericDataset
from tqdm import tqdm
import numpy as np
import os
import copy
from models.lwf import MultiClassCrossEntropy
from models.si import init_si, update_si, update_omega, si_loss
from torch.utils.tensorboard import SummaryWriter
from functools import partial

torch.backends.cudnn.benchmark = True


def train_first_stage(model, writer, stage, device, args):
    # only ce loss
    train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True,
                                target_list=list(range(args.num_labeled_classes)))
    test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                 target_list=list(range(args.num_labeled_classes)))

    optimizer = SGD(model.parameters(), lr=args.ce_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    criterion = nn.CrossEntropyLoss()

    epochs = args.ce_epochs if not args.debug else 1

    for epoch in range(epochs):

        model.train()
        if args.first_ssl and not args.unfix_all:
            model.fix_encoder_layers()

        loss_record = AverageMeter()
        ce_loss_record = AverageMeter()
        acc_record = AverageMeter()

        for x, label, _ in train_loader:
            x, label = x.to(device), label.to(device)
            output, _, _ = model.ce_stage(x)

            loss_ce = criterion(output, label)

            loss = loss_ce

            acc = accuracy(output, label)

            loss_record.update(loss.item(), x.size(0))
            ce_loss_record.update(loss_ce.item(), x.size(0))
            acc_record.update(acc[0].item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        exp_lr_scheduler.step()

        print('Train Stage 1 \t Epoch: {} \t Avg Loss: {:.4f} Avg Acc: {:.4f}'.format(epoch, loss_record.avg, acc_record.avg))

        writer.add_scalar('Train Stage 1/Loss', loss_record.avg, epoch)
        writer.add_scalar('Train Stage 1/CE Loss', ce_loss_record.avg, epoch)
        writer.add_scalar('Train Stage 1/Acc for 0-{}'.format(args.num_labeled_classes), acc_record.avg, epoch)

        acc_record.reset()
        model.eval()
        with torch.no_grad():
            for x, label, _ in test_loader:
                x, label = x.to(device), label.to(device)
                output, _, _ = model.ce_stage(x)
            acc = accuracy(output, label)
            acc_record.update(acc[0].item(), x.size(0))

            print('Test Stage 1 \t Epoch: {} \t Avg Acc: {:.4f}'.format(epoch, acc_record.avg))
            writer.add_scalar('Test Stage 1/Acc for 0-{}'.format(args.num_labeled_classes), acc_record.avg, epoch)

    save_dir = os.path.join(args.model_dir, 'stage_1.pth')
    torch.save(model.state_dict(), save_dir)

    return save_dir


def train_ssl(model, writer, stage, device, args):
    lr = args.ssl_lr
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    # target_list_lower = (args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage) if stage > 0 else 0
    # target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list = list(range(100))
    train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True,
                                  target_list=target_list)

    epochs = args.ssl_epochs if not args.debug else 1

    for epoch in range(epochs):
        loss_record = AverageMeter()
        # ewc_record = AverageMeter()
        model.train()
        model.unfix_feature()
        if args.ft_ssl and stage >= args.ft_ssl_stage:
            model.fix_encoder_layers()

        for (x1, x2), _, _ in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            loss = model.ssl_stage(x1, x2)
            loss_record.update(loss.item(), x1.size(0))

            # if args.ewc and stage > 0:
            #     ewc_loss = model.ewc_loss(device, stage) * args.ewc_lambda
            #     loss = loss + ewc_loss

            #     ewc_record.update(ewc_loss.item(), x1.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        exp_lr_scheduler.step()

        print('Train Stage {} SSL \t Epoch: {} \t Avg Loss: {:.4f}'.format(stage + 1, epoch, loss_record.avg))
        writer.add_scalar('Train Stage {}/SSL/Loss'.format(stage + 1), loss_record.avg, epoch)

        # if args.ewc and stage > 0:
        #     print('Train Stage {} SSL \t Epoch: {} \t Avg EWC Loss: {:.4f}'.format(stage + 1, epoch, ewc_record.avg))
        #     writer.add_scalar('Train Stage {}/EWC/Loss'.format(stage + 1), ewc_record.avg, epoch)

    save_dir = os.path.join(args.model_dir, 'ssl_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)


def train_cluster(model, writer, stage, device, args):

    lr = args.cluster_lr
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    # criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list_train = list(range(target_list_lower, target_list_upper))
    target_list_test = list(range(target_list_lower, target_list_upper))

    train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='twice', target_list=target_list_train)
    test_dataset = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    epochs = args.cluster_epochs if not args.debug else 1

    for epoch in range(epochs):
        loss_record = AverageMeter()
        # ce_add_loss_record = AverageMeter()
        bce_loss_record = AverageMeter()
        mse_loss_record = AverageMeter()
        # ewc_record = AverageMeter()

        model.train()
        if not args.unfix_all:
            model.fix_encoder_layers()
        if args.fix_cluster:
            model.fix_feature()

        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

        preds = np.array([])
        targets = np.array([])

        for batch_idx, ((x, x_bar), label, _) in enumerate(train_loader):
            # label is only used for computing training acc
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            classify_output, cluster_output, features = model.ce_stage(x)
            classify_output_bar, cluster_output_bar, _ = model.ce_stage(x_bar)

            prob1 = F.softmax(classify_output, dim=1)
            prob2 = F.softmax(cluster_output, dim=1)
            prob1_bar = F.softmax(classify_output_bar, dim=1)
            prob2_bar = F.softmax(cluster_output_bar, dim=1)

            rank_feat = features.detach()

            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2)
            _, prob2_ulb = PairEnum(prob2_bar)

            loss = torch.tensor(0.0).to(device)

            if args.bce:
                loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
                loss += loss_bce

                bce_loss_record.update(loss_bce.item(), x.size(0))

                wta_sum = target_ulb.sum(0)
                wta_sum_min = wta_sum.min().item()
                wta_sum_max = wta_sum.max().item()
                writer.add_scalar('Train Stage {}/WTA/Sum min'.format(stage + 1), wta_sum_min, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Train Stage {}/WTA/Sum max'.format(stage + 1), wta_sum_max, epoch * len(train_loader) + batch_idx)


            if args.mse:
                loss_mse = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
                loss += loss_mse

                mse_loss_record.update(loss_mse.item(), x.size(0))


            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, classify_output.detach().max(1)[1].cpu().numpy())

        exp_lr_scheduler.step()

        acc = cluster_acc(targets.astype(int), preds.astype(int))
        nmi = nmi_score(targets.astype(int), preds.astype(int))

        print('Train Cluster Stage {} \t Epoch :{} \t Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(stage + 1, epoch, loss_record.avg, acc))
        writer.add_scalar('Train Stage {}/Cluster/Loss'.format(stage + 1), loss_record.avg, epoch)
        writer.add_scalar('Train Stage {}/Cluster/Acc'.format(stage + 1), acc, epoch)
        writer.add_scalar('Train Stage {}/Cluster/NMI'.format(stage + 1), nmi, epoch)
        if args.bce:
            writer.add_scalar('Train Stage {}/Cluster/BCE Loss'.format(stage + 1), bce_loss_record.avg, epoch)
        if args.mse:
            writer.add_scalar('Train Stage {}/Cluster/MSE Loss'.format(stage + 1), mse_loss_record.avg, epoch)
        # if args.ewc and stage > 0:
        #     writer.add_scalar('Train Stage {}/Cluster/EWC Loss'.format(stage + 1), ewc_record.avg, epoch)


        model.eval()
        preds = np.array([])
        cluster_preds = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for batch_idx, (x, label, _) in enumerate(test_loader):
                x, label = x.to(device), label.to(device)
                _, cluster_output, _ = model.ce_stage(x)
                targets = np.append(targets, label.cpu().numpy())
                cluster_preds = np.append(cluster_preds, cluster_output.detach().max(1)[1].cpu().numpy())
        cluster_preds = cluster_preds + target_list_lower


        # acc for cluster head:
        acc_cluster = cluster_acc(targets.astype(int), cluster_preds.astype(int))
        nmi = nmi_score(targets.astype(int), cluster_preds.astype(int))

        print('Test Cluster Stage {} Avg Acc for Cluster Head: {:.4f}'.format(stage + 1, acc_cluster))
        writer.add_scalar('Test Stage {}/Cluster/Acc for Cluster Head'.format(stage + 1), acc_cluster, epoch)
        writer.add_scalar('Test Stage {}/Cluster/NMI'.format(stage + 1), nmi, epoch)

    save_dir = os.path.join(args.model_dir, 'cluster_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)


def train_ce(model, writer, stage, device, args):
    lr = args.ce_lr
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()


    model.compute_means = True

    target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list_train = list(range(target_list_lower, target_list_upper))
    target_list_test = list(range(target_list_upper))

    train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='once', target_list=target_list_train)
    test_dataset = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test)

    train_dataset = model.pseudo_labeling_and_combine_with_exemplars(train_dataset, target_list_train)



    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    epochs = args.ce_epochs if not args.debug else 1
    epochs = epochs if not args.no_ce else 1

    if args.lwf:
        old_model = copy.deepcopy(model)
        old_model.eval()

    for epoch in range(epochs):

        if not args.no_ce:

            loss_record = AverageMeter()
            dist_loss_record = AverageMeter()

            model.train()
            if not args.unfix_all:
                model.fix_encoder_layers()

            # w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)


            # train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True,
                                        #target_list=target_list_train)
            # test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                        #target_list=target_list_test)

            preds = np.array([])
            targets = np.array([])

            for batch_idx, (x, pseudo) in enumerate(train_loader):
                x, pseudo = x.to(device), pseudo.to(device)
                x = x.float()
                classify_output, _, _ = model.ce_stage(x)
                loss = criterion1(classify_output, pseudo)

                if args.lwf:
                    g = F.sigmoid(classify_output)
                    with torch.no_grad():
                        q = F.sigmoid(old_model(x))
                    if args.debug:
                        print(q.shape)
                        print(g.shape)
                    dist_loss = F.binary_cross_entropy(g[:, :target_list_lower], q[:, :target_list_lower])
                    dist_loss = dist_loss * args.lwf_lambda
                    loss = loss + dist_loss

                    dist_loss_record.update(dist_loss.item(), x.size(0))

                loss_record.update(loss.item(), x.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                targets = np.append(targets, pseudo.cpu().numpy())
                preds = np.append(preds, classify_output.detach().max(1)[1].cpu().numpy())

            exp_lr_scheduler.step()

            acc = cluster_acc(targets.astype(int), preds.astype(int))

            print('Train CE Stage {} \t Epoch :{} \t Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(stage + 1, epoch, loss_record.avg, acc))
            writer.add_scalar('Train Stage {}/CE/Loss'.format(stage + 1), loss_record.avg, epoch)
            writer.add_scalar('Train Stage {}/CE/Dist Loss'.format(stage + 1), dist_loss_record.avg, epoch)
            writer.add_scalar('Train Stage {}/CE/Acc'.format(stage + 1), acc, epoch)

        model.eval()
        preds = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for batch_idx, (x, label, _) in enumerate(test_loader):
                x, label = x.to(device), label.to(device)
                pred = model.classify(x)
                targets = np.append(targets, label.cpu().numpy())
                preds = np.append(preds, pred)

        proj = cluster_pred_2_gt(preds.astype(int), targets.astype(int))
        pacc_fun = partial(pred_2_gt_proj_acc, proj)

        acc = cluster_acc(targets.astype(int), preds.astype(int))
        pacc = pacc_fun(targets.astype(int), preds.astype(int))
        nmi = nmi_score(targets.astype(int), preds.astype(int))

        print('Test CE Stage {} Avg Acc: {:.4f}'.format(stage + 1, acc))
        writer.add_scalar('Test Stage {}/CE/Acc'.format(stage + 1), acc, epoch)
        writer.add_scalar('Test Stage {}/CE/Pacc'.format(stage + 1), pacc, epoch)
        writer.add_scalar('Test Stage {}/CE/NMI'.format(stage + 1), nmi, epoch)

        # acc for labeled classes
        selected_mask = targets < args.num_labeled_classes
        acc_labeled = cluster_acc(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
        pacc_labeled = pacc_fun(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
        nmi_labeled = nmi_score(targets[selected_mask].astype(int), preds[selected_mask].astype(int))

        print('Test CE Stage {} Avg Acc for 0-{}: {:.4f}'.format(stage + 1, args.num_labeled_classes, acc_labeled))
        writer.add_scalar('Test Stage {}/CE/Acc for 0-{}'.format(stage + 1, args.num_labeled_classes), acc_labeled, epoch)
        writer.add_scalar('Test Stage {}/CE/Pacc for 0-{}'.format(stage + 1, args.num_labeled_classes), pacc_labeled, epoch)
        writer.add_scalar('Test Stage {}/CE/NMI for 0-{}'.format(stage + 1, args.num_labeled_classes), nmi_labeled, epoch)

        # acc for unlabeled classes in each stage
        for s in range(stage):
            lower = args.num_labeled_classes + s * args.num_unlabeled_classes_per_stage
            upper = args.num_labeled_classes + (s + 1) * args.num_unlabeled_classes_per_stage
            selected_mask = (targets >= lower) * (targets < upper)

            acc = cluster_acc(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
            pacc = pacc_fun(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
            nmi = nmi_score(targets[selected_mask].astype(int), preds[selected_mask].astype(int))

            print('Test CE Stage {} Avg Acc for {}-{}: {:.4f}'.format(stage + 1, lower, upper, acc))
            writer.add_scalar('Test Stage {}/CE/Acc for {}-{}'.format(stage + 1, lower, upper), acc, epoch)
            writer.add_scalar('Test Stage {}/CE/Pacc for {}-{}'.format(stage + 1, lower, upper), pacc, epoch)
            writer.add_scalar('Test Stage {}/CE/NMI for {}-{}'.format(stage + 1, lower, upper), nmi, epoch)


    save_dir = os.path.join(args.model_dir, 'ce_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)


def update_exemplars(model, stage, args):
    m = args.budgets // model.n_classes
    model.reduce_exemplar_sets(m)
    if stage > 0:
        target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
        target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
        target_list_train = list(range(target_list_lower, target_list_upper))
        train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='once', target_list=target_list_train)
        model.construct_exemplar_set_for_new_classes(train_dataset, target_list_train, m)
    else:
        target_list_train = list(range(args.num_labeled_classes))
        train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='once', target_list=target_list_train)
        model.construct_exemplar_set_for_new_classes_with_label(train_dataset, target_list_train, m)




def ewc(model, device, stage, args):
    supervised = stage == 0
    target_list_lower = (args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage) if stage > 0 else 0
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list = list(range(target_list_lower, target_list_upper))

    loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once', shuffle=True,
                            target_list=target_list)
    model.estimate_fisher(loader, device, stage, supervised)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--ssl_lr', type=float, default=0.01)
    parser.add_argument('--ce_lr', type=float, default=0.01)
    parser.add_argument('--cluster_lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ssl_epochs', default=200, type=int)
    parser.add_argument('--ce_epochs', default=200, type=int)
    parser.add_argument('--cluster_epochs', default=100, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes_per_stage', default=10, type=int)
    parser.add_argument('--num_labeled_classes', default=70, type=int)
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    # parser.add_argument( '--warmup_model_dir', type=str, default= './data/experiments/auto_novel_supervised_learning/resnet_rotnet.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='baseline')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--feature_extractor', type=str, default='resnet18')
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--moco_k', default=4096, type=int, help='queue size; number of negative keys')
    parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder')
    parser.add_argument('--moco_t', default=0.1, type=float, help='softmax temperature')

    parser.add_argument('--bn_splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

    # parser.add_argument('--use_ce_add', action='store_true', default=False)
    # parser.add_argument('--ce_add_full', action='store_true', default=False)

    # parser.add_argument('--lwf', action='store_true', default=False)
    # parser.add_argument('--lwf_ssl', action='store_true', default=False)
    # parser.add_argument('--lwf_cluster', action='store_true', default=False)
    # parser.add_argument('--lwf_t', default=2.0, type=float)
    # parser.add_argument('--lwf_c', default=1.0, type=float)

    # parser.add_argument('--ce_add', action='store_true', default=False)
    parser.add_argument('--bce', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)

    # parser.add_argument('--ce_add_part', action='store_true', default=False)

    # parser.add_argument('--ewc', action='store_true', default=False)
    # parser.add_argument('--ewc_ce', action='store_true', default=False)
    # parser.add_argument('--ewc_online', action='store_true', default=False)
    # parser.add_argument('--ewc_lambda', default=5000, type=float)
    # parser.add_argument('--ewc_gamma', default=1., type=float)

    parser.add_argument('--unfix_all', action='store_true', default=False)
    parser.add_argument('--ft_level', type=int, default=3)
    parser.add_argument('--ft_ssl', action='store_true', default=False)
    parser.add_argument('--ft_ssl_stage', type=int, default=1)
    parser.add_argument('--fix_cluster', action='store_true', default=False)

    parser.add_argument('--first_ssl', action='store_true', default=False)
    parser.add_argument('--skip_ssl', action='store_true', default=False)

    parser.add_argument('--no_ce', action='store_true', default=False)

    parser.add_argument('--skip_first', action='store_true', default=False)
    parser.add_argument('--skip_model_dir', type=str, default='baseline_v2_ceadd_bce_mse/stage_1.pth')

    parser.add_argument('--budgets', default=2000, type=int)
    parser.add_argument('--herding', action='store_true', default=False)

    parser.add_argument('--dist', choices=['euclidean', 'cosine'], default='euclidean')

    parser.add_argument('--lwf', action='store_true', default=False)
    parser.add_argument('--lwf_lambda', type=float, default=1.0)

    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, args.model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # args.model_dir = model_dir + '/' + '{}.pth'.format(args.model_name)
    args.model_dir = model_dir

    print(args)

    model = Baseline_v3(args)
    model = model.to(device)


    comment = '_{}'.format(args.model_name)
    writer = SummaryWriter(comment=comment)

    for stage in range(4):

        if stage == 0:
            if not args.skip_first:
                if args.first_ssl:
                    train_ssl(model, writer, stage, device, args)
                    if not args.unfix_all:
                        model.fix_encoder_layers()
                if not args.no_ce:
                    train_first_stage(model, writer, stage, device, args)
            else:
                state_dict = torch.load(os.path.join(args.exp_root, args.skip_model_dir))
                model.load_state_dict(state_dict)
                model.cuda()
            update_exemplars(model, stage, args)
        else:
            if not args.skip_ssl:
                model.unfix_feature()
                train_ssl(model, writer, stage, device, args)

            if not args.unfix_all:
                model.fix_encoder_layers()
            train_cluster(model, writer, stage, device, args)
            update_exemplars(model, stage, args)
            train_ce(model, writer, stage, device, args)


        # if args.ewc:
        #     ewc(model, device, stage, args)

        model.increment_classes(args.num_unlabeled_classes_per_stage, device)
