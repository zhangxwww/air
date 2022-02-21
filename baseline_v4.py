from email.policy import default
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler, optimizer
from torch.utils.data import DataLoader
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from utils.util import BCE, PairEnum, accuracy, cluster_acc, AverageMeter, seed_torch, cluster_pred_2_gt, pred_2_gt_proj_acc
from utils import ramps
from models.resnet import ResNet, BasicBlock
from models.baseline import Baseline_v4
from data.cifarloader import CIFAR100Loader, CIFAR100Data
from data.exemplers import ExemplarDataset, TransformDataset
# from data.rotationloader import DataLoader, GenericDataset
from tqdm import tqdm
import numpy as np
import os
import copy
from models.lwf import MultiClassCrossEntropy
from models.si import init_si, update_si, update_omega, si_loss
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from data.utils import TransformTwice
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True


def train_first_ssl(model, writer, stage, device, args):
    lr = args.ssl_lr
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    target_list_lower = (args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage) if stage > 0 else 0
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list = list(range(target_list_lower, target_list_upper))

    train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list=target_list)

    epochs = args.ssl_epochs if not args.debug else 1

    for epoch in range(epochs):
        loss_record = AverageMeter()
        # ewc_record = AverageMeter()
        model.train()

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

    save_dir = os.path.join(args.model_dir, 'ssl_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)


def train_first_ce(model, writer, stage, device, args):
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

        loss_record = AverageMeter()
        ce_loss_record = AverageMeter()
        acc_record = AverageMeter()

        for x, label, _ in train_loader:
            x, label = x.to(device), label.to(device)
            output, _, _ = model.ce_stage(x, 0)

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
                output, _, _ = model.ce_stage(x, 0)
            acc = accuracy(output, label)
            acc_record.update(acc[0].item(), x.size(0))

            print('Test Stage 1 \t Epoch: {} \t Avg Acc: {:.4f}'.format(epoch, acc_record.avg))
            writer.add_scalar('Test Stage 1/Acc for 0-{}'.format(args.num_labeled_classes), acc_record.avg, epoch)

    save_dir = os.path.join(args.model_dir, 'stage_1.pth')
    torch.save(model.state_dict(), save_dir)


def test_first_stage(model, writer, stage, device, args):
    test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                 target_list=list(range(args.num_labeled_classes)))
    model.eval()
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for x, label, _ in test_loader:
            x, label = x.to(device), label.to(device)
            pred, _, _ = model.classify(x, 0)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())

        acc = (preds.astype(int) == targets.astype(int)).sum() * 1.0 / preds.shape[0]
        # acc = accuracy(torch.from_numpy(preds).to(device), label)

        print('Test Stage 1 \t Epoch: {} \t Avg Acc: {:.4f}'.format(args.ce_epochs, acc))
        writer.add_scalar('Test Stage 1/Acc for 0-{}'.format(args.num_labeled_classes), acc, args.ce_epochs)


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
        model.fix_backbone()

        preds = np.array([])
        targets = np.array([])

        for batch_idx, ((x, x_bar), label, _) in enumerate(train_loader):
            # label is only used for computing training acc
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            classify_output, cluster_output, features = model.ce_stage(x, stage)
            classify_output_bar, cluster_output_bar, _ = model.ce_stage(x_bar, stage)

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

        model.eval()
        preds = np.array([])
        cluster_preds = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for batch_idx, (x, label, _) in enumerate(test_loader):
                x, label = x.to(device), label.to(device)
                _, cluster_output, _ = model.ce_stage(x, stage)
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


def train_ssl(model, writer, stage, device, args):
    lr = args.inc_ssl_lr
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list_train = list(range(target_list_lower, target_list_upper))

    train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, target_list=target_list_train)

    epochs = args.inc_ssl_epochs if not args.debug else 1

    prototypes = model.calculate_exemplar_means(device, stage)
    prototypes = torch.stack(prototypes, dim=0)
    if args.dist == 'euclidean':
        dist_func = euclidean_distances
    elif args.dist == 'cosine':
        dist_func = cosine_distances
    proto_dist = dist_func(prototypes.cpu().numpy(), prototypes.cpu().numpy())
    proto_dist = torch.from_numpy(proto_dist).to(device)
    old_protos = prototypes[:target_list_lower]
    n_proto = prototypes.shape[0]
    avg_proto_dist = proto_dist.sum() / (n_proto * (n_proto - 1))

    thres1 = avg_proto_dist * args.thres1_ratio
    thres2 = avg_proto_dist * args.thres2_ratio

    for epoch in range(epochs):
        loss_record = AverageMeter()
        model.train()
        model.fix_backbone()


        for (x1, x2), _, _ in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, cluster, feature = model.ce_stage(x1, stage)
            # _, _, feature2 = model.ce_stage(x2, stage)

            feature_2_proto = dist_func(feature.cpu().detach().numpy(), prototypes.cpu().numpy())
            feature_2_proto = torch.from_numpy(feature_2_proto).to(device)
            # feature_2_proto = torch.dist(feature, prototypes)    # (n_feature, n_proto)
            min_f_2_p, min_indices = feature_2_proto.min(dim=1)

            #_, refuse_indices = min_f_2_p.topk(int(args.thres2_ratio * min_f_2_p.shape[0]), dim=0, largest=True)

            refuse_indices = min_f_2_p >= thres2

            feature = feature[~refuse_indices]
            cluster_accept = cluster[~refuse_indices]
            min_f_2_p_accept = min_f_2_p[~refuse_indices]
            min_indices_accept = min_indices[~refuse_indices]

            hard_label_indices = min_f_2_p_accept < thres1
            soft_label_indices = min_f_2_p_accept >= thres1
            #hard_label_indices = min_f_2_p_accept.topk(int(args.thres1_ratio * min_f_2_p_accept.shape[0]), dim=0, largest=False)
            #soft_label

            # x_proto = prototypes[min_indices]

            # ssl: x1 >-< cur_proto, and x1 <-> old_protos
            logits = torch.einsum('nd,kd->nk', (feature, prototypes))
            logits = logits / args.ssl_temperature

            n_hard = hard_label_indices.sum()
            n_soft = soft_label_indices.sum()

            labels_onehot = torch.zeros(n_hard + n_soft, n_proto).to(device)
            labels_onehot[hard_label_indices].scatter_(1, min_indices_accept[hard_label_indices].view(-1, 1), 1)
            # print(labels_onehot.shape)
            # print(n_hard, n_soft, n_proto)
            # print(soft_label_indices.shape)
            # print(cluster_accept.shape)
            labels_onehot[soft_label_indices, -args.num_unlabeled_classes_per_stage:] = F.softmax(cluster_accept[soft_label_indices], dim=1)

            loss = (-F.log_softmax(logits, dim=1) * labels_onehot).sum(dim=1).mean()

            loss_record.update(loss.item(), x1.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train Stage {}/SSL/Loss'.format(stage + 1), loss_record.avg, epoch)
            writer.add_scalar('Train Stage {}/SSL/n refuse'.format(stage + 1), refuse_indices.sum(), epoch)
            writer.add_scalar('Train Stage {}/SSL/n hard'.format(stage + 1), hard_label_indices.sum(), epoch)
            writer.add_scalar('Train Stage {}/SSL/n soft'.format(stage + 1), soft_label_indices.sum(), epoch)

        exp_lr_scheduler.step()

    save_dir = os.path.join(args.model_dir, 'ssl_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)


def train_ce(model, writer, stage, device, args):
    lr = args.ce_lr
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    criterion1 = nn.CrossEntropyLoss()

    target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list_train = list(range(target_list_lower, target_list_upper))
    target_list_test = list(range(target_list_upper))

    train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='twice', target_list=target_list_train)
    test_dataset = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test)

    train_dataset = model.pseudo_labeling_and_combine_with_exemplars(train_dataset, target_list_train, stage)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    epochs = args.ce_epochs if not args.debug else 1

    for epoch in range(epochs):

        loss_record = AverageMeter()

        model.train()
        model.fix_backbone()

        preds = np.array([])
        targets = np.array([])

        for batch_idx, (x, _, pseudo) in enumerate(train_loader):
            x, pseudo = x.to(device), pseudo.to(device)
            x = x.float()
            classify_output, _, _ = model.ce_stage(x, stage)
            loss = criterion1(classify_output, pseudo)

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
        writer.add_scalar('Train Stage {}/CE/Acc'.format(stage + 1), acc, epoch)

        model.eval()
        preds = np.array([])
        preds_with_refuse = np.array([])
        targets = np.array([])
        targets_with_refuse = np.array([])
        with torch.no_grad():
            for batch_idx, (x, label, _) in enumerate(test_loader):
                x, label = x.to(device), label.to(device)
                pred, pred_with_refuse, refuse_index = model.classify(x, stage)
                targets = np.append(targets, label.cpu().numpy())
                preds = np.append(preds, pred.cpu().numpy())
                preds_with_refuse = np.append(preds_with_refuse, pred_with_refuse.cpu().numpy())
                targets_with_refuse = np.append(targets_with_refuse, label[~refuse_index].cpu().numpy())

        proj = cluster_pred_2_gt(preds.astype(int), targets.astype(int))
        pacc_fun = partial(pred_2_gt_proj_acc, proj)

        proj_with_refuse = cluster_pred_2_gt(preds_with_refuse.astype(int), targets_with_refuse.astype(int))
        pacc_fun_with_refuse = partial(pred_2_gt_proj_acc, proj_with_refuse)

        pacc = pacc_fun(targets.astype(int), preds.astype(int))

        pacc_with_refuse = pacc_fun_with_refuse(targets_with_refuse.astype(int), preds_with_refuse.astype(int))

        writer.add_scalar('Test Stage {}/CE/Pacc'.format(stage + 1), pacc, epoch)
        writer.add_scalar('Test Stage {}/CE/Pacc with refuse'.format(stage + 1), pacc_with_refuse, epoch)

        # acc for labeled classes
        selected_mask = targets < args.num_labeled_classes
        pacc_labeled = pacc_fun(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
        selected_mask_with_refuse = targets_with_refuse < args.num_labeled_classes
        pacc_labeled_with_refuse = pacc_fun_with_refuse(targets_with_refuse[selected_mask_with_refuse].astype(int), preds_with_refuse[selected_mask_with_refuse].astype(int))

        writer.add_scalar('Test Stage {}/CE/Pacc for 0-{}'.format(stage + 1, args.num_labeled_classes), pacc_labeled, epoch)
        writer.add_scalar('Test Stage {}/CE/Pacc with refuse for 0-{}'.format(stage + 1, args.num_labeled_classes), pacc_labeled_with_refuse, epoch)
        if len(pacc_labeled_with_refuse.shape) > 0:
            writer.add_scalar('Test Stage {}/CE/N accepted for 0-{}'.format(stage + 1, args.num_labeled_classes), pacc_labeled_with_refuse.shape[0], epoch)

        # acc for unlabeled classes in each stage
        for s in range(stage):
            lower = args.num_labeled_classes + s * args.num_unlabeled_classes_per_stage
            upper = args.num_labeled_classes + (s + 1) * args.num_unlabeled_classes_per_stage
            selected_mask = (targets >= lower) * (targets < upper)
            selected_mask_with_refuse = (targets_with_refuse >= lower) * (targets_with_refuse < upper)

            pacc = pacc_fun(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
            pacc_with_refuse = pacc_fun_with_refuse(targets_with_refuse[selected_mask_with_refuse].astype(int), preds_with_refuse[selected_mask_with_refuse].astype(int))

            writer.add_scalar('Test Stage {}/CE/Pacc for {}-{}'.format(stage + 1, lower, upper), pacc, epoch)
            writer.add_scalar('Test Stage {}/CE/Pacc with refuse for {}-{}'.format(stage + 1, lower, upper), pacc_with_refuse, epoch)
            if len(pacc_with_refuse.shape) > 0:
                writer.add_scalar('Test Stage {}/CE/N accepted for {}-{}'.format(stage + 1, lower, upper), pacc_with_refuse.shape[0], epoch)


    if args.print_cls_statistics:
        np.save('preds_{}.npy'.format(stage), preds.astype(int))
        np.save('ytrue_{}.npy'.format(stage), targets.astype(int))
        np.save('proj_{}.npy'.format(stage), proj)

        np.save('preds_with_refuse_{}.npy'.format(stage), preds_with_refuse.astype(int))
        np.save('ytrue_with_refuse_{}.npy'.format(stage), targets_with_refuse.astype(int))
        np.save('proj_with_refuse_{}.npy'.format(stage), proj_with_refuse)


    save_dir = os.path.join(args.model_dir, 'ce_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)


def update_exemplars(model, stage, args):
    m = args.budgets // model.n_classes
    model.reduce_exemplar_sets(m)
    if stage > 0:
        target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
        target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
        target_list_train = list(range(target_list_lower, target_list_upper))
        train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='twice', target_list=target_list_train)
        model.construct_exemplar_set_for_new_classes(train_dataset, target_list_train, m, stage)
    else:
        target_list_train = list(range(args.num_labeled_classes))
        train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='twice', target_list=target_list_train)
        model.construct_exemplar_set_for_new_classes_with_label(train_dataset, target_list_train, m)

    if args.save_exemplars:
        save_dir = os.path.join(args.model_dir, 'exemplars_{}.pth'.format(stage + 1))
        torch.save(model.exemplar_sets, save_dir)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--ssl_lr', type=float, default=0.01)
    parser.add_argument('--ce_lr', type=float, default=0.01)
    parser.add_argument('--cluster_lr', type=float, default=0.01)
    parser.add_argument('--inc_ssl_lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ssl_epochs', default=100, type=int)
    parser.add_argument('--ce_epochs', default=100, type=int)
    parser.add_argument('--cluster_epochs', default=100, type=int)
    parser.add_argument('--inc_ssl_epochs', default=100, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=70, type=int)
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

    parser.add_argument('--ssl_temperature', default=0.1, type=float, help='softmax temperature')

    # parser.add_argument('--ce_add', action='store_true', default=False)
    parser.add_argument('--bce', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)

    #parser.add_argument('--unfix_all', action='store_true', default=False)
    #parser.add_argument('--ft_level', type=int, default=3)
    #parser.add_argument('--ft_ssl', action='store_true', default=False)
    #parser.add_argument('--ft_ssl_stage', type=int, default=1)

    #parser.add_argument('--fix_cluster', action='store_true', default=False)
    #parser.add_argument('--fix_cluster_level', type=int, default=1)

    parser.add_argument('--first_ssl', action='store_true', default=False)
    parser.add_argument('--first_ce', action='store_true', default=False)
    parser.add_argument('--skip_ssl', action='store_true', default=False)

    # parser.add_argument('--inc_ssl', action='store_true', default=False)
    # parser.add_argument('--inc_ssl_data', choices=['exemplars', 'new_ins', 'both', 'none'], default='none')
    # parser.add_argument('--inc_ssl_lr_decay', type=float, default=1.0)
    # parser.add_argument('--inc_ssl_lr_decay_per_stage', type=float, default=1.0)

    # parser.add_argument('--no_ce', action='store_true', default=False)

    parser.add_argument('--skip_first', action='store_true', default=False)
    parser.add_argument('--skip_model_dir', type=str, default='baseline_v2_ceadd_bce_mse/stage_1.pth')

    parser.add_argument('--budgets', default=2000, type=int)
    parser.add_argument('--herding', action='store_true', default=False)

    parser.add_argument('--dist', choices=['euclidean', 'cosine'], default='euclidean')

    # parser.add_argument('--lwf', action='store_true', default=False)
    # parser.add_argument('--lwf_lambda', type=float, default=1.0)

    parser.add_argument('--discovery_confidence', action='store_true', default=False)
    parser.add_argument('--discovery_confidence_threshold', type=float, default=0.0)
    parser.add_argument('--discovery_confidence_percent', default=1.0, type=float)
    parser.add_argument('--confidence_type', choices=['all', 'class', 'threshold'], default='all')

    parser.add_argument('--geo', action='store_true', default=False)
    parser.add_argument('--geo_dist', choices=['euclidean', 'cosine'], default='euclidean')
    parser.add_argument('--geo_k', default=5, type=int)
    parser.add_argument('--geo_percent', default=0.5, type=float)
    parser.add_argument('--geo_print_statistics', action='store_true', default=False)

    parser.add_argument('--thres1_ratio', type=float, default=0.1)
    parser.add_argument('--thres2_ratio', type=float, default=0.8)

    parser.add_argument('--ema_beta', type=float, default=0.9)

    parser.add_argument('--norm_before_add', action='store_true', default=False)

    # parser.add_argument('--cls_type', choices=['proto', 'lp'], default='proto')
    # parser.add_argument('--lp_k', default=20, type=int)
    # parser.add_argument('--lp_alpha', default=0.99, type=float)

    # parser.add_argument('--update_means', action='store_true', default=False)

    parser.add_argument('--print_cls_statistics', action='store_true', default=False)
    parser.add_argument('--save_exemplars', action='store_true', default=False)

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

    model = Baseline_v4(args)
    model = model.to(device)


    comment = '_{}'.format(args.model_name)
    writer = SummaryWriter(comment=comment)

    for stage in range(4):


        if stage == 0:
            if not args.skip_first:
                if args.first_ssl:
                    train_first_ssl(model, writer, stage, device, args)
                if args.first_ce:
                    train_first_ce(model, writer, stage, device, args)
            else:
                state_dict = torch.load(os.path.join(args.exp_root, args.skip_model_dir))
                model.load_state_dict(state_dict)
                model.cuda()
            update_exemplars(model, stage, args)
            model.sync_weights()
            test_first_stage(model, writer, stage, device, args)
        else:
            train_cluster(model, writer, stage, device, args)
            update_exemplars(model, stage, args)
            train_ssl(model, writer, stage, device, args)
            train_ce(model, writer, stage, device, args)

        model.increment_classes(args.num_unlabeled_classes_per_stage, device)
