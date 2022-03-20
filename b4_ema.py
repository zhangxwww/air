import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler, optimizer
from torch.utils.data import DataLoader
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.neighbors import kneighbors_graph
from utils.util import BCE, PairEnum, accuracy, cluster_acc, AverageMeter, seed_torch, cluster_pred_2_gt, pred_2_gt_proj_acc
from models.model_ema import Baseline_v4_ema
from data.cifarloader import CIFAR100Loader, CIFAR100Data
# from data.rotationloader import DataLoader, GenericDataset
import numpy as np
import os
import copy
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from data.utils import TransformTwice, MixUpWrapper
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

    # for n, p in model.named_parameters():
    #     print('{} {}'.format(n, p.requires_grad))

    # criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list_train = list(range(target_list_lower, target_list_upper))
    target_list_test = list(range(target_list_lower, target_list_upper))

    train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='twice', target_list=target_list_train)
    test_dataset = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    target_list_test_all = list(range(target_list_upper))
    test_dataset_all = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test_all)
    test_loader_all = DataLoader(test_dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    epochs = args.cluster_epochs if not args.debug else 1

    #init_weight = model.moco.encoder_q.net.conv1.weight.data.clone()

    for epoch in range(epochs):
        loss_record = AverageMeter()
        # ce_add_loss_record = AverageMeter()
        bce_loss_record = AverageMeter()
        mse_loss_record = AverageMeter()
        # ewc_record = AverageMeter()

        model.train()

        preds = np.array([])
        targets = np.array([])


        for batch_idx, ((x, x_bar), label, _) in enumerate(train_loader):
            # label is only used for computing training acc
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            classify_output, cluster_output, features = model.ce_stage(x, stage)
            classify_output_bar, cluster_output_bar, features_bar = model.ce_stage(x_bar, stage)

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
                if args.replace_wta:
                    normed_feature = F.normalize(features, p=2, dim=1)
                    # normed_rank_feat = normed_feature.detach()
                    simi = normed_feature.mm(normed_feature.t())
                    # simi = torch.matmul(normed_rank_feat, normed_rank_feat.t())
                    if args.rw_graph:
                        # print(simi.shape, args.rw_graph_k)
                        _, inds = torch.topk(simi, k=args.rw_graph_k)
                        mask = torch.zeros_like(simi).to(device)
                        mask = mask.scatter(1, inds, 1)
                        mask = ((mask + mask.t()) > 0).float()
                        adj = torch.exp((simi - 1) / 2) * mask
                        adj = adj.detach()
                        n = adj.shape[0]
                        D = adj.sum(0)
                        D_sqrt_inv = torch.sqrt(1 / (D + 1e-7))
                        D1 = D_sqrt_inv.unsqueeze(1).repeat(1, n)
                        D2 = D_sqrt_inv.unsqueeze(0).repeat(n, 1)
                        adj = D1 * adj * D2
                        cct = prob2.mm(prob2.t())
                        # loss_bce = ((1 - adj) * cct).sum(1).mean()
                        loss_bce = ((1 - adj) * cct).mean()
                    else:
                        loss_bce = criterion2(prob1_ulb, prob2_ulb, simi.detach().view(-1))

                    if args.rw_feat:
                        features_norm = F.normalize(features, p=2, dim=1)
                        features_bar_norm = F.normalize(features_bar, p=2, dim=1)
                        logits_feat_pos = (features_norm * features_bar_norm).sum(1, keepdim=True)
                        logits_feat_neg = features_norm.mm(features_norm.t())
                        logits_feat = torch.cat((logits_feat_pos, logits_feat_neg), dim=1) / args.ssl_temperature
                        label_feat = torch.zeros(logits_feat.shape[0]).to(device).long()
                        loss_bce += F.cross_entropy(logits_feat, label_feat)
                    if args.rw_label:
                        cluster_norm = F.normalize(cluster_output, p=2, dim=1)
                        cluster_bar_norm = F.normalize(cluster_output_bar, p=2, dim=1)
                        logits_label_pos = (cluster_norm * cluster_bar_norm).sum(1, keepdim=True)
                        logits_label_neg = cluster_norm.mm(cluster_norm.t())
                        logits_label = torch.cat((logits_label_pos, logits_label_neg), dim=1) / args.ssl_temperature
                        label_label = torch.zeros(logits_label.shape[0]).to(device).long()
                        loss_bce += F.cross_entropy(logits_label, label_label)

                else:
                    loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
                loss += loss_bce

                bce_loss_record.update(loss_bce.item(), x.size(0))

                wta_sum = target_ulb.sum(0)
                wta_sum_min = wta_sum.min().item()
                wta_sum_max = wta_sum.max().item()
                writer.add_scalar('Train Stage {}/WTA/Sum min'.format(stage + 1), wta_sum_min, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('Train Stage {}/WTA/Sum max'.format(stage + 1), wta_sum_max, epoch * len(train_loader) + batch_idx)


            if args.mse:
                if args.replace_mse:
                    loss_mse = - (F.normalize(cluster_output, p=2, dim=1) *
                        F.normalize(cluster_output_bar, p=2, dim=1)).sum(1).mean()
                else:
                    loss_mse = F.mse_loss(prob2, prob2_bar)
                if not args.no_fc:
                    loss_mse += F.mse_loss(prob1, prob1_bar)
                loss += loss_mse

                mse_loss_record.update(loss_mse.item(), x.size(0))

            if args.cluster_with_pef:
                exemplars = model.exemplar_sets[:target_list_lower]
                exemplars = [torch.from_numpy(exem).to(device).float() for exem in exemplars]
                exemplars = torch.cat(exemplars, dim=0)

                # old_feature = model.moco.encoder_q(exemplars, 0)
                # old_feature = F.normalize(old_feature, dim=1, p=2)
                # new_feature = model.moco.encoder_q(exemplars, stage)
                # new_feature = F.normalize(new_feature, dim=1, p=2)

                n_exemplars = exemplars.shape[0]
                bs_exemplars = 128
                n_step = int(n_exemplars // bs_exemplars) + 1

                for i in range(n_step):
                    with torch.no_grad():
                        old_feature = F.normalize(model.ema_branch(exemplars[i * bs_exemplars : (i + 1) * bs_exemplars]), dim=1, p=2)
                    new_feature = F.normalize(model.moco.encoder_q(exemplars[i * bs_exemplars : (i + 1) * bs_exemplars]), dim=1, p=2)

                    loss_feature_dist = -(old_feature * new_feature).sum(1).mean()
                    loss += loss_feature_dist * args.cluster_with_pef_weight

                # with torch.no_grad():
                #     old_feature = F.normalize(model.ema_branch(exemplars), dim=1, p=2)
                # model.moco.encoder_q.cuda(1)
                # new_feature = F.normalize(model.moco.encoder_q(exemplars.cuda(1)), dim=1, p=2).cuda(0)

                # loss_feature_dist = -(old_feature * new_feature).sum(dim=1).mean()
                # loss += loss_feature_dist * args.cluster_with_pef_weight

            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #print((init_weight - model.moco.encoder_q.net.conv1.weight.data.clone()).sum().item())

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


        if epoch % 10 == 9:
            test(model, test_loader_all, writer, stage, epoch, device, args)

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


    target_list_test_all = list(range(target_list_upper))
    test_dataset_all = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test_all)
    test_loader_all = DataLoader(test_dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


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


        if epoch % 10 == 9:
            test(model, test_loader_all, writer, stage, epoch + args.cluster_epochs, device, args)

    save_dir = os.path.join(args.model_dir, 'ssl_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)



def train_ssl_with_exemplars(model, writer, stage, device, args):
    lr = args.inc_ssl_lr
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list_train = list(range(target_list_lower, target_list_upper))

    train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='twice', target_list=target_list_train)
    train_dataset = model.pseudo_labeling_and_combine_with_exemplars(train_dataset, target_list_train, stage, device, args)
    if args.mixup:
        train_dataset_mixup = model.mixup(train_dataset, target_list_train, device, args)
    else:
        train_dataset_mixup = train_dataset

    train_loader = DataLoader(train_dataset_mixup, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    target_list_test_all = list(range(target_list_upper))
    test_dataset_all = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test_all)
    test_loader_all = DataLoader(test_dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    epochs = args.inc_ssl_epochs if not args.debug else 1

    prototypes = model.calculate_exemplar_means(device, stage)
    prototypes = torch.stack(prototypes, dim=0)
    if args.dist == 'euclidean':
        dist_func = euclidean_distances
    elif args.dist == 'cosine':
        dist_func = cosine_distances
    proto_dist = dist_func(prototypes.cpu().numpy(), prototypes.cpu().numpy())
    proto_dist = torch.from_numpy(proto_dist).to(device)
    n_proto = prototypes.shape[0]
    avg_proto_dist = proto_dist.sum() / (n_proto * (n_proto - 1))

    thres1 = avg_proto_dist * args.thres1_ratio
    thres2 = avg_proto_dist * args.thres2_ratio

    bce = BCE()

    for epoch in range(epochs):
        loss_record = AverageMeter()
        model.train()

        mix_loader = MixUpWrapper(args.mixup_alpha, train_loader, args)
        train_loader_final = mix_loader if args.batch_mixup else train_loader

        for x, x_bar, pseudo in train_loader_final:
            x, x_bar, pseudo = x.to(device), x_bar.to(device), pseudo.to(device)
            classify_output, cluster_output, feature = model.ce_stage(x, stage)
            classify_output_bar, cluster_output_bar, _ = model.ce_stage(x_bar, stage)

            feature_2_proto = dist_func(feature.cpu().detach().numpy(), prototypes.cpu().numpy())
            feature_2_proto = torch.from_numpy(feature_2_proto).to(device)
            min_f_2_p, min_indices = feature_2_proto.min(dim=1)

            refuse_indices = min_f_2_p >= thres2

            feature_accept = feature[~refuse_indices]
            pseudo_accept = pseudo[~refuse_indices]

            # ssl: x1 >-< cur_proto, and x1 <-> old_protos
            logits = feature_accept.mm(prototypes.t())
            logits = logits / args.ssl_temperature

            loss = (-F.log_softmax(logits, dim=1) * pseudo_accept).sum(dim=1).mean()

            if args.pull_exemplar_features:
                if not args.pef_all:
                    exemplars = model.exemplar_sets[:target_list_lower]
                else:
                    exemplars = model.exemplar_sets[:]
                exemplars = [torch.from_numpy(exem).to(device).float() for exem in exemplars]
                exemplars = torch.cat(exemplars, dim=0)

                n_exemplars = exemplars.shape[0]
                bs_exemplars = 128
                n_step = int(n_exemplars // bs_exemplars) + 1

                for i in range(n_step):
                    with torch.no_grad():
                        old_feature = F.normalize(model.ema_branch(exemplars[i * bs_exemplars : (i + 1) * bs_exemplars]), dim=1, p=2)
                    new_feature = F.normalize(model.moco.encoder_q(exemplars[i * bs_exemplars : (i + 1) * bs_exemplars]), dim=1, p=2)

                    loss_feature_dist = -(old_feature * new_feature).sum(1).mean()
                    loss += loss_feature_dist * args.cluster_with_pef_weight


            if args.ssl_with_cluster:

                prob1 = F.softmax(classify_output, dim=1)
                prob2 = F.softmax(cluster_output, dim=1)
                prob1_bar = F.softmax(classify_output_bar, dim=1)
                prob2_bar = F.softmax(cluster_output_bar, dim=1)

                rank_feat = feature.detach()

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

                if args.bce:
                    loss += bce(prob1_ulb, prob2_ulb, target_ulb)
                if args.mse:
                    loss += F.mse_loss(prob2, prob2_bar)
                    if not args.no_fc:
                        loss += F.mse_loss(prob1, prob1_bar)


            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train Stage {}/SSL/Loss'.format(stage + 1), loss_record.avg, epoch)
            writer.add_scalar('Train Stage {}/SSL/n refuse'.format(stage + 1), refuse_indices.sum(), epoch)

        exp_lr_scheduler.step()

        if args.reselect_exemplars and epoch % args.reselect_exemplars_interval == (args.reselect_exemplars_interval - 1):
            reselect_exemplars(model, stage, args)

        if args.re_mixup:
            train_dataset_mixup = model.mixup(train_dataset, target_list_train, device, args)
            train_loader = DataLoader(train_dataset_mixup, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

        if epoch % 10 == 9:
            test(model, test_loader_all, writer, stage, epoch + args.cluster_epochs, device, args)

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
        np.save(os.path.join(args.model_dir, 'preds_{}.npy'.format(stage), preds.astype(int)))
        np.save(os.path.join(args.model_dir, 'ytrue_{}.npy'.format(stage), targets.astype(int)))
        np.save(os.path.join(args.model_dir, 'proj_{}.npy'.format(stage), proj))

        np.save(os.path.join(args.model_dir, 'preds_with_refuse_{}.npy'.format(stage), preds_with_refuse.astype(int)))
        np.save(os.path.join(args.model_dir, 'ytrue_with_refuse_{}.npy'.format(stage), targets_with_refuse.astype(int)))
        np.save(os.path.join(args.model_dir, 'proj_with_refuse_{}.npy'.format(stage), proj_with_refuse))


    save_dir = os.path.join(args.model_dir, 'ce_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)


def test(model, loader, writer, stage, epoch, device, args):
    model.eval()
    preds = np.array([])
    preds_with_refuse = np.array([])
    targets = np.array([])
    targets_with_refuse = np.array([])
    no_accept = False
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(loader):
            x, label = x.to(device), label.to(device)
            pred, pred_with_refuse, refuse_index = model.classify(x, stage)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            if pred_with_refuse is not None:
                preds_with_refuse = np.append(preds_with_refuse, pred_with_refuse.cpu().numpy())
            else:
                no_accept = True
            targets_with_refuse = np.append(targets_with_refuse, label[~refuse_index].cpu().numpy())

    proj = cluster_pred_2_gt(preds.astype(int), targets.astype(int))
    pacc_fun = partial(pred_2_gt_proj_acc, proj)
    pacc = pacc_fun(targets.astype(int), preds.astype(int))
    writer.add_scalar('Test Stage {}/Pacc'.format(stage + 1), pacc, epoch)

    if not no_accept:
        proj_with_refuse = cluster_pred_2_gt(preds_with_refuse.astype(int), targets_with_refuse.astype(int))
        pacc_fun_with_refuse = partial(pred_2_gt_proj_acc, proj_with_refuse)
        pacc_with_refuse = pacc_fun_with_refuse(targets_with_refuse.astype(int), preds_with_refuse.astype(int))
        writer.add_scalar('Test Stage {}/Pacc with refuse'.format(stage + 1), pacc_with_refuse, epoch)

    # acc for labeled classes
    selected_mask = targets < args.num_labeled_classes
    pacc_labeled = pacc_fun(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
    writer.add_scalar('Test Stage {}/Pacc for 0-{}'.format(stage + 1, args.num_labeled_classes), pacc_labeled, epoch)

    if not no_accept:
        selected_mask_with_refuse = targets_with_refuse < args.num_labeled_classes
        pacc_labeled_with_refuse = pacc_fun_with_refuse(targets_with_refuse[selected_mask_with_refuse].astype(int), preds_with_refuse[selected_mask_with_refuse].astype(int))
        writer.add_scalar('Test Stage {}/Pacc with refuse for 0-{}'.format(stage + 1, args.num_labeled_classes), pacc_labeled_with_refuse, epoch)
        if len(pacc_with_refuse.shape) > 0:
            writer.add_scalar('Test Stage {}/N accepted for 0-{}'.format(stage + 1, args.num_labeled_classes), pacc_labeled_with_refuse.shape[0], epoch)

    # acc for unlabeled classes in each stage
    for s in range(stage):
        lower = args.num_labeled_classes + s * args.num_unlabeled_classes_per_stage
        upper = args.num_labeled_classes + (s + 1) * args.num_unlabeled_classes_per_stage
        selected_mask = (targets >= lower) * (targets < upper)
        selected_mask_with_refuse = (targets_with_refuse >= lower) * (targets_with_refuse < upper)

        pacc = pacc_fun(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
        writer.add_scalar('Test Stage {}/Pacc for {}-{}'.format(stage + 1, lower, upper), pacc, epoch)

        if not no_accept:
            pacc_with_refuse = pacc_fun_with_refuse(targets_with_refuse[selected_mask_with_refuse].astype(int), preds_with_refuse[selected_mask_with_refuse].astype(int))
            writer.add_scalar('Test Stage {}/Pacc with refuse for {}-{}'.format(stage + 1, lower, upper), pacc_with_refuse, epoch)
            if len(pacc_with_refuse.shape) > 0:
                writer.add_scalar('Test Stage {}/N accepted for {}-{}'.format(stage + 1, lower, upper), pacc_with_refuse.shape[0], epoch)


def test_stage(model, writer, stage, device, args):
    target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
    target_list_test_all = list(range(target_list_upper))
    test_dataset_all = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=target_list_test_all)
    test_loader_all = DataLoader(test_dataset_all, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test(model, test_loader_all, writer, stage, args.cluster_epochs + args.inc_ssl_epochs, device, args)


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


def reselect_exemplars(model, stage, args):
    m = args.budgets // model.n_classes
    if stage > 0:
        target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
        target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
        target_list_train = list(range(target_list_lower, target_list_upper))
        train_dataset = CIFAR100Data(root=args.dataset_root, split='train', aug='twice', target_list=target_list_train)

        model.exemplar_sets = model.exemplar_sets[:target_list_lower]
        model.exemplar_sets_aug = model.exemplar_sets_aug[:target_list_lower]

        model.construct_exemplar_set_for_new_classes(train_dataset, target_list_train, m, stage)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--ssl_lr', type=float, default=0.01)
    parser.add_argument('--ce_lr', type=float, default=0.01)
    parser.add_argument('--cluster_lr', type=float, default=0.01)
    parser.add_argument('--inc_ssl_lr', type=float, default=0.01)
    parser.add_argument('--together_lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--ssl_epochs', default=100, type=int)
    parser.add_argument('--ce_epochs', default=100, type=int)
    parser.add_argument('--cluster_epochs', default=100, type=int)
    parser.add_argument('--inc_ssl_epochs', default=100, type=int)
    parser.add_argument('--together_cluster_ssl', action='store_true', default=False)
    parser.add_argument('--together_epochs', default=100, type=int)
    parser.add_argument('--update_exemplar_epochs', default=50, type=int)
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

    parser.add_argument('--no_ce', action='store_true', default=False)
    parser.add_argument('--no_fc', action='store_true', default=False)

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

    parser.add_argument('--branch_feat', choices=['ema', 'conf', 'closest'], default='ema')
    parser.add_argument('--ema_beta', type=float, default=0.9)

    parser.add_argument('--norm_before_add', action='store_true', default=False)

    # parser.add_argument('--cls_type', choices=['proto', 'lp'], default='proto')
    # parser.add_argument('--lp_k', default=20, type=int)
    # parser.add_argument('--lp_alpha', default=0.99, type=float)

    # parser.add_argument('--update_means', action='store_true', default=False)

    parser.add_argument('--branch_depth', type=int, default=3)
    parser.add_argument('--no_sync', action='store_true', default=False)
    parser.add_argument('--all_branch', action='store_true', default=False)
    parser.add_argument('--same_branch', action='store_true', default=False)

    parser.add_argument('--pull_exemplar_features', action='store_true', default=False)
    parser.add_argument('--feature_dist_weight', type=float, default=1.0)

    parser.add_argument('--cluster_with_pef', action='store_true', default=False)
    parser.add_argument('--cluster_with_pef_weight', type=float, default=1.0)

    parser.add_argument('--pef_all', action='store_true', default=False)

    parser.add_argument('--no_refuse', action='store_true', default=False)
    parser.add_argument('--no_cache_means', action='store_true', default=False)

    parser.add_argument('--ssl_exemplars', action='store_true', default=False)
    parser.add_argument('--ssl_with_cluster', action='store_true', default=False)

    parser.add_argument('--reselect_exemplars', action='store_true', default=False)
    parser.add_argument('--reselect_exemplars_interval', type=int, default=10)

    parser.add_argument('--replace_wta', action='store_true', default=False)
    parser.add_argument('--rw_feat', action='store_true', default=False)
    parser.add_argument('--rw_label', action='store_true', default=False)
    parser.add_argument('--rw_graph', action='store_true', default=False)
    parser.add_argument('--rw_graph_k', type=int, default=10)

    parser.add_argument('--replace_mse', action='store_true', default=False)

    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--mixup_n', type=int, default=512)
    parser.add_argument('--mixup_alpha', type=float, default=0.5)
    parser.add_argument('--re_mixup', action='store_true', default=False)
    parser.add_argument('--batch_mixup', action='store_true', default=False)

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

    model = Baseline_v4_ema(args)
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
            print('Update ...')
            update_exemplars(model, stage, args)
            # if args.branch_depth == 3 and not args.no_sync:
            #     model.sync_weights()
            print('Sycn ...')
            model.sync_new_branches()
            print('Test First Stage ...')
            test_first_stage(model, writer, stage, device, args)
        else:
            train_cluster(model, writer, stage, device, args)
            update_exemplars(model, stage, args)
            if args.ssl_exemplars:
                train_ssl_with_exemplars(model, writer, stage, device, args)
            else:
                train_ssl(model, writer, stage, device, args)
            if not args.no_ce:
                train_ce(model, writer, stage, device, args)
            model.sync_weights(args.ema_beta)
            test_stage(model, writer, stage, device, args)
        model.increment_classes(args.num_unlabeled_classes_per_stage, device)
