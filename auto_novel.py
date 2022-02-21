import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import BCE, PairEnum, cluster_acc, AverageMeter, seed_torch
from utils import ramps
from models.resnet import ResNet, BasicBlock
from data.cifarloader import CIFAR100Loader, CIFAR100LoaderMix, StageCIFAR100Loader, PARTITION_CONFIG, CIFAR100Data
from tqdm import tqdm
import numpy as np
import os
import copy
from models.lwf import MultiClassCrossEntropy
from models.si import init_si, update_si, update_omega, si_loss
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

def train_first_stage(model, train_loader, eval_loader, optimizer, exp_lr_scheduler, args, writer):

    if args.si:
        W, p_old = init_si(model)

    criterion1 = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup( epoch, args.rampup_length)
        for batch_idx, ((x, x_bar), label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar = F.softmax( output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax( output2, dim=1), F.softmax(output2_bar, dim=1)


            loss_ce = criterion1(output1, label)

            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss( prob2, prob2_bar)

            loss = loss_ce + w * consistency_loss

            if args.si:
                update_si(model, W, p_old)
                loss = loss + si_loss(model) * args.si_c

            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        writer.add_scalar('loss/train', loss_record.avg, epoch)

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('Test on classification head')
        args.head = 'head1'
        test(model, eval_loader, args, writer, epoch, classes='0-40')
        print('Test on cluster head')
        args.head = 'head2'
        test(model, eval_loader, args, writer, epoch, classes='0-40')

    if args.si:
        update_omega(model, W)


def train_IL(model, train_loader, eval_loader, optimizer, exp_lr_scheduler, stage_idx, n_known, args, writer, partition_config):
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    if args.lwf:
        prev_model = copy.deepcopy(model)
        prev_model.cuda()
        prev_model.eval()

    if args.si:
        W, p_old = init_si(model)

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        if args.lwf:
            loss_dist_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup( epoch, args.rampup_length)
        for batch_idx, ((x, x_bar), _, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)


            rank_feat = feat.detach()

            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :args.  topk], rank_idx2[:, :args.topk]

            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)

            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().to(device)
            target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2)
            _, prob2_ulb = PairEnum(prob2_bar)

            label = output2.detach().max(1)[1] + args.num_labeled_classes

            loss_ce_add = w * criterion1(output1, label) / args.rampup_coefficient * args.increment_coefficient
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)


            loss = loss_bce + loss_ce_add + w * consistency_loss

            if args.lwf:
                with torch.no_grad():
                    dist_target, _, _ = prev_model(x)
                logits_dist = output1[:, :n_known]
                dist_loss = MultiClassCrossEntropy(logits_dist, dist_target[:, :n_known], args.lwf_t)
                loss = loss + dist_loss * args.lwf_c

                loss_dist_record.update(dist_loss.item(), x.size(0))

            if args.si:
                update_si(model, W, p_old)
                loss = loss + si_loss(model) * args.si_c

            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        writer.add_scalar('loss/train', loss_record.avg, epoch + stage_idx * args.epochs)
        if args.lwf:
            writer.add_scalar('loss/train distill', loss_dist_record.avg, epoch + stage_idx * args.epochs)

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        for config in partition_config:
            testset = CIFAR100Data(root=args.dataset_root, split='test', aug=None, target_list=config[0], partial=True)
            loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
            print('Test on classification head {}'.foramt(config[2]))
            args.head = 'head1'
            # test(model, eval_loader, args, writer, epoch + stage_idx * args.epochs, config[2])
            test(model, loader, args, writer, epoch + stage_idx * args.epochs, config[2])
            print('Test on cluster head {}'.format(config[2]))
            args.head = 'head2'
            test(model, loader, args, writer, epoch + stage_idx * args.epochs, config[2])

        print('Test on classfication head')
        args.head = 'head1'
        test(model, eval_loader, args, writer, epoch + stage_idx * args.epochs)
        print('Test on cluster')
        args.head = 'head2'
        test(model, eval_loader, args, writer, epoch + stage_idx * args.epochs)


    if args.si:
        update_omega(model, W)


def test(model, test_loader, args, writer, epoch=-1, classes='all'):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output1, output2, _ = model(x)
        if args.head == 'head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score( targets, preds), ari_score(targets, preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    if epoch >= 0:
        writer.add_scalar('acc/test {} {}'.format(args.head, classes), acc, epoch)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=40, type=int)
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument( '--warmup_model_dir', type=str, default= './data/experiments/auto_novel_supervised_learning/resnet_rotnet.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='auto_novel')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--lwf', action='store_true', default=False)
    parser.add_argument('--lwf_t', default=2.0, type=float)
    parser.add_argument('--lwf_c', default=1.0, type=float)

    parser.add_argument('--si', action='store_true', default=False)
    parser.add_argument('--si_c', default=0.1, type=float)

    parser.add_argument('--exemplers', action='store_true', default=False)
    parser.add_argument('--budgets', default=2000, type=int)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir + '/' + '{}.pth'.format(args.model_name)

    print(args)

    model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # print(num_classes, args.num_labeled_classes, args.num_unlabeled_classes)

    comment = '_{}'.format(args.model_name)
    writer = SummaryWriter(comment=comment)

    if args.mode == 'train':
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False

    stages = ['stage 1', 'stage 2', 'stage 3', 'stage 4']

    if args.exemplers:
        exemplers_set = None

    for stage_idx, stage in enumerate(stages):
        print(stage)
        partition_config = PARTITION_CONFIG[stage]

        train_loader, train_dataset = StageCIFAR100Loader(args.dataset_root, args.batch_size, split='train',
                                           aug='twice', shuffle=True,
                                           partition_config=partition_config)
        test_loader, _ = StageCIFAR100Loader(args.dataset_root, args.batch_size, split='test',
                                          aug=None, shuffle=False,
                                          partition_config=partition_config)


        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        if stage_idx == 0:
            train_first_stage(model, train_loader, test_loader, optimizer, exp_lr_scheduler, args, writer)

            if args.exemplers:
                indices = np.random.choice(len(train_dataset), args.budgets, replace=False)
                # TODO
                exemplers_set = None
        else:
            cur_num_classes = PARTITION_CONFIG['num classes'][stage_idx]
            pre_num_classes = PARTITION_CONFIG['num classes'][stage_idx - 1]
            cur_new_classes = PARTITION_CONFIG['new classes'][stage_idx]
            pre_new_classes = PARTITION_CONFIG['new classes'][stage_idx - 1]

            save_weight_1 = model.head1.weight.data.clone()
            save_bias_1 = model.head1.bias.data.clone()

            feature_dim_1 = model.head1.in_features

            model.head1 = nn.Linear(feature_dim_1, cur_num_classes).to(device)
            model.head1.weight.data[:pre_num_classes] = save_weight_1
            model.head1.bias.data[:] = torch.min(save_bias_1) - 1
            model.head1.bias.data[:pre_num_classes] = save_bias_1

            if stage_idx > 1:
                save_weight_2 = model.head2.weight.data.clone()
                save_bias_2 = model.head2.bias.data.clone()

            feature_dim_2 = model.head2.in_features

            model.head2 = nn.Linear(feature_dim_2, cur_new_classes).to(device)

            if stage_idx > 1:
                model.head2.weight.data[:pre_new_classes] = save_weight_2
                model.head2.bias.data[:] = torch.min(save_bias_2) - 1
                model.head2.bias.data[:pre_new_classes] = save_bias_2
            else:
                model.head2.weight.data[:] = model.head1.weight.data.clone()
                model.head2.bias.data[:] = model.head1.bias.data.clone()


            optimizer = SGD(model.parameters(), lr=args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

            train_IL(model, train_loader, test_loader, optimizer, exp_lr_scheduler,
                     stage_idx, pre_num_classes, args, writer)

    torch.save(model.state_dict(), args.model_dir)

    unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list=range( args.num_labeled_classes, num_classes))

    unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(args.num_labeled_classes, num_classes))

    labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range( args.num_labeled_classes))

    all_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(num_classes))


    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader, args, writer)
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, args, writer)
    print('test on all classes (test split)')
    test(model, all_eval_loader, args, writer)
    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (train split)')
    test(model, unlabeled_eval_loader, args, writer)
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test, args, writer)
