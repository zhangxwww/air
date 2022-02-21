import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, dataloader
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import BCE, PairEnum, accuracy, cluster_acc, AverageMeter, seed_torch
from utils import ramps
from models.resnet import ResNet, BasicBlock
from models.baseline import Baseline
from data.cifarloader import CIFAR100Loader, CIFAR100LoaderMix, StageCIFAR100Loader, PARTITION_CONFIG, CIFAR100Data
from data.rotationloader import DataLoader, GenericDataset
from tqdm import tqdm
import numpy as np
import os
import copy
from models.lwf import MultiClassCrossEntropy
from models.si import init_si, update_si, update_omega, si_loss
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True

def ssl(model, writer, stage, device, args):
    train_dataset = GenericDataset(args.dataset_name, 'train', dataset_root=args.dataset_root)
    test_dataset = GenericDataset(args.dataset_name, 'test', dataset_root=args.dataset_root)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    lr = args.lr if stage == 0 else args.lr * 0.1
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 210, 160, 200], gamma=0.2)

    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    save_dir = os.path.join(args.model_dir, 'ssl_stage_{}.pth'.format(stage + 1))

    lwf = (args.lwf or args.lwf_ssl) and stage > 0

    if lwf:
        prev_model = copy.deepcopy(model).to(device)
        prev_model.eval()

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        dist_loss_record = AverageMeter()
        acc_record = AverageMeter()

        exp_lr_scheduler.step()
        model.train()
        for batch_idx, (data, label) in enumerate(train_loader(epoch)):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output, _, _, _ = model(data)
            loss = criterion(output, label)

            if args.lwf and stage > 0:
                with torch.no_grad():
                    distill_output, _, _, _ = prev_model(data)
                distill_loss = MultiClassCrossEntropy(output, distill_output, args.lwf_t)
                loss = loss + distill_loss * args.lwf_c

                dist_loss_record.update(distill_loss.item(), data.size(0))


            acc = accuracy(output, label)
            acc_record.update(acc[0].item(), data.size(0))
            loss_record.update(loss.item(), data.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train SSL Stage {} Epoch: {} \t Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(stage + 1, epoch, loss_record.avg, acc_record.avg))
        writer.add_scalar('Train Stage {}/SSL/Loss'.format(stage + 1), loss_record.avg, epoch)
        writer.add_scalar('Train Stage {}/SSL/Acc'.format(stage + 1), acc_record.avg, epoch)
        if lwf:
            writer.add_scalar('Train Stage {}/SSL/Distill Loss'.format(stage + 1), dist_loss_record.avg, epoch)

        acc_record.reset()
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(test_loader(epoch)):
                data, label = data.to(device), label.to(device)
                output, _, _, _ = model(data)
                acc = accuracy(output, label)
                acc_record.update(acc[0].item(), data.size(0))

            print('Test SSL Stage {} Acc: {:.4f}'.format(stage + 1, acc_record.avg))
            writer.add_scalar('Test Stage {}/SSL/Acc'.format(stage + 1), acc_record.avg, epoch)

        if acc_record.avg > best_acc:
            best_acc = acc_record.avg
            torch.save(model.state_dict(), save_dir)

    return save_dir


def train_first_stage(model, writer, stage, device, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)

    # if args.si:
    #     W, p_old = init_si(model)

    criterion1 = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        acc_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup( epoch, args.rampup_length)

        target_list = list(range(args.num_labeled_classes))
        train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                      aug='twice', shuffle=True, target_list=list(target_list))
        test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                     aug=None, shuffle=False, target_list=list(target_list))

        for batch_idx, ((x, x_bar), label, idx) in enumerate(train_loader):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            _, output, _, _ = model(x)
            _, output_bar, _, _ = model(x_bar)
            prob1, prob1_bar = F.softmax(output, dim=1), F.softmax(output_bar, dim=1)

            loss_ce = criterion1(output, label)

            consistency_loss = F.mse_loss(prob1, prob1_bar)

            loss = loss_ce + w * consistency_loss

            # if args.si:
            #     update_si(model, W, p_old)
            #     loss = loss + si_loss(model) * args.si_c

            acc = accuracy(output, label)

            loss_record.update(loss.item(), x.size(0))
            acc_record.update(acc[0].item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train CL Stage {} Epoch: {} Avg Loss: {:.4f} \t Avg Acc: {:.4f}'.format(stage + 1, epoch, loss_record.avg, acc_record.avg))

        writer.add_scalar('Train Stage {}/CL/Loss'.format(stage + 1), loss_record.avg, epoch)
        writer.add_scalar('Train Stage {}/CL/Acc'.format(stage + 1), acc_record.avg, epoch)

        acc_record.reset()
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, label, idx) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                _, output, _, _ = model(data)
                acc = accuracy(output, label)
                acc_record.update(acc[0].item(), data.size(0))

            print('Test CL Stage {} Acc: {:.4f}'.format(stage + 1, acc_record.avg))
            writer.add_scalar('Test Stage {}/CL/Acc'.format(stage + 1), acc_record.avg, epoch)

    save_dir = os.path.join(args.model_dir, 'cl_stage_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)

    return save_dir

    # if args.si:
    #     update_omega(model, W)


def train_IL(model, writer, stage, device, args):

    lr = args.lr * 0.1
    optimizer = SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=args.gamma)


    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    lwf = args.lwf or args.lwf_cluster

    if lwf:
        prev_model = copy.deepcopy(model)
        prev_model.cuda()
        prev_model.eval()

    # if args.si:
    #     W, p_old = init_si(model)

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        dist_loss_record = AverageMeter()
        ce_add_loss_record = AverageMeter()

        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)

        target_list_lower = args.num_labeled_classes + (stage - 1) * args.num_unlabeled_classes_per_stage
        target_list_upper = args.num_labeled_classes + stage * args.num_unlabeled_classes_per_stage
        target_list_train = list(range(target_list_lower, target_list_upper))
        target_list_test = list(range(target_list_upper))

        train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                      aug='twice', shuffle=True, target_list=target_list_train)
        test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                      aug=None, shuffle=False, target_list=target_list_test)

        preds = np.array([])
        targets = np.array([])
        for batch_idx, ((x, x_bar), label, idx) in enumerate(train_loader):
            # label is only used for computing training acc
            x, x_bar = x.to(device), x_bar.to(device)
            _, classify_output, cluster_output, feat = model(x)
            _, classify_output_bar, cluster_output_bar, feat = model(x_bar)

            prob1 = F.softmax(classify_output, dim=1)
            prob2 = F.softmax(cluster_output, dim=1)
            prob1_bar = F.softmax(classify_output_bar, dim=1)
            prob2_bar = F.softmax(cluster_output_bar, dim=1)

            rank_feat = feat.detach()

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

            if args.ce_add_full:
                label = cluster_output.detach().max(1)[1] + target_list_lower
                loss_ce_add = w * criterion1(classify_output, label) / args.rampup_coefficient * args.increment_coefficient

            else:
                label = cluster_output.detach().max(1)[1]
                loss_ce_add = w * criterion1(classify_output[:, target_list_lower:], label) / args.rampup_coefficient * args.increment_coefficient

            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)

            loss = loss_bce + w * consistency_loss

            if args.use_ce_add:
                if not args.delay_ce_add or args.delay_ce_add_epoch < epoch:
                    loss = loss + loss_ce_add

                    ce_add_loss_record.update(loss_ce_add.item(), x.size(0))

            if lwf:
                with torch.no_grad():
                    _, distill_output, _, _ = prev_model(x)
                logits_dist = classify_output[:, :target_list_lower]
                target_dist = distill_output[:, :target_list_lower]
                dist_loss = MultiClassCrossEntropy(logits_dist, target_dist, args.lwf_t)
                loss = loss + dist_loss * args.lwf_c

                dist_loss_record.update(dist_loss.item(), x.size(0))

            # if args.si:
            #     update_si(model, W, p_old)
            #     loss = loss + si_loss(model) * args.si_c

            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, classify_output.max(1)[1].cpu().numpy())

            loss_record.update(loss.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = cluster_acc(targets.astype(int), preds.astype(int))


        writer.add_scalar('Train Stage {}/CL/Loss'.format(stage + 1), loss_record.avg, epoch)
        writer.add_scalar('Train Stage {}/CL/Acc'.format(stage + 1), acc, epoch)
        if lwf:
            writer.add_scalar('Train Stage {}/CL/Distill Loss'.format(stage + 1), dist_loss_record.avg, epoch)
        if args.use_ce_add:
            writer.add_scalar('Train Stage {}/CL/CE Add Loss'.format(stage + 1), ce_add_loss_record.avg, epoch)

        print('Train CL Stage {} Epoch: {} Avg Loss: {:.4f}'.format(stage + 1, epoch, loss_record.avg))

        model.eval()
        preds = np.array([])
        cluster_preds = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for batch_idx, (data, label, idx) in enumerate(test_loader):
                data, label = data.to(device), label.to(device)
                _, output, cluster_output, _ = model(data)
                targets = np.append(targets, label.cpu().numpy())
                preds = np.append(preds, output.max(1)[1].cpu().numpy())
                cluster_preds = np.append(cluster_preds, cluster_output.max(1)[1].cpu().numpy())
        acc = cluster_acc(targets.astype(int), preds.astype(int))

        cluster_preds = cluster_preds + target_list_lower

        print('Test CL Stage {} Avg Acc: {:.4f}'.format(stage + 1, acc))
        writer.add_scalar('Test Stage {}/CL/Acc'.format(stage + 1), acc, epoch)

        # acc for labeled classes
        selected_mask = targets < args.num_labeled_classes
        acc_labeled = cluster_acc(targets[selected_mask].astype(int), preds[selected_mask].astype(int))
        # acc_labeled = accuracy(targets[selected_mask].astype(int), preds[selected_mask].astype(int))[0]

        print('Test CL Stage {} Avg Acc for 0-{}: {:.4f}'.format(stage + 1, args.num_labeled_classes, acc_labeled))
        writer.add_scalar('Test Stage {}/CL/Acc for Labeled'.format(stage + 1), acc_labeled, epoch)

        # acc for unlabeled classes in each stage
        for s in range(stage):
            lower = args.num_labeled_classes + s * args.num_unlabeled_classes_per_stage
            upper = args.num_labeled_classes + (s + 1) * args.num_unlabeled_classes_per_stage
            selected_mask = (targets >= lower) * (targets < upper)
            acc = cluster_acc(targets[selected_mask].astype(int), preds[selected_mask].astype(int))

            print('Test CL Stage {} Avg Acc for {}-{}: {:.4f}'.format(stage + 1, lower, upper, acc))
            writer.add_scalar('Test Stage {}/CL/Acc for {}-{}'.format(stage + 1, lower, upper), acc, epoch)

        # acc for cluster head:
        selected_mask = (targets >= target_list_lower) * (targets < target_list_upper)
        acc_cluster = cluster_acc(targets[selected_mask].astype(int), cluster_preds[selected_mask].astype(int))

        print('Test CL Stage {} Avg Acc for Cluster Head: {:.4f}'.format(stage + 1, acc_cluster))
        writer.add_scalar('Test Stage {}/CL/Acc for Cluster Head'.format(stage + 1), acc_cluster, epoch)

    # if args.si:
    #     update_omega(model, W)

    save_dir = os.path.join(args.model_dir, 'cl_stage_{}.pth'.format(stage + 1))
    torch.save(model.state_dict(), save_dir)

    return save_dir



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
    parser.add_argument('--num_unlabeled_classes_per_stage', default=20, type=int)
    parser.add_argument('--num_labeled_classes', default=40, type=int)
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

    parser.add_argument('--use_ce_add', action='store_true', default=False)
    parser.add_argument('--ce_add_full', action='store_true', default=False)

    parser.add_argument('--lwf', action='store_true', default=False)
    parser.add_argument('--lwf_ssl', action='store_true', default=False)
    parser.add_argument('--lwf_cluster', action='store_true', default=False)
    parser.add_argument('--lwf_t', default=2.0, type=float)
    parser.add_argument('--lwf_c', default=1.0, type=float)

    parser.add_argument('--skip', action='store_true', default=False)
    parser.add_argument('--start_stage', default=0, type=int)
    parser.add_argument('--skip_ssl', action='store_true', default=False)

    parser.add_argument('--delay_ce_add', action='store_true', default=False)
    parser.add_argument('--delay_ce_add_epoch', default=100, type=int)

    # parser.add_argument('--si', action='store_true', default=False)
    # parser.add_argument('--si_c', default=0.1, type=float)

    # parser.add_argument('--exemplers', action='store_true', default=False)
    # parser.add_argument('--budgets', default=2000, type=int)

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

    # model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)
    model = Baseline(args)
    model = model.to(device)

    # num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # print(num_classes, args.num_labeled_classes, args.num_unlabeled_classes)

    comment = '_{}'.format(args.model_name)
    writer = SummaryWriter(comment=comment)

    # if args.mode == 'train':
    #     state_dict = torch.load(args.warmup_model_dir)
    #     model.load_state_dict(state_dict, strict=False)
    #     for name, param in model.named_parameters():
    #         if 'head' not in name and 'layer4' not in name:
    #             param.requires_grad = False


    # if args.exemplers:
    #     exemplers_set = None

    stages = ['stage 1', 'stage 2', 'stage 3', 'stage 4']

    for stage_idx, stage in enumerate(stages):
        # partition_config = PARTITION_CONFIG[stage]

        # train_loader, train_dataset = StageCIFAR100Loader(args.dataset_root, args.batch_size, split='train',
        #                                    aug='twice', shuffle=True,
        #                                    partition_config=partition_config)
        # test_loader, _ = StageCIFAR100Loader(args.dataset_root, args.batch_size, split='test',
        #                                   aug=None, shuffle=False,
        #                                   partition_config=partition_config)

        if not args.skip or args.start_stage <= stage_idx:

            model.unfix_first_three_layers()
            ssl(model, writer, stage_idx, device, args)
            model.fix_first_three_layers()
            if stage_idx == 0:
                train_first_stage(model, writer, stage_idx, device, args)
            else:
                train_IL(model, writer, stage_idx, device, args)

        model.increment_classes(args.num_unlabeled_classes_per_stage, device)

        if args.skip and args.start_stage == stage_idx + 1:
            state_dict = torch.load(os.path.join(args.exp_root, args.model_name, 'cl_stage_{}.pth').format(args.start_stage + 1))
            model.load_state_dict(state_dict, strict=True)


        # train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='')


        # optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        # if stage_idx == 0:
        #     train_first_stage(model, train_loader, test_loader, optimizer, exp_lr_scheduler, args, writer)

        #     if args.exemplers:
        #         indices = np.random.choice(len(train_dataset), args.budgets, replace=False)
        #         # TODO
        #         exemplers_set = None
        # else:
        #     cur_num_classes = PARTITION_CONFIG['num classes'][stage_idx]
        #     pre_num_classes = PARTITION_CONFIG['num classes'][stage_idx - 1]
        #     cur_new_classes = PARTITION_CONFIG['new classes'][stage_idx]
        #     pre_new_classes = PARTITION_CONFIG['new classes'][stage_idx - 1]

        #     save_weight_1 = model.head1.weight.data.clone()
        #     save_bias_1 = model.head1.bias.data.clone()

        #     feature_dim_1 = model.head1.in_features

        #     model.head1 = nn.Linear(feature_dim_1, cur_num_classes).to(device)
        #     model.head1.weight.data[:pre_num_classes] = save_weight_1
        #     model.head1.bias.data[:] = torch.min(save_bias_1) - 1
        #     model.head1.bias.data[:pre_num_classes] = save_bias_1

        #     if stage_idx > 1:
        #         save_weight_2 = model.head2.weight.data.clone()
        #         save_bias_2 = model.head2.bias.data.clone()

        #     feature_dim_2 = model.head2.in_features

        #     model.head2 = nn.Linear(feature_dim_2, cur_new_classes).to(device)

        #     if stage_idx > 1:
        #         model.head2.weight.data[:pre_new_classes] = save_weight_2
        #         model.head2.bias.data[:] = torch.min(save_bias_2) - 1
        #         model.head2.bias.data[:pre_new_classes] = save_bias_2
        #     else:
        #         model.head2.weight.data[:] = model.head1.weight.data.clone()
        #         model.head2.bias.data[:] = model.head1.bias.data.clone()


        #     optimizer = SGD(model.parameters(), lr=args.lr * 0.1, momentum=args.momentum, weight_decay=args.weight_decay)
        #     exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        #     train_IL(model, train_loader, test_loader, optimizer, exp_lr_scheduler,
        #              stage_idx, pre_num_classes, args, writer)

    # torch.save(model.state_dict(), args.model_dir)

    # unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list=range( args.num_labeled_classes, num_classes))

    # unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(args.num_labeled_classes, num_classes))

    # labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range( args.num_labeled_classes))

    # all_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list=range(num_classes))


    # print('Evaluating on Head1')
    # args.head = 'head1'
    # print('test on labeled classes (test split)')
    # test(model, labeled_eval_loader, args, writer)
    # print('test on unlabeled classes (test split)')
    # test(model, unlabeled_eval_loader_test, args, writer)
    # print('test on all classes (test split)')
    # test(model, all_eval_loader, args, writer)
    # print('Evaluating on Head2')
    # args.head = 'head2'
    # print('test on unlabeled classes (train split)')
    # test(model, unlabeled_eval_loader, args, writer)
    # print('test on unlabeled classes (test split)')
    # test(model, unlabeled_eval_loader_test, args, writer)
