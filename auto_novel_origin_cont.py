import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.util import BCE, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch
from utils import ramps
from models.resnet import ResNet, BasicBlock
from data.cifarloader import CIFAR100Loader, CIFAR100LoaderMix
from tqdm import tqdm
import numpy as np
import os


def train_IL(model, train_loader, args):
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup( epoch, args.rampup_length)
        for batch_idx, ((x, x_bar), _, idx) in enumerate(tqdm(train_loader)):
            x, x_bar = x.to(device), x_bar.to(device)
            output1, output2, feat = model(x)
            output1_bar, output2_bar, _ = model(x_bar)
            prob1, prob1_bar, prob2, prob2_bar = F.softmax( output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax( output2, dim=1), F.softmax(output2_bar, dim=1)

            # mask_lb = label < args.num_labeled_classes

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

            # loss_ce = criterion1(output1[mask_lb], label[mask_lb])

            label = output2.detach().max(1)[1]
            if not args.part:
                label = label + args.num_labeled_classes

            if not args.part:
                loss_ce_add = w * criterion1(output1, label) / args.rampup_coefficient * args.increment_coefficient
            else:
                loss_ce_add = w * criterion1(output1[:, -10:], label) / args.rampup_coefficient * args.increment_coefficient
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss( prob2, prob2_bar)

            loss = loss_bce + loss_ce_add + w * consistency_loss

            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Epoch: {} Avg Loss: {:.4f}'.format( epoch, loss_record.avg))

        # print('test on labeled classes')
        # args.head = 'head1'
        # test(model, labeled_eval_loader, args)
        # print('test on unlabeled classes')
        # args.head = 'head2'
        # test(model, unlabeled_eval_loader, args)


def test(model, test_loader, args):
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser( description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--step_size', default=340, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_unlabeled_classes', default=10, type=int)
    parser.add_argument('--num_labeled_classes', default=70, type=int)
    parser.add_argument('--dataset_root', type=str, default='./dataset')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument( '--warmup_model_dir', type=str, default= './data/experiments/auto_novel_origin/resnet_IL_cifar100.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--IL', action='store_true', default=True, help='w/ incremental learning')
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, svhn')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')

    parser.add_argument('--part', action='store_true', default=False)
    parser.add_argument('--fix_encoder', action='store_true', default=False)

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

    model = ResNet(BasicBlock, [2, 2, 2, 2], 80, 10).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if args.mode == 'train':
        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters():
            if args.fix_encoder:
                if 'head' not in name:
                    param.requires_grad = False
            else:
                if 'head' not in name and 'layer4' not in name:
                    param.requires_grad = False

    if args.dataset_name == 'cifar100':
        train_loader_80_90 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True,
                                            target_list=range(80, 90))
        train_loader_90_100 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True,
                                            target_list=range(80, 90))

        labeled_eval_loader_70 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                            target_list=range(70))

        unlabeled_eval_loader_test_70_80 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                                    target_list=range(70, 80))
        unlabeled_eval_loader_test_80_90 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                                    target_list=range(80, 90))
        unlabeled_eval_loader_test_90_100 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                                    target_list=range(90, 100))

        all_eval_loader_90 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                        target_list=range(90))
        all_eval_loader_100 = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False,
                                        target_list=range(100))






    save_weight = model.head1.weight.data.clone()
    save_bias = model.head1.bias.data.clone()
    model.head1 = nn.Linear(512, 90).to(device)
    model.head1.weight.data[:80] = save_weight
    model.head1.bias.data[:] = torch.min(save_bias) - 1.
    model.head1.bias.data[:80] = save_bias
    model.head2 = nn.Linear(512, 10).to(device)

    train_IL(model, train_loader_80_90, args)

    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader_70, args)
    print('test on unlabeled classes 70-80 (test split)')
    test(model, unlabeled_eval_loader_test_70_80, args)
    print('test on unlabeled classes 80-90 (test split)')
    test(model, unlabeled_eval_loader_test_80_90, args)
    print('test on all classes (test split)')
    test(model, all_eval_loader_90, args)


    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test_80_90, args)






    save_weight = model.head1.weight.data.clone()
    save_bias = model.head1.bias.data.clone()
    model.head1 = nn.Linear(512, 100).to(device)
    model.head1.weight.data[:90] = save_weight
    model.head1.bias.data[:] = torch.min(save_bias) - 1.
    model.head1.bias.data[:90] = save_bias
    model.head2 = nn.Linear(512, 10).to(device)

    train_IL(model, train_loader_90_100, args)

    torch.save(model.state_dict(), args.model_dir)
    print("model saved to {}.".format(args.model_dir))


    print('Evaluating on Head1')
    args.head = 'head1'
    print('test on labeled classes (test split)')
    test(model, labeled_eval_loader_70, args)
    print('test on unlabeled classes 70-80 (test split)')
    test(model, unlabeled_eval_loader_test_70_80, args)
    print('test on unlabeled classes 80-90 (test split)')
    test(model, unlabeled_eval_loader_test_80_90, args)
    print('test on unlabeled classes 90-100 (test split)')
    test(model, unlabeled_eval_loader_test_90_100, args)
    print('test on all classes (test split)')
    test(model, all_eval_loader_100, args)


    print('Evaluating on Head2')
    args.head = 'head2'
    print('test on unlabeled classes (test split)')
    test(model, unlabeled_eval_loader_test_90_100, args)