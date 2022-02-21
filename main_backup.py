import torch
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from utils.util import cluster_acc, seed_torch
from utils.util import WarmUpLR
from models.model import Air, NCE, modularity_loss, WTA
from data.cifarloader import CIFAR100LoaderMix, CIFAR100Loader
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from rich import print


def main(model, train_loader, labeled_eval_loader_test, unlabeled_eval_loader,
         unlabeled_eval_loader_test, args):
    optimizer = SGD(model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=0.9)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                milestones=[60, 120, 160],
                                                gamma=args.lr_decay_gamma)

    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    writer = SummaryWriter()
    for epoch in range(args.epochs):
        train(model, optimizer, exp_lr_scheduler, warmup_scheduler,
              train_loader, writer, epoch, args)

        print('\nTest on labeled classes')
        test(model, labeled_eval_loader_test, writer, epoch, 1, args)
        print('\nTest on unlabeled classes')
        test(model, unlabeled_eval_loader, writer, epoch, 2, args)

    torch.save(model.state_dict(), args.model_path)

    print('\nTest on labeled classes')
    test(model, labeled_eval_loader_test, writer, None, 1, args)
    print('\nTest on unlabeled classes')
    test(model, unlabeled_eval_loader_test, writer, None, 2, args)

    writer.close()


def train(model, optimizer, scheduler, warmup, train_loader, writer, epoch,
          args):
    print('\nEpoch: {}'.format(epoch))
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)

        optimizer.zero_grad()
        output, feature, z, adj, adj_recon = model(data)

        label_mask = target < args.num_labeled_classes

        nl = args.num_labeled_classes
        s = WTA(output, args.wta_k)

        loss_ce = F.cross_entropy(output[label_mask, :nl], target[label_mask])
        loss_nce_instance = NCE(
            feature,
            args.nce_temperature) if args.instance_nce else torch.tensor(0)
        loss_nce_structure = NCE(
            z, args.nce_temperature) if args.structure_nce else torch.tensor(0)
        # TODO bce
        loss_reconstruct = F.mse_loss(
            adj_recon, adj) if args.reconstruct else torch.tensor(0)
        loss_modularity = modularity_loss(
            output, adj) if args.modularity else torch.tensor(0)
        loss_bce = F.binary_cross_entropy(
            adj_recon.reshape(-1), s) if args.bce else torch.tensor(0) * 0.01

        loss = loss_ce + loss_nce_instance + loss_nce_structure + loss_reconstruct + loss_modularity + loss_bce

        loss.backward()
        optimizer.step()

        pred = output[label_mask, :nl].max(1)[1]
        train_acc = accuracy_score(target[label_mask].cpu().numpy(),
                                   pred.cpu().numpy())

        writer.add_scalar('train-acc', train_acc,
                          epoch * len(train_loader) + batch_idx)

        writer.add_scalar('train loss', loss.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('train loss ce', loss_ce.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('train loss nce_instance', loss_nce_instance.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('train loss nce_structure',
                          loss_nce_structure.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('train loss reconstruct', loss_reconstruct.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('train loss modularity', loss_modularity.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('train loss bce', loss_bce.item(),
                          epoch * len(train_loader) + batch_idx)

        if epoch < args.warm:
            warmup.step()
    if epoch >= args.warm:
        scheduler.step()


def test(model, eval_loader, writer, epoch, head, args):
    model.eval()
    with torch.no_grad():
        preds = np.array([], dtype=np.int64)
        targets = np.array([], dtype=np.int64)

        nl = args.num_labeled_classes

        for batch_idx, (data, target, _) in enumerate(eval_loader):
            data, target = data.to(args.device), target.to(args.device)
            output, _, _, _, _ = model(data)

            label_mask = target < nl

            target = target[label_mask] if head == 1 else target[~label_mask]
            targets = np.append(targets, target.cpu().numpy())

            output = output[label_mask, :nl] if head == 1 else output[
                ~label_mask, nl:]

            pred = output.max(1)[1]

            if head == 2:
                pred = pred + nl

            preds = np.append(preds, pred.cpu().numpy())

        if head == 1:
            l_acc = accuracy_score(targets, preds)
            print('\nTest set: Labeled Accuracy: {:.4f}'.format(l_acc))

            if epoch is not None:
                writer.add_scalar('l-acc', l_acc, epoch)
        else:
            u_acc = cluster_acc(targets, preds)
            u_nmi = nmi_score(targets, preds)
            u_ari = ari_score(targets, preds)
            print(
                '\nTest set: Unlabeled Accuracy: {:.4f}, NMI: {:.4f}, ARI: {:.4f}'
                .format(u_acc, u_nmi, u_ari))

            if epoch is not None:
                writer.add_scalar('u-acc', u_acc, epoch)
                writer.add_scalar('u-nmi', u_nmi, epoch)
                writer.add_scalar('u-ari', u_ari, epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Air')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.2)
    # parser.add_argument('--lr_decay_step', default=170, type=int)
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
                        choices=['resnet18', 'resnet34'])

    parser.add_argument('--knn_graph_k', default=5, type=int)
    parser.add_argument('--knn_dist', default='cosine', type=str)

    parser.add_argument('--nce_temperature', default=0.1, type=float)
    parser.add_argument('--wta_k', default=5, type=int)

    parser.add_argument('--instance_nce', action='store_true', default=False)
    parser.add_argument('--structure_nce', action='store_true', default=False)
    parser.add_argument('--reconstruct', action='store_true', default=False)
    parser.add_argument('--modularity', action='store_true', default=False)
    parser.add_argument('--bce', action='store_true', default=False)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    seed_torch(args.seed)

    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

    model_dir = os.path.join(args.exp_root, t)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_path = os.path.join(model_dir, 'model.pth')

    print('\nArgs')
    print(args)

    model = Air(args).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    mix_train_loader = CIFAR100LoaderMix(
        root=args.dataset_root,
        batch_size=args.batch_size,
        split='train',
        aug='once',
        shuffle=True,
        labeled_list=range(args.num_labeled_classes),
        unlabeled_list=range(args.num_labeled_classes, num_classes),
        num_workers=args.workers)
    unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root,
                                           batch_size=args.batch_size,
                                           split='train',
                                           aug=None,
                                           shuffle=False,
                                           target_list=range(
                                               args.num_labeled_classes,
                                               num_classes),
                                           num_workers=args.workers)
    unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root,
                                                batch_size=args.batch_size,
                                                split='test',
                                                aug=None,
                                                shuffle=False,
                                                target_list=range(
                                                    args.num_labeled_classes,
                                                    num_classes),
                                                num_workers=args.workers)
    labeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root,
                                              batch_size=args.batch_size,
                                              split='test',
                                              aug=None,
                                              shuffle=False,
                                              target_list=range(
                                                  args.num_labeled_classes),
                                              num_workers=args.workers)

    print('\nData Loaded')

    main(model, mix_train_loader, labeled_eval_loader_test,
         unlabeled_eval_loader, unlabeled_eval_loader_test, args)
