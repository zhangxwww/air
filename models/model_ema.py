import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .moco import ModelMoCo, ModelBase
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from torch.utils.data import DataLoader
import copy
import numpy as np


class Baseline_v4_ema(nn.Module):
    def __init__(self, args):
        super(Baseline_v4_ema, self).__init__()

        self.args = args
        self.moco = ModelMoCo(dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, bn_splits=args.bn_splits)
        self.feature_dim = args.moco_dim

        self.ema_branch = ModelBase(feature_dim=args.moco_dim, arch='resnet18', bn_splits=args.bn_splits)
        for p in self.ema_branch.parameters():
            p.requires_grad = False
            p.grad = None


        # self.cluster_head = nn.ModuleList([nn.Linear(self.feature_dim, args.num_unlabeled_classes_per_stage, bias=False)\
            # for _ in range(4)])
        self.cluster_head = nn.Linear(self.feature_dim, args.num_unlabeled_classes_per_stage, bias=False)
        self.bn = nn.BatchNorm1d(self.feature_dim, momentum=0.01)
        self.fc = nn.Linear(self.feature_dim, args.num_labeled_classes, bias=False)

        self.exemplar_sets = []
        self.exemplar_sets_aug = []

        self.compute_means = True
        self.exemplar_means = []
        self.means_labels = []

        self.n_classes = args.num_labeled_classes
        self.already_calc_means_for_stage = [False, False, False, False]

    def forward(self, x, stage):
        assert False
        x = self.moco.encoder_q(x, stage)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

    def ssl_stage(self, x1, x2):
        loss = self.moco(x1, x2)
        return loss

    def ce_stage(self, x, stage):
        feature = self.moco.encoder_q(x)
        out = self.bn(feature)
        out = F.relu(out)
        classification = self.fc(out)
        cluster = self.cluster_head(out)
        return classification, cluster, feature

    def increment_classes(self, n, device):
        #self.cluster_head = nn.Linear(self.feature_dim, self.args.num_unlabeled_classes_per_stage, bias=False).to(device)

        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False).to(device)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

        self.cluster_head = nn.Linear(self.feature_dim, self.args.num_unlabeled_classes_per_stage, bias=False).to(device)

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
        for y, P_y in enumerate(self.exemplar_sets_aug):
            self.exemplar_sets_aug[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, dataset_aug, n, stage):
        mode = self.training
        self.eval()

        n_max = len(dataset)
        dataset = torch.utils.data.TensorDataset(dataset)
        dataset_aug = torch.utils.data.TensorDataset(dataset_aug)
        exemplar_sets = []
        exemplar_sets_aug = []
        if n_max > 0:
            if self.args.herding:
                first_entry = True
                dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
                for image in dataloader:
                    image = torch.cat(image, dim=0)
                    image = image.cuda()
                    with torch.no_grad():
                        feature = self.moco.encoder_q(image).cpu()
                    if first_entry:
                        features = feature
                        first_entry = False
                    else:
                        features = torch.cat([features, feature], dim=0)
                features = F.normalize(features, dim=1, p=2)

                class_mean = torch.mean(features, dim=0, keepdim=True)
                class_mean = F.normalize(class_mean, dim=1, p=2)

                exemplar_features = torch.zeros_like(features[:min(n, n_max)])
                list_of_selected = []
                for k in range(min(n, n_max)):
                    if k > 0:
                        exemplar_sum = torch.sum(exemplar_features[:k], dim=0).unsqueeze(0)
                        features_means = (features + exemplar_sum) / (k + 1)
                        features_dists = features_means - class_mean
                    else:
                        features_dists = features - class_mean
                    index_selected = np.argmin(torch.norm(features_dists, p=2, dim=1))
                    if index_selected in list_of_selected:
                        raise ValueError('Exemplars should not be repeated')
                    list_of_selected.append(index_selected)

                    exemplar_sets.append(dataset[index_selected][0].numpy())
                    exemplar_features[k] = copy.deepcopy(features[index_selected])

                    exemplar_sets_aug.append(dataset_aug[index_selected][0].numpy())

                    features[index_selected] = features[index_selected] + 10000
            else:
                indeces_selected = np.random.choice(n_max, min(n, n_max), replace=False)
                for k in indeces_selected:
                    exemplar_sets.append(dataset[k][0].numpy())
                    exemplar_sets_aug.append(dataset_aug[k][0].numpy())

        self.exemplar_sets.append(np.array(exemplar_sets))
        self.exemplar_sets_aug.append(np.array(exemplar_sets_aug))

        self.train(mode=mode)

    def construct_exemplar_set_for_new_classes(self, dataset, target_list, m, stage):
        mode = self.training
        self.eval()

        loader = DataLoader(dataset, batch_size=128, shuffle=True)

        data = []
        data_aug = []
        pseudo = []
        logits = []

        with torch.no_grad():
            for (x, aug), _, _ in loader:
                x, aug = x.cuda(), aug.cuda()
                _, cluster, _ = self.ce_stage(x, stage)
                out = cluster.argmax(dim=1) + target_list[0]
                data.append(x)
                data_aug.append(aug)
                pseudo.append(out.detach().clone())
                logits.append(cluster.detach().clone())
        data = torch.cat(data, dim=0).cpu()
        data_aug = torch.cat(data_aug, dim=0).cpu()
        pseudo = torch.cat(pseudo, dim=0).cpu()
        logits = torch.cat(logits, dim=0).cpu()

        if self.args.geo:
            n = len(dataset)
            k = int(n * self.args.geo_percent)
            k = min(k, n)
            dist_func = euclidean_distances if self.args.geo_dist == 'euclidean' else cosine_distances
            dist = dist_func(logits.cpu().detach().numpy(), logits.cpu().detach().numpy())
            dist = torch.tensor(dist).to(logits.device)
            values, _ = torch.topk(dist, self.args.geo_k + 1, largest=False, sorted=True)
            max_dis_in_k_neighbors = values[:, -1]
            _, min_max_indices_in_k_neighbors = torch.topk(max_dis_in_k_neighbors, k, largest=False)

            data = data[min_max_indices_in_k_neighbors]
            data_aug = data_aug[min_max_indices_in_k_neighbors]
            pseudo = pseudo[min_max_indices_in_k_neighbors]
            logits = logits[min_max_indices_in_k_neighbors]


        if self.args.discovery_confidence and self.args.confidence_type == 'all':
            n = len(dataset)
            k = int(n * self.args.discovery_confidence_percent)
            k = min(k, n)
            confidence = F.softmax(logits, dim=1)
            value, _ = confidence.max(1)
            _, topk = value.topk(k)
            data = data[topk]
            data_aug = data_aug[topk]
            pseudo = pseudo[topk]
            logits = logits[topk]

        if self.args.discovery_confidence and self.args.confidence_type == 'threshold':
            confidence = F.softmax(logits, dim=1)
            value, _ = confidence.max(1)
            larger_than_threshold = value > self.args.discovery_confidence_threshold
            data = data[larger_than_threshold]
            data_aug = data_aug[larger_than_threshold]
            pseudo = pseudo[larger_than_threshold]
            logits = logits[larger_than_threshold]

        if self.args.geo_print_statistics:
            print('Geo statistic: {}'.format(data.shape[0]))

        for y in target_list:
            selected = pseudo == y
            data_selected = data[selected]
            data_aug_selected = data_aug[selected]
            logits_selected = logits[selected]

            if self.args.discovery_confidence and self.args.confidence_type == 'class':
                n = data_selected.shape[0]
                k = int(n * self.args.discovery_confidence_percent)
                k = min(k, n)
                confidence = F.softmax(logits_selected, dim=1)
                value, _ = confidence.max(1)

                _, topk = value.topk(k)
                data_selected = data_selected[topk]
                data_aug_selected = data_aug_selected[topk]

            self.construct_exemplar_set(data_selected, data_aug_selected, m, stage)

        self.train(mode=mode)

    def construct_exemplar_set_for_new_classes_with_label(self, dataset, target_list, m):
        loader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True)
        data = []
        data_aug = []
        targets = []
        for (x, aug), y, _ in loader:
            data.append(x)
            data_aug.append(aug)
            targets.append(y)
        data = torch.cat(data, dim=0).cuda()
        data_aug = torch.cat(data_aug, dim=0).cuda()
        targets = torch.cat(targets).cuda()
        for t in target_list:
            selected = targets == t
            data_in_t = data[selected]
            data_aug_in_t = data_aug[selected]
            self.exemplar_sets.append(data_in_t[:m].cpu().numpy())
            self.exemplar_sets_aug.append(data_aug_in_t[:m].cpu().numpy())

    @torch.no_grad()
    def calculate_exemplar_means(self, device, stage):
        #if self.compute_means:
        exemplar_means = []
        means_labels = []

        if self.args.no_cache_means:
            for l, P_y in enumerate(self.exemplar_sets):
                exemplars = []
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                if len(exemplars) == 0:
                    continue
                exemplars = torch.stack(exemplars).to(device)
                with torch.no_grad():
                    if l < 70:
                        features = self.ema_branch(exemplars)
                    else:
                        features = self.moco.encoder_q(exemplars)
                features = F.normalize(features, dim=1, p=2)
                mu_y = features.mean(0, keepdim=True)
                mu_y = F.normalize(mu_y, dim=1, p=2)
                exemplar_means.append(mu_y.squeeze())
                means_labels.append(l)

            self.exemplar_means = exemplar_means
            self.means_labels = means_labels

        return self.exemplar_means

    @torch.no_grad()
    def classify(self, x, stage):
        mode = self.training
        self.eval()

        with torch.no_grad():
            self.calculate_exemplar_means(x.device, stage)

        if self.args.dist == 'euclidean':
            dist_func = euclidean_distances
        elif self.args.dist == 'cosine':
            dist_func = cosine_distances

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)        # (n_classes, feature_dim)

        if self.args.branch_feat in ['ema', 'conf']:
            if self.args.branch_feat == 'ema':
                with torch.no_grad():
                    feature = self.moco.encoder_q(x, 0)
                    if self.args.norm_before_add:
                        feature = F.normalize(feature, dim=1, p=2) # (batch_size, feature_dim)
                    for s in range(1, stage + 1):
                        branch_feature = self.moco.encoder_q(x, s)
                        if self.args.norm_before_add:
                            branch_feature = F.normalize(branch_feature, dim=1, p=2)
                        feature = feature * self.args.ema_beta + branch_feature * (1 - self.args.ema_beta)
                    feature = F.normalize(feature, dim=1, p=2) # (batch_size, feature_dim)
            elif self.args.branch_feat == 'conf':
                with torch.no_grad():
                    cluster = []
                    branch_feature = []
                    for b in range(stage + 1):
                        _, c, b = self.ce_stage(x, b)
                        cluster.append(c)
                        branch_feature.append(b)
                    cluster = torch.stack(cluster, dim=0).softmax(dim=2)  # b, n, c
                    branch_feature = torch.stack(branch_feature, dim=0)   # b, n, d
                    conf_of_each_branch, _ = cluster.max(dim=2)           # b, n
                    _, max_conf_branch_index = conf_of_each_branch.max(dim=0)  # n
                    feature = branch_feature[max_conf_branch_index, torch.arange(x.shape[0]), :]
                    feature = F.normalize(feature, dim=1, p=2) # (batch_size, feature_dim)

            dists = dist_func(feature.cpu().numpy(), means.cpu().numpy())   # (batch_size, n_classes)
            dists = torch.from_numpy(dists).to(x.device)

        elif self.args.branch_feat == 'closest':
            feature1 = F.normalize(self.ema_branch(x), dim=1, p=2)
            feature2 = F.normalize(self.moco.encoder_q(x), dim=1, p=2)
            dist1 = dist_func(feature1.cpu().numpy(), means.cpu().numpy())
            dist2 = dist_func(feature2.cpu().numpy(), means.cpu().numpy())
            dist1 = torch.from_numpy(dist1).to(x.device)
            dist2 = torch.from_numpy(dist2).to(x.device)
            branch_dists = torch.stack([dist1, dist2], dim=0)

            values, _ = branch_dists.min(dim=2)
            _, values_indices = values.min(dim=0)
            dists = branch_dists[values_indices, torch.arange(x.shape[0]), :]

        mean_labels = torch.tensor(self.means_labels).long().to(x.device)
        value, preds = dists.min(1)
        # TODO check ?
        preds = mean_labels[preds]

        n_proto = means.shape[0]
        proto_dist = dist_func(means.cpu().numpy(), means.cpu().numpy())
        avg_proto_dist = proto_dist.sum() / (n_proto * (n_proto - 1))

        thres2 = avg_proto_dist * self.args.thres2_ratio

        refuse_index = value >= thres2

        preds_with_refuse = None
        if stage > 0 and (~refuse_index).sum() > 0:
            preds_with_refuse = dists[~refuse_index].argmin(1)
            preds_with_refuse = mean_labels[preds_with_refuse]

        self.train(mode=mode)

        return preds, preds_with_refuse, refuse_index


    @torch.no_grad()
    def pseudo_labeling_and_combine_with_exemplars(self, dataset, target_list, branch, device, args):

        n_classes = target_list[-1] + 1
        if dataset is not None:
            loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
            data = []
            data_aug = []
            pseudo = []

            prototypes = self.calculate_exemplar_means(device, branch)
            prototypes = torch.stack(prototypes, dim=0)

            if self.args.dist == 'euclidean':
                dist_func = euclidean_distances
            elif self.args.dist == 'cosine':
                dist_func = cosine_distances

            proto_dist = dist_func(prototypes.cpu().numpy(), prototypes.cpu().numpy())
            proto_dist = torch.from_numpy(proto_dist).to(device)
            n_proto = prototypes.shape[0]
            avg_proto_dist = proto_dist.sum() / (n_proto * (n_proto - 1))

            thres1 = avg_proto_dist * self.args.thres1_ratio
            thres2 = avg_proto_dist * self.args.thres2_ratio

            label_2_proto = [0] * n_classes
            for p, l in enumerate(self.means_labels):
                label_2_proto[l] = p
            label_2_proto = torch.tensor(label_2_proto).long().to(device)

            for (x, aug), _, _ in loader:
                x, aug = x.cuda(), aug.cuda()
                _, cluster, feature = self.ce_stage(x, branch)
                out = cluster.argmax(dim=1) + target_list[0]

                feature_to_proto = dist_func(feature.cpu().detach().numpy(), prototypes.cpu().detach().numpy())
                feature_to_proto = torch.from_numpy(feature_to_proto).to(device)

                min_f_2_p, min_indices = feature_to_proto.min(dim=1)

                refuse_indices = min_f_2_p >= thres2

                feature = feature[~refuse_indices]
                cluster_accept = cluster[~refuse_indices]
                min_f_2_p_accept = min_f_2_p[~refuse_indices]
                min_indices_accept = min_indices[~refuse_indices]

                data.append(x[~refuse_indices])
                data_aug.append(aug[~refuse_indices])

                hard_label_indices = min_f_2_p_accept < thres1
                soft_label_indices = min_f_2_p_accept >= thres1

                n_hard = hard_label_indices.sum()
                n_soft = soft_label_indices.sum()


                labels_onehot = torch.zeros(n_hard + n_soft, n_proto).to(device)
                labels_onehot[hard_label_indices].scatter_(1,min_indices_accept[hard_label_indices].view(-1, 1), 1)
                labels_onehot[soft_label_indices, -self.args.num_unlabeled_classes_per_stage:] = F.softmax(cluster_accept[soft_label_indices], dim=1)

                pseudo.append(labels_onehot)

            data = torch.cat(data, dim=0).cpu()
            data_aug = torch.cat(data_aug, dim=0).cpu()
            pseudo = torch.cat(pseudo, dim=0).cpu()
        else:
            data = torch.tensor([]).cpu()
            data_aug = torch.tensor([]).cpu()
            pseudo = torch.tensor([]).cpu()

        for y, P_y in enumerate(self.exemplar_sets):
            img = torch.tensor(P_y)
            img_aug = torch.tensor(self.exemplar_sets_aug[y])
            if img.shape[0] == 0:
                continue
            label = torch.zeros(img.shape[0], n_proto).scatter(1, torch.tensor([label_2_proto[y]] * img.shape[0]).view(-1, 1), 1)
            data = torch.cat([data, img], dim=0)
            data_aug = torch.cat([data_aug, img_aug], dim=0)
            pseudo = torch.cat([pseudo, label], dim=0)

        return torch.utils.data.TensorDataset(data, data_aug, pseudo)

    @torch.no_grad()
    def mixup(self, dataset, target_list, device, args):
        n_classes = target_list[-1] + 1
        n_proto = len(self.exemplar_means)
        label_2_proto = [0] * n_classes
        for p, l in enumerate(self.means_labels):
            label_2_proto[l] = p
        label_2_proto = torch.tensor(label_2_proto).long()

        data = torch.tensor([])
        data_aug = torch.tensor([])
        pseudo = torch.tensor([])

        for y, P_y in enumerate(self.exemplar_sets):
            if y < args.num_labeled_classes:
                continue
            img = torch.tensor(P_y)
            img_aug = torch.tensor(self.exemplar_sets_aug[y])
            if img.shape[0] == 0:
                continue
            label = torch.zeros(img.shape[0], n_proto).scatter(1, torch.tensor([label_2_proto[y]] * img.shape[0]).view(-1, 1), 1)
            data = torch.cat([data, img], dim=0)
            data_aug = torch.cat([data_aug, img_aug], dim=0)
            pseudo = torch.cat([pseudo, label], dim=0)

        n_in_new_classes = data.shape[0]
        k = min(args.mixup_n, n_in_new_classes)

        random_select_1 = torch.randperm(n_in_new_classes)[:k]
        random_select_2 = torch.randperm(n_in_new_classes)[:k]

        alpha = np.random.beta(args.mixup_alpha, args.mixup_alpha, k)
        alpha = torch.from_numpy(alpha).float().view(-1, 1, 1, 1)

        mixup_data = data[random_select_1] * alpha + data[random_select_2] * (1 - alpha)
        mixup_aug = data_aug[random_select_1] * alpha + data[random_select_2] * (1 - alpha)
        mixup_pseudo = pseudo[random_select_1] * alpha.view(-1, 1) + pseudo[random_select_2] * (1 - alpha.view(-1, 1))
        mixup_pseudo = F.softmax(mixup_pseudo, dim=1)

        return torch.utils.data.ConcatDataset([dataset, torch.utils.data.TensorDataset(mixup_data, mixup_aug, mixup_pseudo)])


    @torch.no_grad()
    def sync_weights(self, ema_beta):
        for p_new, p_ema in zip(self.moco.encoder_q.parameters(), self.ema_branch.parameters()):
            p_ema.data = p_ema.data * ema_beta + p_new.data * (1 - ema_beta)
            p_ema.requires_grad = False
            p_ema.grad = None


    @torch.no_grad()
    def sync_new_branches(self):
        for p_new, p_ema in zip(self.moco.encoder_q.parameters(), self.ema_branch.parameters()):
            p_ema.data.copy_(p_new.data)
            p_ema.requires_grad = False
            p_ema.grad = None
