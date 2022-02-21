import enum
from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .moco import ModelMoCo
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from torch.utils.data import DataLoader, dataloader
import copy
import numpy as np

class Baseline(nn.Module):
    def __init__(self, args):
        super(Baseline, self).__init__()

        self.args = args

        if self.args.feature_extractor == 'resnet18':
            self.features = models.resnet18()

        n_features = self.features.fc.in_features
        self.features.fc = nn.Identity()

        self.rotation_head = nn.Linear(n_features, 4)
        self.classification_head = nn.Linear(n_features, args.num_labeled_classes)
        self.cluster_head = nn.Linear(n_features, args.num_unlabeled_classes_per_stage)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        rotation_logits = self.rotation_head(x)
        classification_logits = self.classification_head(x)
        cluster_logits = self.cluster_head(x)
        return rotation_logits, classification_logits, cluster_logits, x

    def fix_first_three_layers(self):
        for name, param in self.features.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False

    def unfix_first_three_layers(self):
        for name, param in self.features.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = True

    def increment_classes(self, n_new_classes, device):
        save_weight = self.classification_head.weight.data.clone()
        save_bias = self.classification_head.bias.data.clone()

        feature_dim_in = self.classification_head.in_features
        feature_dim_out = self.classification_head.out_features

        self.classification_head = nn.Linear(feature_dim_in, feature_dim_out + n_new_classes).to(device)
        self.classification_head.weight.data[:feature_dim_out] = save_weight
        self.classification_head.bias.data[:] = torch.min(save_bias) - 1
        self.classification_head.bias.data[:feature_dim_out] = save_bias

        nn.init.xavier_uniform_(self.cluster_head.weight)
        nn.init.constant_(self.cluster_head.bias, 0)


class Baseline_v2(nn.Module):
    def __init__(self, args):
        super(Baseline_v2, self).__init__()

        self.args = args
        self.moco = ModelMoCo(dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, bn_splits=args.bn_splits)

        self.feature_dim = args.moco_dim

        self.classification_head = nn.Linear(self.feature_dim, args.num_labeled_classes)
        self.cluster_head = nn.Linear(self.feature_dim, args.num_unlabeled_classes_per_stage)

    def forward(self, x):
        pass

    def ssl_stage(self, x1, x2):
        loss = self.moco(x1, x2)
        return loss

    def ce_stage(self, x):
        feature = self.moco.encoder_q(x)
        classification_logits = self.classification_head(feature)
        cluster_logits = self.cluster_head(feature)
        return classification_logits, cluster_logits, feature

    def fix_feature(self):
        self.moco.encoder_q.eval()
        if self.args.unfix_last_layer:
            list(self.moco.encoder_q.net.children())[6].train()
            list(self.moco.encoder_q.net.children())[9].train()

    def fix_encoder_layers(self):
        self.moco.encoder_q.eval()
        if self.args.ft_ssl_level < 2:
            list(self.moco.encoder_q.net.children())[3].train()
        if self.args.ft_ssl_level < 3:
            list(self.moco.encoder_q.net.children())[4].train()
        if self.args.ft_ssl_level < 4:
            list(self.moco.encoder_q.net.children())[5].train()
        if self.args.ft_ssl_level < 5:
            list(self.moco.encoder_q.net.children())[6].train()
            list(self.moco.encoder_q.net.children())[9].train()

    def unfix_feature(self):
        self.moco.encoder_q.train()

    def increment_classes(self, n_new_classes, device):
        save_weight = self.classification_head.weight.data.clone()
        save_bias = self.classification_head.bias.data.clone()

        feature_dim_in = self.classification_head.in_features
        feature_dim_out = self.classification_head.out_features

        self.classification_head = nn.Linear(feature_dim_in, feature_dim_out + n_new_classes).to(device)
        self.classification_head.weight.data[:feature_dim_out] = save_weight
        self.classification_head.bias.data[:] = torch.min(save_bias) - 1
        self.classification_head.bias.data[:feature_dim_out] = save_bias

        nn.init.xavier_uniform_(self.cluster_head.weight)
        nn.init.constant_(self.cluster_head.bias, 0)

    def estimate_fisher(self, dataloader, device, stage, supervised=False):
        est_fister_info = {}
        for n, p in self.moco.encoder_q.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fister_info[n] = p.detach().clone().zero_()

        mode = self.training
        self.eval()

        for index, (x, y, idx) in enumerate(dataloader):
            x = x.to(device)
            output = self.classification_head(self.moco.encoder_q(x))
            if supervised:
                label = y.to(device)
            else:
                label = output.max(1)[1]

            negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            self.zero_grad()
            negloglikelihood.backward()

            for n, p in self.moco.encoder_q.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    est_fister_info[n] += p.grad.detach() ** 2

        est_fister_info = {n : p / index for n, p in est_fister_info.items()}

        for n, p in self.moco.encoder_q.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_EWC_prev_stage{}'.format(n, '' if self.args.ewc_online else stage), p.detach().clone())
                if self.args.ewc_online and stage == 1:
                    existing_values = getattr(self, '{}_EWC_estimated_fisher'.format(n))
                    est_fister_info[n] += self.args.ewc_gamma * existing_values
                self.register_buffer('{}_EWC_estimated_fisher{}'.format(n, '' if self.args.ewc_online else stage), est_fister_info[n])

        self.train(mode=mode)

    def ewc_loss(self, device, stage):
        try:
            losses = []
            for s in range(stage):
                for n, p in self.moco.encoder_q.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self, '{}_EWC_prev_stage{}'.format(n, '' if self.args.ewc_online else s))
                        fisher = getattr(self, '{}_EWC_estimated_fisher{}'.format(n, '' if self.args.ewc_online else s))
                        fisher = self.args.ewc_gamma * fisher if self.args.ewc_online else fisher
                        losses.append((fisher * (p - mean) ** 2).sum())
            return  (1./2) * sum(losses)
        except AttributeError:
            return torch.tensor(0., device=device)


class Baseline_v3(nn.Module):
    def __init__(self, args):
        super(Baseline_v3, self).__init__()

        self.args = args
        self.moco = ModelMoCo(dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, bn_splits=args.bn_splits)

        self.feature_dim = args.moco_dim

        self.cluster_head = nn.Linear(self.feature_dim, args.num_unlabeled_classes_per_stage, bias=False)
        self.bn = nn.BatchNorm1d(self.feature_dim, momentum=0.01)
        self.fc = nn.Linear(self.feature_dim, args.num_labeled_classes, bias=False)

        self.exemplar_sets = []
        self.exemplar_sets_aug = []

        self.compute_means = True
        self.exemplar_means = []

        self.n_classes = args.num_labeled_classes

    def forward(self, x):
        x = self.moco.encoder_q(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

    def ssl_stage(self, x1, x2):
        loss = self.moco(x1, x2)
        return loss

    def ce_stage(self, x):
        feature = self.moco.encoder_q(x)
        out = self.bn(feature)
        out = F.relu(out)
        classification = self.fc(out)
        cluster = self.cluster_head(out)
        return classification, cluster, feature

    def increment_classes(self, n, device):
        self.cluster_head = nn.Linear(self.feature_dim, self.args.num_unlabeled_classes_per_stage, bias=False).to(device)

        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False).to(device)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, x):

        mode = self.training
        self.eval()

        if self.compute_means:
            exemplar_means = []
            for l, P_y in enumerate(self.exemplar_sets):
                exemplars = []
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                if len(exemplars) == 0:
                    continue
                exemplars = torch.stack(exemplars).to(x.device)
                with torch.no_grad():
                    features = self.moco.encoder_q(exemplars)
                features = F.normalize(features, dim=1, p=2)
                mu_y = features.mean(0, keepdim=True)
                mu_y = F.normalize(mu_y, dim=1, p=2)
                exemplar_means.append(mu_y.squeeze())

            self.exemplar_means = exemplar_means
            if not self.args.update_means:
                self.compute_means = False

        if self.args.cls_type == 'lp':
            all_exempalrs = []
            one_hot_labels = []
            for l, P_y in enumerate(self.exemplar_sets):
                exemplars = []
                for ex in P_y:
                    exemplars.append(torch.from_numpy(ex))
                    if len(exemplars) == 0:
                        continue
                    exemplars = torch.stack(exemplars).to(x.device)
                    with torch.no_grad():
                        features = self.moco.encoder_q(exemplars)
                    features = F.normalize(features, dim=1, p=2)

                    n = P_y.shape[0]
                    all_exempalrs.append(features)
                    #if l < self.args.num_labeled_classes:
                    label = torch.zeros(n, len(self.exemplar_sets)).scatter(1, torch.tensor([l] * n).view(-1, 1), 1).to(x.device)
                    one_hot_labels.append(label)

            all_exempalrs = torch.cat(all_exempalrs, dim=0)
            one_hot_labels = torch.cat(one_hot_labels, dim=0)

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)        # (n_classes, feature_dim)
        # means = torch.stack([means] * batch_size)  # ()
        # means = means.transpose(1, 2)

        with torch.no_grad():
            feature = self.moco.encoder_q(x)
            feature = F.normalize(feature, dim=1, p=2) # (batch_size, feature_dim)
        # feature = feature.unsqueeze(2)
        # feature = feature.expand_as(means)

        if self.args.dist == 'euclidean':
            dist_func = euclidean_distances
        elif self.args.dist == 'cosine':
            dist_func = cosine_distances

        if self.args.cls_type == 'proto':
            dists = dist_func(feature.cpu().numpy(), means.cpu().numpy())   # (batch_size, n_classes)
            preds = dists.argmin(1)

        elif self.args.cls_type == 'lp':
            all = torch.cat([all_exempalrs, feature], dim=0)
            N = all.shape[0]
            dists = dist_func(all.cpu().numpy(), all.cpu().numpy())
            dists = torch.tensor(dists, device=x.device)
            W = torch.exp(-dists / 2)

            _, indices = torch.topk(W, self.args.lp_k)
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).float()
            W = W * mask

            D = W.sum(0)
            D_sqrt_inv = torch.sqrt(1.0 / (D + 1e-8))
            D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
            D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
            S = D1 * W * D2

            ys = one_hot_labels
            yu = torch.zeros(feature.shape[0], len(self.exemplar_sets)).to(x.device)
            y = torch.cat([ys, yu], dim=0)
            Flp = torch.matmul(torch.inverse(torch.eye(N).to(x.device) - self.args.lp_alpha * S + 1e-8), y)
            preds = Flp[all_exempalrs.shape[0]:, :].argmax(1)

        self.train(mode=mode)

        return preds

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
        for y, P_y in enumerate(self.exemplar_sets_aug):
            self.exemplar_sets_aug[y] = P_y[:m]

    def construct_exemplar_set(self, dataset, dataset_aug, n):
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

    def pseudo_labeling_and_combine_with_exemplars(self, dataset, target_list):
        mode = self.training
        self.eval()

        if dataset is not None:
            loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

            data = []
            data_aug = []
            pseudo = []

            with torch.no_grad():
                for (x, aug), _, _ in loader:
                    x, aug = x.cuda(), aug.cuda()
                    _, cluster, _ = self.ce_stage(x)
                    out = cluster.argmax(dim=1) + target_list[0]
                    data.append(x)
                    data_aug.append(aug)
                    pseudo.append(out.detach().clone())
            data = torch.cat(data, dim=0).cpu()
            data_aug = torch.cat(data_aug, dim=0).cpu()
            pseudo = torch.cat(pseudo, dim=0).cpu().long()
        else:
            data = torch.tensor([])
            data_aug = torch.tensor([])
            pseudo = torch.tensor([]).long()

        for y, P_y in enumerate(self.exemplar_sets):
            img = torch.tensor(P_y)
            label = [y] * len(P_y)
            label = torch.tensor(label)
            label = label.long()
            data = torch.cat([data, img], dim=0)
            pseudo = torch.cat([pseudo, label], dim=0)

        for y, P_y in enumerate(self.exemplar_sets_aug):
            img = torch.tensor(P_y)
            data_aug = torch.cat([data_aug, img], dim=0)

        self.train(mode=mode)

        return torch.utils.data.TensorDataset(data, data_aug, pseudo)

    def construct_exemplar_set_for_new_classes(self, dataset, target_list, m):
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
                _, cluster, _ = self.ce_stage(x)
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

            self.construct_exemplar_set(data_selected, data_aug_selected, m)

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


    def fix_encoder_layers(self):
        self.moco.encoder_q.eval()
        if self.args.ft_level < 2:
            list(self.moco.encoder_q.net.children())[3].train()
        if self.args.ft_level < 3:
            list(self.moco.encoder_q.net.children())[4].train()
        if self.args.ft_level < 4:
            list(self.moco.encoder_q.net.children())[5].train()
        if self.args.ft_level < 5:
            list(self.moco.encoder_q.net.children())[6].train()
            list(self.moco.encoder_q.net.children())[9].train()

    def fix_cluster(self):
        self.moco.encoder_q.eval()
        if self.args.fix_cluster_level < 2:
            list(self.moco.encoder_q.net.children())[3].train()
        if self.args.fix_cluster_level < 3:
            list(self.moco.encoder_q.net.children())[4].train()
        if self.args.fix_cluster_level < 4:
            list(self.moco.encoder_q.net.children())[5].train()
        if self.args.fix_cluster_level < 5:
            list(self.moco.encoder_q.net.children())[6].train()
            list(self.moco.encoder_q.net.children())[9].train()

    def fix_feature(self):
        self.moco.encoder_q.eval()

    def unfix_feature(self):
        self.moco.encoder_q.train()


class Baseline_v4(nn.Module):
    def __init__(self, args):
        super(Baseline_v4, self).__init__()

        self.args = args
        self.moco = ModelMoCo(dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t, bn_splits=args.bn_splits, multi_branch=True)
        self.feature_dim = args.moco_dim

        self.cluster_head = nn.ModuleList([nn.Linear(self.feature_dim, args.num_unlabeled_classes_per_stage, bias=False)\
            for _ in range(4)])
        # self.cluster_head = nn.Linear(self.feature_dim, args.num_unlabeled_classes_per_stage, bias=False)
        self.bn = nn.BatchNorm1d(self.feature_dim, momentum=0.01)
        self.fc = nn.Linear(self.feature_dim, args.num_labeled_classes, bias=False)

        self.exemplar_sets = []
        self.exemplar_sets_aug = []

        self.compute_means = True
        self.exemplar_means = []

        self.n_classes = args.num_labeled_classes

    def forward(self, x, stage):
        x = self.moco.encoder_q(x, stage)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc(x)
        return x

    def ssl_stage(self, x1, x2):
        loss = self.moco(x1, x2)
        return loss

    def ce_stage(self, x, stage):
        feature = self.moco.encoder_q(x, stage)
        out = self.bn(feature)
        out = F.relu(out)
        classification = self.fc(out)
        cluster = self.cluster_head[stage](out)
        return classification, cluster, feature

    def increment_classes(self, n, device):
        #self.cluster_head = nn.Linear(self.feature_dim, self.args.num_unlabeled_classes_per_stage, bias=False).to(device)

        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False).to(device)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

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
                        feature = self.moco.encoder_q(image, stage).cpu()
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


    def pseudo_labeling_and_combine_with_exemplars(self, dataset, target_list, stage):
        mode = self.training
        self.eval()

        if dataset is not None:
            loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

            data = []
            data_aug = []
            pseudo = []

            with torch.no_grad():
                for (x, aug), _, _ in loader:
                    x, aug = x.cuda(), aug.cuda()
                    _, cluster, _ = self.ce_stage(x, stage)
                    out = cluster.argmax(dim=1) + target_list[0]
                    data.append(x)
                    data_aug.append(aug)
                    pseudo.append(out.detach().clone())
            data = torch.cat(data, dim=0).cpu()
            data_aug = torch.cat(data_aug, dim=0).cpu()
            pseudo = torch.cat(pseudo, dim=0).cpu().long()
        else:
            data = torch.tensor([])
            data_aug = torch.tensor([])
            pseudo = torch.tensor([]).long()

        for y, P_y in enumerate(self.exemplar_sets):
            img = torch.tensor(P_y)
            label = [y] * len(P_y)
            label = torch.tensor(label)
            label = label.long()
            data = torch.cat([data, img], dim=0)
            pseudo = torch.cat([pseudo, label], dim=0)

        for y, P_y in enumerate(self.exemplar_sets_aug):
            img = torch.tensor(P_y)
            data_aug = torch.cat([data_aug, img], dim=0)

        self.train(mode=mode)

        return torch.utils.data.TensorDataset(data, data_aug, pseudo)


    @torch.no_grad()
    def calculate_exemplar_means(self, device, stage):
        #if self.compute_means:
        exemplar_means = []
        for l, P_y in enumerate(self.exemplar_sets):
            if l < len(self.exemplar_means):
                continue
            exemplars = []
            for ex in P_y:
                exemplars.append(torch.from_numpy(ex))
            if len(exemplars) == 0:
                continue
            exemplars = torch.stack(exemplars).to(device)
            with torch.no_grad():
                features = self.moco.encoder_q(exemplars, stage)
            features = F.normalize(features, dim=1, p=2)
            mu_y = features.mean(0, keepdim=True)
            mu_y = F.normalize(mu_y, dim=1, p=2)
            exemplar_means.append(mu_y.squeeze())

        self.exemplar_means += exemplar_means
            #self.compute_means = False

        return self.exemplar_means


    def classify(self, x, stage):
        mode = self.training
        self.eval()

        with torch.no_grad():
            self.calculate_exemplar_means(x.device, stage)

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)        # (n_classes, feature_dim)

        with torch.no_grad():
            feature = self.moco.encoder_q(x, 0)
            if self.args.norm_before_add:
                feature = F.normalize(feature, dim=1, p=2) # (batch_size, feature_dim)
            for s in range(1, stage):
                branch_feature = self.moco.encoder_q(x, s)
                if self.args.norm_before_add:
                    branch_feature = F.normalize(branch_feature, dim=1, p=2)
                feature = feature * self.args.ema_beta + branch_feature * (1 - self.args.ema_beta)
            feature = F.normalize(feature, dim=1, p=2) # (batch_size, feature_dim)

        if self.args.dist == 'euclidean':
            dist_func = euclidean_distances
        elif self.args.dist == 'cosine':
            dist_func = cosine_distances

        dists = dist_func(feature.cpu().numpy(), means.cpu().numpy())   # (batch_size, n_classes)
        dists = torch.from_numpy(dists).to(x.device)
        value, preds = dists.min(1)

        n_proto = means.shape[0]
        proto_dist = dist_func(means.cpu().numpy(), means.cpu().numpy())
        avg_proto_dist = proto_dist.sum() / (n_proto * (n_proto - 1))

        thres2 = avg_proto_dist * self.args.thres2_ratio

        refuse_index = value >= thres2

        preds_with_refuse = None
        if stage > 0:
            preds_with_refuse = dists[~refuse_index].argmin(1)

        self.train(mode=mode)

        return preds, preds_with_refuse, refuse_index

    def fix_backbone(self):
        self.moco.encoder_q.eval()
        self.moco.encoder_q.net.layer4.train()
        self.moco.encoder_q.net.fc.train()

    @torch.no_grad()
    def sync_weights(self):
        encoder_layers =self.moco.encoder_q.net
        old_weights_for_res4 = {
            name: param.data.clone() for name, param in encoder_layers.layer4[0].named_parameters()
        }
        old_weights_for_linear = {
            name: param.data.clone() for name, param in encoder_layers.fc[0].named_parameters()
        }
        for branch in range(1, 4):
            for name, param in encoder_layers.layer4[branch].named_parameters():
                param.data = old_weights_for_res4[name]
            for name, param in encoder_layers.fc[branch].named_parameters():
                param.data = old_weights_for_linear[name]
