import os
import os.path
import hashlib
import errno
from tqdm import tqdm
from PIL import Image
import numpy as np
import itertools
import torch
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F

def mixup(data, aug, target, args):
    with torch.no_grad():
        selector_for_new_classes = torch.any(target[:, args.num_labeled_classes:] > 0, dim=1)

        data_new = data[selector_for_new_classes]
        aug_new = aug[selector_for_new_classes]
        target_new = target[selector_for_new_classes]

        selector_for_current_data = torch.any(target_new[:, -args.num_unlabeled_classes_per_stage:] > 0, dim=1)
        selector_for_previous_data = ~selector_for_current_data

        n_cur = selector_for_current_data.sum().item()
        n_pre = selector_for_previous_data.sum().item()

        if args.bmix_diff_alpha and n_cur > 0 and n_pre > 0:

            data_cur = data_new[selector_for_current_data]
            aug_cur = aug_new[selector_for_current_data]
            target_cur = target_new[selector_for_current_data]

            data_pre = data_new[selector_for_previous_data]
            aug_pre = aug_new[selector_for_previous_data]
            target_pre = target_new[selector_for_previous_data]

            # if args.debug:
            #     print(n_cur, n_pre)

            if n_cur < n_pre:
                choice = np.random.choice(n_cur, n_pre)
                data_cur = data_cur[choice]
                aug_cur = aug_cur[choice]
                target_cur = target_cur[choice]
            else:
                choice = np.random.choice(n_pre, n_cur)
                data_pre = data_pre[choice]
                aug_pre = aug_pre[choice]
                target_pre = target_pre[choice]
            assert data_cur.shape[0] == data_pre.shape[0]

            c = np.random.beta(args.mixup_alpha, args.mixup_beta)
            md = c * data_cur + (1 - c) * data_pre
            ma = c * aug_cur + (1 - c) * aug_pre
            mt = c * target_cur + (1 - c) * target_pre

        else:
            bs = data_new.shape[0]
            c = np.random.beta(args.mixup_alpha, args.mixup_beta)
            perm = torch.randperm(bs).to(data.device)
            md = c * data_new + (1 - c) * data_new[perm]
            ma = c * aug_new + (1 - c) * aug_new[perm]
            mt = c * target_new + (1 - c) * target_new[perm]
        if args.pseudo_softmax:
            mt = F.normalize(mt, p=2, dim=1)
        return torch.cat((data, md), dim=0), torch.cat((aug, ma), dim=0), torch.cat((target, mt), dim=0)

class MixUpWrapper():
    def __init__(self, dataloader, args):
        self.dataloader = dataloader
        self.args = args

    def mixup_loader(self, loader):
        for data, aug, target in loader:
            yield mixup(data, aug, target, self.args)

    def __iter__(self):
        return self.mixup_loader(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class TransformKtimes:
    def __init__(self, transform, k=10):
        self.transform = transform
        self.k = k

    def __call__(self, inp):
        return torch.stack([self.transform(inp) for i in range(self.k)])

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
                )


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files
