from __future__ import print_function

import os
import math
import time
import random
import functools
from datetime import datetime

import cv2
import torch
import matplotlib
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from collections import Counter

import torch.optim as optim
from scipy.signal import hilbert
from sklearn.metrics import roc_curve, auc

# from utils.novograd import NovoGrad
# from lion_pytorch import Lion


def get_optim(optim, lr, momentum=0.9, weight_decay=0):
    match optim.lower():
        case 'sgd':
            _get_optim = lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
        case 'adam':
            # _get_optim = lambda p: torch.optim.RAdam(p, lr=lr, weight_decay=weight_decay)
            _get_optim = lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay)
        # case 'novograd':
        #     _get_optim = lambda p: NovoGrad(p, lr=lr, grad_averaging=True, weight_decay=weight_decay)
        # case 'lion':
        #     _get_optim = lambda p: Lion(p, lr=lr, weight_decay=weight_decay)
        case _:
            raise ValueError(f'Unknown optim {optim}')

    return _get_optim


class DummyRun:
    def __init__(self):
        ...

    def __call__(self, *args, **kwargs):
        ...

    def log(self, *args, **kwargs):
        ...

    def finish(self, *args, **kwargs):
        ...


def scatterplot(inv_c, outputs_c, scores=None, declare_thr=0.35, c_ind=None):
    # Plotting predicted involvement versus true involvement
    involvement = np.array(inv_c)
    predicted_label = np.array(outputs_c)
    if c_ind is None:
        c_ind = np.zeros_like(involvement)
    c_ind = np.array(c_ind)[involvement > 0]
    if scores is None:
        corr = np.corrcoef(involvement[involvement > 0], predicted_label[involvement > 0])[0, 1]
        mae = np.abs(np.subtract(involvement[involvement > 0], predicted_label[involvement > 0])).sum()
    else:
        corr, mae = scores["corr"], scores["mae"]
    fig = plt.figure(figsize=(5, 5))
    ax = sns.scatterplot(x=involvement[involvement > 0], y=predicted_label[involvement > 0], legend=False,
                         hue=c_ind)
    sns.swarmplot(x=involvement[involvement == 0], y=predicted_label[involvement == 0], size=2,
                  legend=False, ax=ax)
    diag = np.arange(0, 1, .05)
    sns.lineplot(x=diag, y=diag, color='r', ax=ax)
    ax.axvspan(-.1, 0.1, -.1, declare_thr + 0.02, alpha=.2, facecolor='lightgreen')
    ax.axvspan(-.1, 0.1, declare_thr + 0.02, 1., alpha=.2, facecolor='red')
    ax.axvspan(0.11, 1.1, -.1, declare_thr + 0.02, alpha=.2, facecolor='grey')
    ax.axvspan(0.11, 1.1, declare_thr + 0.02, 1., alpha=.2, facecolor='moccasin')
    ax.axvline(x=.101, linewidth=.6, linestyle='--', color='black')
    ax.axhline(y=declare_thr, linewidth=.6, linestyle='--', color='black')
    ax.axis('square')
    ax.set(ylim=[-.1, 1.1], xlim=[-.1, 1.1])
    ax.set(title=f'Correlation Coefficient = {corr:.3f} | MAE = {mae:.1f}',
           xlabel='True Involvement', ylabel='Predicted Involvement')
    return fig


def auroc_plot(inv_c, outputs_c):
    # Plotting AUROC curves and Sensitivity at different Specificity
    inv_c = np.array(inv_c)
    outputs_c = np.array(outputs_c)
    fpr, tpr, threshold = roc_curve(inv_c > 0, outputs_c)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=(5, 5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(0.8, tpr[(1 - fpr) >= 0.2].max(), marker='x', linestyle='--', markersize=15)
    plt.plot(0.6, tpr[(1 - fpr) >= 0.4].max(), marker='x', linestyle='--', markersize=15)
    plt.plot(0.4, tpr[(1 - fpr) >= 0.6].max(), marker='x', linestyle='--', markersize=15)
    sens = [tpr[(1 - fpr) >= 0.2].max(), tpr[(1 - fpr) >= 0.4].max(), tpr[(1 - fpr) >= 0.6].max()]
    return fig, sens


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


class BreakPoint(object):
    def __init__(self, debug=False):
        self.debug = debug

    def __call__(self):
        if self.debug:
            breakpoint()


class To2D:
    def __init__(self, in_channels=3, rescale=True):
        self.in_channels = in_channels
        self.rescale = rescale

    @staticmethod
    def reshape_to2d_image(patches):  # H W T
        m, n, t = patches.shape
        k = int(np.sqrt(t))
        image = np.concatenate(np.concatenate(patches.transpose(2, 0, 1).reshape(k, k, m, n), axis=1), axis=1)
        return image

    @staticmethod
    def rescale2d(img):
        # max_val = np.max(img)
        # min_val = np.min(img)
        # return (img - min_val) / float(max_val - min_val)
        return (img - float(img.mean())) / img.std()

    def __call__(self, core):
        img = self.reshape_to2d_image(core)
        if self.in_channels == 3:
            if self.rescale:
                # img = (self.rescale2d(img).astype('float32') * 255).astype('uint8')
                img = self.rescale2d(img.astype('float32'))
                # print(img.shape)
                # img = (self.normalize(img) * 255).astype('uint8')
                # img = (self.rescale2d(img.astype('float32')) * 255)
            img = img[:, :, np.newaxis]
            # img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img)  # , mode='RGB')
        else:
            assert self.in_channels == 1
            if self.rescale:
                # img = (img - img.mean()) / img.std()
                img = self.rescale2d(img.astype('float32'))
            img = Image.fromarray(img)
        return img


def get_scheduler_func(scheduler, lr, epochs, steps_per_epoch=None, pct_start=0.01):
    if scheduler == 'none':
        assert steps_per_epoch is not None

    if scheduler != 'none':
        if scheduler == 'triangle':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(
                opt, 0, lr,
                step_size_up=(steps_per_epoch * epochs) // 2,
                mode='triangular', cycle_momentum=False)
        elif scheduler == 'cyclic':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(
                opt, 0, lr,
                step_size_up=(steps_per_epoch * epochs) // 2,
                mode='triangular', cycle_momentum=False)
        elif scheduler == 'cosine':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch, 1e-4)
        elif scheduler == 'multistep':
            n_iters = steps_per_epoch * epochs
            milestones = [0.25 * n_iters, 0.5 * n_iters,
                          0.75 * n_iters]  # hard-coded steps for now, suitable for resnet18
            get_scheduler = lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.3)
        elif scheduler == 'onecycle':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.OneCycleLR(
                opt, lr, epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start, anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=100000,  # 2.0,
                final_div_factor=100000,  # 10000.0,
                three_phase=False,
                last_epoch=-1)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {scheduler}.")
    else:
        get_scheduler = lambda opt: None

    return get_scheduler


def make_analytical(x: np.ndarray):
    return np.abs(hilbert(x)) ** 0.3


def scale_01(x: np.ndarray):
    return (x - x.min()) / (x.max() - x.min())


def plot_raw_rf(rf, contours: list = None, contours_c: list = ('r', 'b'), frm_idx=None, i=0, fig_size=(6, 6),
                target_shape=None, patch_masks=None, patch_idx=None, ax=None, convert_whole_rf=False):
    if convert_whole_rf:
        rf_frame = make_analytical(rf)[..., frm_idx] if frm_idx is not None else rf
        rf = rf[..., frm_idx] if frm_idx is not None else rf
    else:
        rf = rf[..., frm_idx] if frm_idx is not None else rf
        rf_frame = make_analytical(rf)

    line_widths = 1
    if target_shape is not None:
        rf_frame = cv2.resize(rf_frame, dsize=target_shape)
        line_widths = 0.2
    rf_frame = scale_01(rf_frame)

    if ax is None:
        plt.figure(i, figsize=fig_size, frameon=False)
        ax = plt.gca()

    rf_frame = np.flipud(rf_frame)
    if patch_masks is not None:
        patch_mask = patch_masks[patch_idx]
        com = np.argwhere(patch_mask == 1)
        rmin, rmax = min(com[:, 0]), max(com[:, 0])
        cmin, cmax = min(com[:, 1]), max(com[:, 1])
        selected_patch = rf[rmin:rmax, cmin:cmax]
        ref_shape = target_shape if target_shape is not None else rf_frame.shape
        patch_to_img_ratio = 3 if ref_shape[0] <= ref_shape[1] else 1.1
        upscale_ratio = (min(ref_shape) / patch_to_img_ratio) / min(selected_patch.shape)
        selected_patch = scale_01(cv2.resize(selected_patch,
                                             (0, 0),
                                             fx=upscale_ratio, fy=upscale_ratio
                                             ))
        start_r, start_c = 5, 5  # 5, rf_frame.shape[1] - selected_patch.shape[1] - 5

        h, w = selected_patch.shape
        rf_frame[start_r:start_r + h, start_c:start_c + w] = np.flipud(selected_patch)

        bounding_box = np.zeros(ref_shape)
        bounding_box[start_r:start_r + h, start_c:start_c + w] = 1
        plt.contour(bounding_box, colors='magenta', linewidths=line_widths * 2)

    ax.imshow(rf_frame, cmap='gray', vmin=rf_frame.min(), vmax=rf_frame.max())

    if contours is not None:
        for contour, contour_c in zip(contours, contours_c):
            if target_shape is not None:
                contour = cv2.resize(contour.astype('uint8'), dsize=target_shape, interpolation=cv2.INTER_NEAREST)
            plt.contour(np.flipud(contour), colors=contour_c)

    target_patch = None
    if patch_masks is not None:
        for i, patch_mask in enumerate(patch_masks):
            if target_shape is not None:
                patch_mask = cv2.resize(patch_mask.astype('uint8'), dsize=target_shape, interpolation=cv2.INTER_NEAREST)
            if patch_idx is not None and (i == patch_idx):
                target_patch = patch_mask
            plt.contour(np.flipud(patch_mask), colors='y', linewidths=line_widths)
        if target_patch is not None:
            plt.contour(np.flipud(target_patch), colors='magenta', linewidths=line_widths * 2)

    plt.axis('off')
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    return ax


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fix_random_seed(seed, benchmark=False, deterministic=True):
    """Ensure reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def create_timestamped_folder(base_path='models'):
    # Get the current date and time
    now = datetime.now()
    # Format the folder name
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Create the full path
    full_path = os.path.join(base_path, folder_name)
    # Create the directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    return full_path, folder_name


def save_args_to_file(args, folder_path):
    args_path = os.path.join(folder_path, 'arguments.txt')
    with open(args_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Arguments saved to {args_path}")


def count_params(model):
    requires_grad_count = 0
    frozen_count = 0

    for param in model.parameters():
        if param.requires_grad:
            requires_grad_count += param.numel()
        else:
            frozen_count += param.numel()
    print(f'requires_grad_count = {requires_grad_count},  frozen_count = {frozen_count}')
    # return requires_grad_count, frozen_count


def make_weights_for_balanced_classes(target):
    class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    try:
        return torch.tensor([weight[t.item()] for t in target])
    except IndexError as E:
        return torch.zeros_like(target)


def get_balanced_batch_sampler(target, *args, **kwargs):
    weights = make_weights_for_balanced_classes(torch.tensor(target))
    batch_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return batch_sampler


def accumulate_dict(accumulator, single_dict, idx=None):
    if single_dict:
        if not accumulator:
            accumulator = {}
            for k in single_dict.keys():
                accumulator[k] = []
        for k, v in single_dict.items():
            if hasattr(v, '__len__'):
                if idx is not None:
                    v = idx[v]
                accumulator[k].extend(list(v))
            else:
                accumulator[k].append(v)
        return accumulator
    return None


def save_fig(images, predicted_masks, idx=1):
    plt.imsave(f'/h/minht/projects/teusformer/logs/img.png', images["pixel_values"][idx][0].squeeze(),
               cmap='gray')
    plt.imsave(f'/h/minht/projects/teusformer/logs/gt.png', images["ground_truth_mask"][idx].squeeze(),
               cmap='jet', vmin=0, vmax=1)
    plt.imsave(f'/h/minht/projects/teusformer/logs/pred.png',
               predicted_masks[idx].squeeze().cpu().detach().numpy(),
               cmap='jet', vmin=0, vmax=1)


def get_pre_extracted_features(norm_emb, pre_extracted_path):
    if '.pt' in pre_extracted_path:
        pre_extracted_feats = torch.load(pre_extracted_path, weights_only=False, map_location='cpu').numpy()
    else:
        pre_extracted_feats = np.load(pre_extracted_path, mmap_mode='r')
    if norm_emb == 'yes':
        pre_extracted_feats = ((pre_extracted_feats - pre_extracted_feats.mean(axis=1, keepdims=True)) /
                               pre_extracted_feats.std(axis=1, keepdims=True))
    return pre_extracted_feats


def describe_dataset_splits(datasets):
    """
    Describe the distribution of 'y' (class), 'g' (subgroup), and '_a' (attribute) for each dataset split.

    Args:
        datasets (dict): Dictionary with keys as split names ('train', 'val', 'test', etc.)
                         and values as dataset objects with attributes 'y', 'g', '_a' (tensors or lists).

    Returns:
        pd.DataFrame: Multi-indexed DataFrame with counts for each category across splits.
    """
    stats = dict()

    for split, dataset in datasets.items():
        split_stats = {}
        for field in ['y', 'g', '_a']:
            values = dataset.__getattribute__(field)
            counts = Counter(values if isinstance(values, list) else values.tolist())
            for k, v in counts.items():
                split_stats[f'{field}={k}'] = v
        stats[split] = split_stats

    df = pd.DataFrame(stats).fillna(0).astype(int)
    return df.sort_index()
