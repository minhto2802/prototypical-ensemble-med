import os

os.environ['WANDB_INIT_TIMEOUT'] = '600'  # 10 minutes

import argparse
from functools import partial
from dataclasses import dataclass
from typing import Type, Tuple, Dict, Any

import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import datasets_med as dsets
from utils.datasets import get_dataloaders
from utils.eval_helpers import eval_metrics
from utils.models import PrototypeTransformer
from utils.utils import init_model, train_ensemble, get_train_loader
from utils.misc import DummyRun, fix_random_seed, describe_dataset_splits

GLOBAL_DEFAULTS = {
    "run_name": "default_run",
    "log_dir": "./logs",
    "db": (bool, False),
    "slurm_id": os.environ['SLURM_JOB_ID'],
    "suffix": "",
    "wdb_group": 'null',
    "seed": 42,
    "data_dir": '',
    "metadata_path": '',
    "multiclass": (bool, None),
    "ckpt_dir": f"/checkpoint/minht/{os.environ['SLURM_JOB_ID']}",
    "norm_emb": (bool, None),
}


@dataclass
class Args:
    data_dir: str = "embeddings/tabpfn"
    metadata_path: str = "embeddings/tabpfn/metadata.csv"
    pretrained_path: str = None
    norm_emb: bool = True
    dataset_name: str = 'Features'
    device: str = 'cuda'
    workers: int = 0
    batch_size_train: int = 256
    batch_size_eval: int = 256
    train_attr: str = 'no'
    epochs: int = 10
    lr: float = 1e-3
    multiclass: bool = False
    num_stages: int = 51
    emb_dim: int = 192
    num_prototypes: int = None
    scheduler: str = 'none'
    cov_reg: float = 5e4
    entropic: int = 10
    show_freq: int = 10
    optim: str = 'sgd'
    trn_split: str = 'va'
    d_model: int = 256
    ff_dim: int = 1024
    alpha: float = 0.1

    def update_with_dict(self, extra: dict):
        for k, v in extra.items():
                if hasattr(self, k):
                    if (v is not None) and (v != ''):
                        setattr(self, k, v)


class ArgsParser:
    def __init__(self, dpe_cls: Type, transformer_cls: Type, global_args: Dict[str, Any] = None):
        self.dpe_cls = dpe_cls
        self.transformer_cls = transformer_cls
        self.global_args = global_args or {}
        self.parser = argparse.ArgumentParser()

        self._add_args_group("dpe", self.dpe_cls)
        self._add_args_group("t", self.transformer_cls)
        self._add_global_args()

    def _add_args_group(self, prefix: str, arg_class: Type):
        group = self.parser.add_argument_group(f"{prefix} arguments")
        for field in arg_class.__dataclass_fields__.values():
            arg_name = f"--{prefix}.{field.name}"
            group.add_argument(arg_name, type=field.type, default=field.default)

    def _add_global_args(self):
        group = self.parser.add_argument_group("global arguments")
        for arg_name, arg_type_default in self.global_args.items():
            if isinstance(arg_type_default, tuple):
                typ, default = arg_type_default
            else:
                typ, default = type(arg_type_default), arg_type_default
            group.add_argument(f"--{arg_name}", type=typ, default=default)

    def parse(self) -> Tuple[Any, Any, Dict[str, Any]]:
        args = self.parser.parse_args()
        args_dpe, args_t, args_global = {}, {}, {}

        for k, v in vars(args).items():
            if k.startswith("dpe."):
                args_dpe[k.split(".", 1)[1]] = v
            elif k.startswith("t."):
                args_t[k.split(".", 1)[1]] = v
            else:
                args_global[k] = v

        dpe_args = self.dpe_cls(**args_dpe)
        transformer_args = self.transformer_cls(**args_t)
        dpe_args.update_with_dict(args_global)
        transformer_args.update_with_dict(args_global)

        return dpe_args, transformer_args, args_global


def init_dataset(args, split_map=None):
    # Simulate dsets dictionary
    datasets = dict()
    split_map = split_map if split_map is not None else {'train': 'tr', 'val': 'va', 'test': 'te'}
    for split in split_map.keys():
        features = np.load(f"{args.data_dir}/feats_{split_map[split]}.npy")
        if args.norm_emb:
            features = ((features - features.mean(axis=1, keepdims=True)) / features.std(axis=1, keepdims=True))
        split_key = split_map[split]
        datasets[split] = vars(dsets)[args.dataset_name](
            root=args.data_dir,
            split=split_key,
            metadata_path=args.metadata_path,
            transform=None,
            train_attr=args.train_attr,
            pre_extracted_feats=features,
            multiclass=args.multiclass,
        )
    print(describe_dataset_splits(datasets))
    return datasets


def get_class_balanced_sampler(dataset):
    # Assumes dataset returns (index, filename, label, attr, group, features)
    targets = [int(dataset[i][2]) for i in range(len(dataset))]  # extract labels

    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # or a multiple for over-sampling
        replacement=True  # allow repeated samples
    )
    return sampler


def train(eval_splits=None, balance_sampling=True, *args, **kwargs):
    args = Args(*args, **kwargs)
    datasets = init_dataset(args)
    eval_splits = ['train_eval', 'val', 'test'] if eval_splits is None else eval_splits
    tracking_metrics = ['loss', 'acc', 'bacc', 'wga']

    loaders = dict()
    # loaders['train'] = DataLoader(datasets['train'], batch_size=args.batch_size_train, shuffle=True)
    if balance_sampling:
        sampler = get_class_balanced_sampler(datasets['train'])
        shuffle = False
    else:
        sampler, shuffle = None, True
    loaders['train'] = DataLoader(datasets['train'],
                                  batch_size=args.batch_size_train,
                                  shuffle=shuffle,
                                  sampler=sampler)

    for split in eval_splits:
        loaders[split] = DataLoader(datasets[split.split('_')[0]], batch_size=args.batch_size_eval)

    model = nn.Linear(args.emb_dim, 2).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,  # peak learning rate
            steps_per_epoch=len(loaders['train']),
            epochs=args.epochs,
            pct_start=0.3,  # % of cycle spent increasing LR
            anneal_strategy='cos',  # cosine annealing
            final_div_factor=1e4  # how low LR will go after peak
        )
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError('Not implemented scheduler')

    metrics = dict()
    all_res = dict()
    for split in eval_splits:
        all_res[split] = []
        metrics[split] = dict()
        for k in tracking_metrics:
            metrics[split][k] = []

    pbar = tqdm(range(args.epochs), desc='Vanilla Training', leave=True)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_losses = []
        for _, _, y, _, _, x in loaders['train']:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None and (args.scheduler in ['onecycle']):
                scheduler.step()
            epoch_losses.append(loss.item())

        # Validation
        postfix = dict()
        for split in eval_splits:
            metrics[split], res = evaluate(model, loaders[split], criterion, args.device, metrics[split])
            all_res[split].append(res)
            if split == 'val':
                for k in tracking_metrics:
                    postfix[k] = metrics[split][k][-1]
        postfix['train_loss'] = round(np.mean(epoch_losses), 3)
        # pbar.set_postfix(postfix)

        if scheduler is not None and (args.scheduler not in ['onecycle']):
            scheduler.step()

    return all_res


def evaluate(model, loader, criterion, device, metrics):
    model.eval()
    all_logits, all_y, all_a, all_g, epoch_losses = [], [], [], [], []
    with torch.no_grad():
        for _, _, y, a, _, x in loader:
            x = x.to(device)
            logits = model(x).cpu()
            loss = criterion(logits, y)
            all_logits.append(logits)
            all_y.append(y)
            all_a.append(a)
            epoch_losses.append(loss.item())
        metrics['loss'].append(round(np.mean(epoch_losses), 3))

    preds = torch.cat(all_logits).softmax(dim=1).numpy()
    targets = torch.cat(all_y).numpy()
    attributes = torch.cat(all_a).numpy()
    groups = np.array(loader.dataset.g)

    # Compute metrics using your eval_helpers
    res = eval_metrics(preds=preds, targets=targets, attributes=attributes, gs=groups, thres=0.5)

    metrics['acc'].append(round(res['overall']['accuracy'] * 100, 1))
    metrics['bacc'].append(round(res['overall'].get('balanced_acc', 0.0) * 100, 1))
    metrics['wga'].append(round(res['min_group']['accuracy'] * 100, 1))
    return metrics, res


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, _scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels, all_attrs, all_groups = [], [], [], []

    pbar = tqdm(loader, desc=f"[Epoch {epoch}] Training", leave=False)
    for _, _, y, a, _, feat in pbar:
        optimizer.zero_grad()
        inputs = feat.to(device)
        labels = y.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # pbar.set_postfix(loss=loss.item())
        running_loss += loss.item() * inputs.size(0)

        all_preds.append(outputs.detach().softmax(dim=1).cpu())
        all_labels.append(labels.cpu())
        all_attrs.append(a)

        y_str = y.cpu().numpy().astype(str)
        a_str = a.cpu().numpy().astype(str)
        all_groups.append(np.char.add(y_str, a_str).tolist())

        if _scheduler is not None:
            _scheduler.step()

    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    attributes = torch.cat(all_attrs, dim=0).numpy()
    groups = np.array(sum(all_groups, []))  # flatten

    avg_loss = running_loss / len(loader.dataset)
    res = eval_metrics(preds, labels, attributes, groups)

    return avg_loss, res


def evaluate_transformer(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    ds = loader.dataset
    classes, attributes, groups = np.array(ds.y), np.array(ds._a), np.array(ds.g)

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", leave=False)
        for _, _, y, a, _, feat in pbar:
            inputs = feat.to(device)
            labels = y.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            all_preds.append(outputs.softmax(dim=1).cpu())
            pbar.set_postfix(loss=loss.item())

    preds = torch.cat(all_preds, dim=0).numpy()

    avg_loss = running_loss / len(loader.dataset)
    res = eval_metrics(preds, classes, attributes, groups)

    return avg_loss, res


def train_dpe(run=None, *args, **kwargs):
    run = run if run is not None else DummyRun()
    args = Args(*args, **kwargs)
    # Convert args to dict and add a prefix

    split_map = {'val': args.trn_split, 'test': 'te'}  # 'train': 'tr'
    datasets = init_dataset(args, split_map=split_map)

    dataloaders = dict()
    for split in ['val', 'test']:
        dataloaders[split] = DataLoader(dataset=datasets[split], num_workers=0, pin_memory=False,
                                        batch_size=args.batch_size_eval, shuffle=False, drop_last=False)

    clf_head = nn.Sequential(nn.Identity(), nn.Identity())
    clf_head.emb_dim = args.emb_dim
    num_classes = len(np.unique(datasets['val'].y))

    train_ensemble_func = partial(
        train_ensemble, datasets=datasets, dataloaders=dataloaders, lr=args.lr, epochs=args.epochs,
        num_stages=args.num_stages, full_model=clf_head, show_freq=args.show_freq, entropic=args.entropic,
        cov_reg=args.cov_reg, optim=args.optim, wd_weight=10, alpha=args.alpha,
        scheduler=args.scheduler,
        init_train_loader=partial(get_train_loader, attr_availability=args.train_attr, workers=0,
                                  dataset_name=args.dataset_name, trn_split=args.trn_split,
                                  batch_size=args.batch_size_train, metadata_path=args.metadata_path, transform=None,
                                  multiclass=args.multiclass),
        init_model_func=partial(init_model, device='cuda', num_classes=num_classes),
        run=run,
    )

    metrics = dict()
    *metrics['DPE'], prototype_ensemble = train_ensemble_func(random_subset=True)

    return metrics['DPE'], prototype_ensemble


def train_transformer(_pe, run=None, *args, **kwargs):
    run = run if run is not None else DummyRun()
    args = Args(*args, **kwargs)

    # Prefix args with 'transformer/' so it's clear in wandb
    split_map = {'train': args.trn_split, 'test': 'te'}

    datasets = init_dataset(args, split_map=split_map)
    describe_dataset_splits(datasets)

    if args.num_prototypes is not None:
        _pe = _pe[:args.num_prototypes]

    pe_concat = torch.concat([_[0] for _ in _pe], dim=1).detach()
    dist_concat = torch.concat([_[1] for _ in _pe], dim=0).detach()

    model = PrototypeTransformer(
        prototypical_ensemble=pe_concat,
        dist_ensemble=dist_concat,
        num_layers=2,
        num_heads=4,
        ff_dim=args.ff_dim,
        dropout=0.1,
        d_model=args.d_model,
        train_mode='freeze',
    ).to('cuda')

    loaders = get_dataloaders(datasets, batch_size_train=args.batch_size_train, batch_size_eval=256, workers=0,
                              stage=-1,
                              no_class_balanced_training=False, shuffle_in_training=False)
    train_loader = loaders['train']
    test_loader = loaders['test']
    criterion = nn.CrossEntropyLoss()

    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,  # peak learning rate
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.0,  # % of cycle spent increasing LR
        anneal_strategy='cos',  # cosine annealing
        final_div_factor=1e4  # how low LR will go after peak
    )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(args.epochs):
        train_loss, train_res = train_one_epoch(model, train_loader, optimizer, criterion,
                                                device=args.device, epoch=epoch, _scheduler=scheduler)
        test_loss, test_res = evaluate_transformer(model, test_loader, criterion, device=args.device)

        # Log transformer training metrics
        run.log({
            "transformer_train/loss": train_loss,
            "transformer_train/acc": train_res['overall']['accuracy'] * 100,
            "transformer_train/bacc": train_res['overall'].get('balanced_acc', 0) * 100,
            "transformer_train/wga": train_res['min_group']['accuracy'] * 100,
            "transformer_test/loss": test_loss,
            "transformer_test/acc": test_res['overall']['accuracy'] * 100,
            "transformer_test/bacc": test_res['overall'].get('balanced_acc', 0) * 100,
            "transformer_test/wga": test_res['min_group']['accuracy'] * 100,
        }, commit=False)
        run.log({"transformer_epoch": epoch + 1})

    for g in test_res['per_group']:
        print(f'{g}:', round(test_res['per_group'][g]['accuracy'] * 100, 2))

    return model


def main():
    parser = ArgsParser(Args, Args, global_args=GLOBAL_DEFAULTS)
    dpe_args, transformer_args, global_args = parser.parse()

    # Log to wandb
    if global_args['db']:
        run = DummyRun()
        dpe_args.workers, transformer_args.workers = 0, 0  # No multiprocessing in debug mode
    else:
        run = wandb.init(
            project="dpe-med-tab",
            name=f"Run_seed{global_args['seed']}{global_args['suffix']}",
            group=global_args['wdb_group'],
            config={
                **{f"dpe/{k}": v for k, v in dpe_args.__dict__.items()},
                **{f"transformer/{k}": v for k, v in transformer_args.__dict__.items()},
                **{f"global/{k}": v for k, v in global_args.items()},
            },
            dir=global_args['ckpt_dir'],
        )
        wandb.define_metric("transformer_epoch")
        wandb.define_metric("transformer_*", step_metric="transformer_epoch")

    fix_random_seed(global_args['seed'])

    # Run training under the same run
    res, pe = train_dpe(run=run, **dpe_args.__dict__)
    _ = train_transformer(pe, run=run, **transformer_args.__dict__)
    wandb.finish()


if __name__ == '__main__':
    main()
