import os
import time
import pprint
from glob import glob

import wandb
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import pylab as plt

import torch
import torch.nn as nn

import torchvision
import torch.nn.functional as F

from jsonargparse import ArgumentParser, ActionConfigFile

from utils import (
    get_augmentations, get_dataset, get_dataloaders, get_model,
    get_criterion, eval_metrics, get_acc, log_wandb, bert_adamw_optimizer, bert_lr_scheduler, timer
)
from utils.misc import *
from utils import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart

from utils import datasets_med as dsets, DATASETS

torch.set_warn_always(False)


def get_args():
    parser = ArgumentParser(env_prefix='', default_env=True, logger=True, print_config='--print_config')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-eval', default=512, type=int)

    parser.add_argument('--workers', default=12, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd-weight', default=10, type=float)
    parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam', 'bert_adam'])
    parser.add_argument('--optim-weight-decay', default=0, type=float)
    parser.add_argument('--scheduler', choices=['none', 'onecycle', 'bert'], default='none',
                        help='learning rate scheduler')

    parser.add_argument('--cov-reg', default=1e5, type=float)
    parser.add_argument('--dfr-reg', default=0.1, type=float)

    parser.add_argument('--dataset-name', default='HAM10000', type=str, choices=DATASETS)
    parser.add_argument('--stage', default=0, type=int)
    parser.add_argument('--num-stages', default=1, type=int)
    parser.add_argument('--subsample-target', default=None, type=none_or_str)
    parser.add_argument('--filter-perc', type=int, default=0)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('-ncbt', '--no-class-balanced-training', action='store_true')
    parser.add_argument('-dns', '--dynamic-num-samples', action='store_true')
    parser.add_argument('-sit', '--shuffle-in-training', action='store_true')
    parser.add_argument('--train-attr', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--norm-emb', type=str, default='yes', choices=['yes', 'no'])
    parser.add_argument('--data-dir', type=str, default='/home/minht/scratch/datasets')
    parser.add_argument('--feats-dir', type=str, default=None)
    parser.add_argument('--text-arch', type=str, default='bert-base-uncased')
    parser.add_argument('--subsample-type', type=str, default=None)
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--multiclass', action='store_true')
    parser.add_argument('-fts', '--fix-training-set', action='store_true')

    parser.add_argument('--model-name', default='resnet50', type=str)
    parser.add_argument('--pretrained-path', default=None, type=str)
    parser.add_argument('--pretrained-in', action='store_true')
    parser.add_argument('-fsf', '--force-saving-feats', action='store_true')

    parser.add_argument("--train-mode", type=str,
                        default='full', choices=['full', 'freeze'],
                        help='finetuning or linear probing')
    parser.add_argument("--loss-name", type=str,
                        default='ce', choices=['ce', 'isomax'],
                        help='Using ERM or IsoMax as the loss function')
    parser.add_argument("-ec", "--ensemble-criterion", type=str, choices=['wga_val', 'last'], default='wga_val')
    parser.add_argument('-es', '--entropic-scale', default=10, type=float)

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-nw', '--no-wandb', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument("--wdb-group", type=str, default=None,
                        help='Grouping factor for wandb runs')
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help='Checkpoint directory')

    args = parser.parse_args()
    return args


def eval_model(model, dataloader, criterion, device='cuda', return_feats=False, debug=False):
    from tqdm import tqdm

    model.eval()
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    feats, preds, losses = [], [], []

    with torch.no_grad():
        for _, inputs, labels, *_ in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            _feats = backbone(inputs).squeeze()
            outputs = model.fc(_feats) if hasattr(model, 'fc') else model[-1](_feats)

            loss = criterion(outputs, labels)
            _preds = F.softmax(outputs, -1)

            if return_feats:
                feats.append(_feats.detach().cpu())
            preds.append(_preds.detach().cpu())
            losses.append(loss.detach().cpu())

    if debug:
        breakpoint()

    if len(feats):
        feats = torch.concatenate(feats).numpy()
    preds = torch.concatenate(preds).numpy()
    losses = torch.concatenate(losses).numpy()
    return feats, preds, losses


@timer
def train_model(args,
                model, criterion, optimizer, scheduler, dataloaders,
                num_epochs=25, device='cuda', ckpt_path=None, prototypes_ensemble=None, stage=0,
                train_mode='full', skipped_phase: list = None, run=DummyRun(), wd_weight=10,
                worst_val_metrics=None, ensemble_bw=None, worst_metric='recall', ensemble_last=None,
                ensemble_best_test=None, cov_reg=1e4, ensemble_best_val=None, eval_freq=1,
                ):
    """

    :param args:
    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param dataloaders:
    :param num_epochs:
    :param device:
    :param ckpt_path:
    :param stage:
    :param skipped_phase:
    :param run:
    :param wd_weight:
    :param prototypes_ensemble: C x N x D ~ num classes x num prototypes x num dimensions
    :param train_mode: can be 'full' or 'freeze' to ensure batch-norm stats are not updated
    :return:
    """
    since = time.time()
    results = {}
    metric_columns = ['balanced_acc', 'accuracy', 'AUROC']

    best_acc, best_bal_acc, best_worst_metric, best_wga_test = .0, .0, .0, .0

    for epoch in range(num_epochs):
        metrics = None
        print(f'[Stage {stage}] Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if (phase not in dataloaders.keys()) or (skipped_phase is not None and phase in skipped_phase):
                continue

            if phase == 'train' and train_mode == 'full':  # so that BN stats won't be updated
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            n_instances = 0
            running_loss = 0.
            running_loss_clf = 0.
            running_loss_cov = 0.
            running_corrects = 0.
            all_preds = []

            # Iterate over data.
            for _, inputs, labels, _, feats in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                cov_loss = torch.tensor(0.0)

                with torch.set_grad_enabled(phase == 'train'):
                    if stage > 0:
                        with torch.set_grad_enabled(args.train_mode == 'full'):
                            if feats.ndim == 1:
                                feats = model[:-1](inputs)
                                if args.norm_emb == 'yes':
                                    feats = (feats - feats.mean(dim=1, keepdims=True)) / feats.std(dim=1, keepdims=True)
                            else:
                                feats = feats.to(device)

                        outputs = model[-1](feats)
                        clf_loss = criterion(outputs, labels)

                        if isinstance(criterion, IsoMaxPlusLossSecondPart):
                            head = model[-1]
                            n_classes = head.prototypes.shape[0]
                            wd = torch.einsum('ijk,ilk->ijl',
                                              [head.prototypes[:, None],
                                               head.prototypes[:, None]]) * wd_weight  # 0.5  # * 20
                            wd = wd.squeeze().mean()
                            loss = clf_loss + wd
                            if (prototypes_ensemble is not None) and (cov_reg > 0):
                                _prototypes = torch.concat([head.prototypes[:, None], prototypes_ensemble], dim=1)
                                with torch.set_grad_enabled(cov_reg > 0):
                                    n_pro, n_dim = _prototypes.shape[1:]
                                    cov = torch.einsum('ijk,ilk->ijl', [_prototypes, _prototypes]) / (n_dim - 1)
                                    cov_loss = torch.abs(cov[:, 0, 1:].sum(1).div(n_pro).mean())  # * 1e7
                                    if cov_reg:
                                        loss = loss + cov_loss * cov_reg
                        else:
                            weight = model[-1].weight if args.loss_name == 'ce' else model[-1].prototypes
                            loss = clf_loss + args.dfr_reg * torch.norm(weight, 1)
                    else:
                        outputs = model(inputs)
                        loss = clf_loss = criterion(outputs, labels)

                    if phase != 'train':
                        all_preds.append(outputs.detach().softmax(1).cpu())

                    _, preds = torch.max(outputs, 1)
                    # cov_loss = torch.tensor(0.0)
                    n_instances += len(inputs)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_clf += clf_loss.item() * inputs.size(0)
                running_loss_cov += cov_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            epoch_loss = running_loss / n_instances
            epoch_acc = running_corrects.double() / n_instances
            results[phase] = {'loss': epoch_loss, 'acc': epoch_acc}
            if stage == 0:
                run.log({f'loss/{phase}': epoch_loss}, commit=False)
                run.log({f'ovr_acc/{phase}': epoch_acc}, commit=False)

            if phase == 'train':
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print(f'train CLF Loss: {running_loss_clf / n_instances:.6f}')
                print(f'train Cov Loss: {running_loss_cov / n_instances:.6f}')
                if stage == 0:
                    run.log({'running_loss_clf': running_loss_clf / n_instances}, commit=False)
                    run.log({'running_loss_cov': running_loss_cov}, commit=False)

            # Evaluation on validation and/or test sets
            if phase != 'train':
                ds = dataloaders[phase].dataset
                all_preds = torch.concat(all_preds, dim=0).numpy()
                res = eval_metrics(all_preds, np.array(ds.y), np.array(ds._a), np.array(ds.g))

                if max(ds.y) < 10:
                    print('Per group ACC: ',
                          list(np.around([res['per_group'][i]['accuracy'] * 100 for i in res['per_group'].keys()], 2)))
                    print('Per class F1-Score: ',
                          list(np.around([res['per_class'][i]['f1-score'] * 100 for i in res['per_class'].keys()], 2)))
                if metrics is None:
                    metrics = pd.DataFrame({k: 0. for k in metric_columns}, index=[phase])
                metrics.loc[phase] = [np.round(res['overall'][k] * 100, 2) for k in metric_columns]
                log_wandb(run, phase, res)

                if phase != 'train':
                    if worst_val_metrics is None:
                        worst_val_metrics = {}
                    k = f'wga_{phase}'
                    if k not in worst_val_metrics.keys():
                        worst_val_metrics[k] = [res['min_group']['accuracy']]
                    else:
                        worst_val_metrics[k].append(res['min_group']['accuracy'])

                # Validation phase metrics
                if phase == 'val':
                    if epoch_acc > best_acc and ckpt_path:
                        best_acc = float(epoch_acc)
                        if stage == 0:
                            torch.save(model.state_dict(), f"{ckpt_path}/ckpt_best_acc.pt")
                    epoch_bal_acc = res['overall']['balanced_acc']
                    if epoch_bal_acc > best_bal_acc and ckpt_path:
                        best_bal_acc = float(epoch_bal_acc)
                        if stage == 0:
                            torch.save(model.state_dict(), f"{ckpt_path}/ckpt_best_bal_acc.pt")

                    if worst_val_metrics is None:
                        worst_val_metrics = {}
                    for k in res['per_class'][0].keys():
                        tmp = []
                        if k in ['support']:
                            continue
                        for c in res['per_class'].keys():
                            tmp.append(res['per_class'][c][k])
                        if k not in worst_val_metrics.keys():
                            worst_val_metrics[k] = [np.min(tmp)]
                        else:
                            worst_val_metrics[k].append(np.min(tmp))
                        if stage == 0:
                            run.log({f'worst_val_metrics/{k}': worst_val_metrics[k][-1]}, commit=False)

                    if worst_val_metrics[worst_metric][-1] >= best_worst_metric:
                        best_worst_metric = worst_val_metrics[worst_metric][-1]
                        if stage == 0:
                            torch.save(model.state_dict(), f"{ckpt_path}/ckpt_best_{worst_metric}.pt")
                        if ensemble_bw is not None:
                            assert isinstance(ensemble_bw, list)
                            classifier = _extract_classifier(args.loss_name, model)
                            if len(ensemble_bw) < stage:
                                ensemble_bw.append(classifier)
                            else:
                                ensemble_bw[-1] = classifier

                if phase == 'test':
                    if worst_val_metrics['wga_test'][-1] >= best_wga_test:
                        best_wga_test = worst_val_metrics['wga_test'][-1]
                        if ensemble_best_test is not None:
                            assert isinstance(ensemble_best_test, list)
                            classifier = _extract_classifier(args.loss_name, model)

                            if len(ensemble_best_test) < stage:
                                ensemble_best_test.append(classifier)
                            else:
                                ensemble_best_test[-1] = classifier

        print(metrics)
        if worst_val_metrics is not None:
            df = pd.DataFrame(worst_val_metrics)
            if len(df) > 2:
                g = sns.heatmap(df.corr(),
                                cmap='RdYlGn',
                                annot=True,
                                fmt='.3f',
                                vmin=-1, vmax=1,
                                cbar=False)
                # run.log({'plots/worst_group_acc_corr': wandb.Image(g.get_figure())},
                #         commit=False)
                plt.close()

        if stage == 0:
            run.log({'stage': stage, 'epoch': epoch})

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

    if ckpt_path is not None and (stage == 0):
        torch.save(model.state_dict(), f"{ckpt_path}/ckpt_last.pt")

    return model, results, worst_val_metrics, ensemble_bw, ensemble_best_test


def _extract_classifier(loss_name, model):
    if loss_name == 'ce':
        classifier = [model[-1].weight[:, None].detach().clone()]
    else:
        classifier = [
            model[-1].prototypes[:, None].detach().clone(),
            model[-1].distance_scale.detach().clone()
        ]
    return classifier


@timer
def evaluate_ensemble_fixed_backbone(ensemble,
                                     dataloader, model, distance_scales=None,
                                     run=DummyRun(), phase='test', return_preds=False, norm_emb='yes'):
    print(f'[{phase}]')
    ensemble = ensemble.transpose(0, 1).to('cuda')
    preds_list = torch.zeros(ensemble.shape[0], len(dataloader.dataset), dataloader.dataset.num_labels)
    model.eval()
    # classes = []
    position = 0

    ds = dataloader.dataset
    classes, attributes, groups = np.array(ds.y), np.array(ds._a), np.array(ds.g)

    with torch.no_grad():
        for _, x, y, _, feats in tqdm(dataloader):
            if feats.ndim == 1:
                feats = model[:-1](x.to('cuda'))
                if norm_emb == 'yes':
                    feats = (feats - feats.mean(dim=1, keepdims=True)) / feats.std(dim=1, keepdims=True)
            else:
                feats = feats.to('cuda')

            for i, weight in enumerate(ensemble):
                if distance_scales is not None:
                    model[-1].prototypes = torch.nn.Parameter(weight, requires_grad=False)
                    model[-1].distance_scale = nn.Parameter(distance_scales[i], requires_grad=False)
                else:
                    model[-1].weight = torch.nn.Parameter(weight, requires_grad=False)
                model.eval()
                preds_list[i][position:position + feats.shape[0]] = model[-1](feats.squeeze())
            position += feats.shape[0]
            # classes.append(y)
    # classes = torch.concat(classes).cpu().squeeze().numpy()

    for i in range(preds_list.shape[0] - 1, -1, -1):
        preds = preds_list[i].softmax(1).argmax(1).numpy()
        get_acc(preds, classes, groups)

    if preds_list[-1].ndim == 2:
        preds = preds_list.softmax(2).mean(0).detach().cpu().numpy()
        res = eval_metrics(preds, classes, attributes, groups)
        log_wandb(run, f'ensemble_{phase}_avg', res, prefix='ensemble_')

        # dist = -preds_list.max(-1)[0]
        # dist_min, dist_max = dist.min(dim=1, keepdims=True)[0], dist.max(dim=1, keepdims=True)[0]
        # dist = (dist - dist_min) / (dist_max - dist_min)
        # idx = dist.min(dim=0)[1]
        # idx = idx[None, :].T.repeat(1, 2)[None]
        # preds_min_dist = preds_list.softmax(dim=-1).gather(0, idx).numpy().squeeze()
        preds_min_dist = preds_list.max(0)[0].softmax(1).detach().cpu().numpy()
        res = eval_metrics(preds_min_dist, classes, attributes, groups)
        log_wandb(run, f'ensemble_{phase}_min_dist', res, prefix='ensemble_')

    print('Averaging Ensemble')
    get_acc(preds_list.softmax(2).mean(0).argmax(1).numpy(), classes, groups)

    print('Min Ensemble Distance')
    get_acc(preds_list.max(0)[0].softmax(1).argmax(1).numpy(), classes, groups)

    if return_preds:
        return preds_list.detach().cpu()


def extract_features(args, dataset):
    from torch.utils.data import DataLoader

    model = get_model(args.dataset_name, dataset.num_labels, args.train_mode, args.pretrained_path,
                      loss_name='ce', model=None, pretrained_in=False, model_name=args.model_name)

    model.to(args.device)

    dataloader = DataLoader(dataset=dataset, num_workers=args.workers, batch_size=args.batch_size_eval,
                            shuffle=False, pin_memory=False)

    criterion = get_criterion('ce', reduction='none')
    feats, preds, losses = eval_model(model, dataloader, criterion, return_feats=True)

    return feats


def get_pre_extracted_features(args, dataset, set_name, pre_extracted_feats=None, force_saving=False):
    if pre_extracted_feats is None:
        if args.pretrained_path is not None:
            pretrained_path = glob(args.pretrained_path)
            assert len(pretrained_path) == 1
            pre_extracted_path = f'{os.path.dirname(pretrained_path[0])}/feats_{set_name}.npy'
        else:
            assert args.feats_dir is not None
            pre_extracted_path = f'{args.feats_dir}/feats_{set_name}.npy'
            print(pre_extracted_path)

        if (os.path.exists(pre_extracted_path) == 1) and (not force_saving):
            pre_extracted_feats = np.load(pre_extracted_path, mmap_mode='r')
        else:
            pre_extracted_feats = extract_features(args, dataset=dataset)
            np.save(pre_extracted_path, pre_extracted_feats)
        if args.norm_emb == 'yes':
            pre_extracted_feats = ((pre_extracted_feats - pre_extracted_feats.mean(axis=1, keepdims=True)) /
                                   pre_extracted_feats.std(axis=1, keepdims=True))
    return pre_extracted_feats


def main(args):
    run = DummyRun()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if not args.no_wandb:
        # Log in to your W&B account
        wandb.login()
        # init wandb using config and experiment name
        name = args.dataset_name + args.suffix
        run = wandb.init(config=vars(args),
                         project='prototypical-ensemble-med',
                         group=f'{args.wdb_group}',
                         dir=args.ckpt_dir,
                         name=name,
                         resume='allow',
                         )
        wandb.define_metric('ensemble_stage')
        wandb.define_metric('ensemble_*', step_metric="ensemble_stage")

    pprint.PrettyPrinter(indent=4).pprint(args.as_dict())

    fix_random_seed(args.seed, True, True)
    prototype_ensemble = None

    pretrained_path = args.pretrained_path
    model = None
    worst_val_metrics, ensemble_bw, ensemble_best_test = None, [], []
    prototype_ensemble_last = []

    datasets = dict()
    datasets['val'] = vars(dsets)[args.dataset_name](args.data_dir, 'va', args, multiclass=args.multiclass)
    datasets['test'] = vars(dsets)[args.dataset_name](args.data_dir, 'te', args, multiclass=args.multiclass)
    pprint.PrettyPrinter(indent=4).pprint(datasets)
    worst_metric = 'wga_val'  # if args.train_attr == 'yes' else 'recall'
    train_mode = args.train_mode
    epochs = args.epochs
    subsample_type = args.subsample_type
    pre_extracted_feats, pre_extracted_feats_test = None, None

    for stage in range(args.stage, args.num_stages):
        if args.stage == 0:
            datasets['train'] = vars(dsets)[args.dataset_name](args.data_dir, 'tr', args, train_attr='no',
                                                               augmentation=True, multiclass=args.multiclass)
            print(np.unique(np.array(datasets['train'].g)[datasets['train'].idx], return_counts=True)[1])
            emb_dim = None
        # elif (args.stage == stage) or (stage == 1):
        else:
            trn_split = 'va'
            for set_name in ['val', 'test']:
                datasets[set_name].feats = get_pre_extracted_features(args, datasets[set_name], set_name,
                                                                      pre_extracted_feats,
                                                                      force_saving=args.force_saving_feats)

            if (not args.fix_training_set) or (stage == args.stage):
                datasets['train'] = vars(dsets)[args.dataset_name](
                    args.data_dir, trn_split, args, train_attr=args.train_attr, subsample_type=subsample_type,  # va
                    augmentation=False, stage=stage, pre_extracted_feats=datasets['val'].feats,
                    dynamic_num_samples=args.dynamic_num_samples, multiclass=args.multiclass)
                print(np.unique(np.array(datasets['train'].g)[datasets['train'].idx], return_counts=True)[1])
                print(args.subsample_type, len(datasets['train']))
            emb_dim = datasets['train'].feats[0].shape[-1]

        dataloaders = get_dataloaders(
            datasets, args.batch_size, args.batch_size_eval, args.workers,
            args.stage, args.no_class_balanced_training, args.shuffle_in_training)
        for k, v in dataloaders.items():
            print(f'[{k}] n steps: ', len(v))

        if args.debug:
            exit()

        model = get_model(
            args.dataset_name, datasets['train'].num_labels, train_mode,
            pretrained_path, loss_name=args.loss_name, model=model,
            pretrained_in=args.pretrained_in if stage == 0 else False,
            emb_dim=emb_dim)

        model.to(args.device)

        criterion = get_criterion(args.loss_name, entropic_scale=args.entropic_scale)

        match args.optim:
            case 'sgd':
                optimizer_ft = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                               weight_decay=args.optim_weight_decay)
            case 'adam':
                optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                                 weight_decay=args.optim_weight_decay)
            case 'bert_adam':
                optimizer_ft = bert_adamw_optimizer(model, lr=args.lr,
                                                    momentum=0.9,
                                                    weight_decay=args.optim_weight_decay)
                # optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.lr,
                #                                  weight_decay=args.optim_weight_decay)

        exp_lr_scheduler = get_scheduler_func(
            args.scheduler, args.lr, args.epochs, len(dataloaders['train']))(optimizer_ft)

        # print(prototypes_ensemble)
        if args.ensemble_criterion == 'wga_val':
            ensemble_bw = ensemble_bw if stage > 0 else None
        else:
            ensemble_bw = None

        model, _, worst_val_metrics, ensemble_bw, ensemble_best_test = train_model(
            args, model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
            num_epochs=epochs, device=args.device, ckpt_path=args.ckpt_dir,
            prototypes_ensemble=prototype_ensemble, stage=stage, train_mode=args.train_mode,
            run=run, wd_weight=args.wd_weight, worst_val_metrics=worst_val_metrics,
            ensemble_bw=ensemble_bw,
            ensemble_best_test=None,  # ensemble_best_test if stage > 0 else None,
            worst_metric=worst_metric, cov_reg=args.cov_reg)

        if args.train_mode == 'freeze':  # Fix the current backbone
            pretrained_path = None

        if (stage > 0) and (args.ensemble_criterion == 'last'):
            classifier = _extract_classifier(args.loss_name, model)
            prototype_ensemble_last.append(classifier)

        ensemble_dicts = {
            worst_metric: ensemble_bw,
            # 'best_test': ensemble_best_test,
            'last': prototype_ensemble_last,
        }
        for i, (k, ensemble) in enumerate(ensemble_dicts.items()):
            if (ensemble is not None) and (len(ensemble) > 0):
                assert isinstance(ensemble, list)

                dist_scales = [_[1] for _ in ensemble] if args.loss_name == 'isomax' else None
                ens = torch.concat([_[0] for _ in ensemble], dim=1).detach()

                print(f'Evaluating ensemble {k}')
                evaluate_ensemble_fixed_backbone(
                    ens, dataloaders['val'], model, distance_scales=dist_scales, run=run, phase=f'val_{k}',
                    norm_emb=args.norm_emb)
                evaluate_ensemble_fixed_backbone(
                    ens, dataloaders['test'], model, distance_scales=dist_scales, run=run, phase=f'test_{k}',
                    norm_emb=args.norm_emb)
                torch.save(ens, f"{args.ckpt_dir}/prototype_ensemble_{k}.pt")
                torch.save(dist_scales, f"{args.ckpt_dir}/dist_scales_{k}.pt")

                if k == args.ensemble_criterion:
                    prototype_ensemble = ens.clone()

        run.log({'ensemble_stage': stage})

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main(get_args())
