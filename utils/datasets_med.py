import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from transformers import BertTokenizer, AutoTokenizer, DistilBertTokenizer, GPT2Tokenizer
from torchvision import datasets

from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Synthetic dataset
    "CMNIST",
    # Current subpop datasets
    "Papila",
    "HAM10000",
    "ISIC2018",
    "BK",
    "Camelyon17"
]

IMAGE_DATASETS = ['Papila', 'HAM10000', 'ISIC2018', 'BK', 'Camelyon17']


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset not found: {dataset_name}")
    return globals()[dataset_name]


class SubpopDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    INPUT_SHAPE = None  # Subclasses should override
    SPLITS = {  # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']  # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, train_attr='yes', subsample_type=None, duplicates=None,
                 augmentation=False, stage=None, dynamic_num_samples=False, pre_extracted_feats=None,
                 transform_eval=None, multiclass=False):
        df = pd.read_csv(metadata)

        for split_idx in df.split.unique():
            df.loc[df.split == split_idx, 'split_indices'] = np.arange(len(df[df.split == split_idx]), dtype=int)

        self.feats = None
        class_target = 'yy' if multiclass else 'y'
        if subsample_type is not None:
            assert split in ['tr', 'va']
            df['subgroup'] = df.y.astype(str) + df.a.astype(str)
            df = subsample(
                df, target=class_target if train_attr == 'no' else 'subgroup',
                target_split=0 if split == 'tr' else 1,
                dynamic_num_samples=dynamic_num_samples,
                stage=stage)

        if split == 'te':
            match type(self).__name__:
                case 'Living17':
                    split = 'zs'
                case 'ImagenetBG':
                    print('here')
                    split = 'mixed_rand'

        df = df[df["split"] == (self.SPLITS[split])]
        self.metadata = df

        if pre_extracted_feats is not None:
            self.feats = pre_extracted_feats[df.split_indices.values.astype('int')]

        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df[class_target].tolist()
        # self.yy = df["yy"].tolist()  # multi-class ground truth
        self.a = df["a"].tolist() if train_attr == 'yes' else [0] * len(df["a"].tolist())
        self._a = df["a"].tolist()
        tmp_df = pd.DataFrame({'classes': np.array(self.y).astype('U'),
                               'attributes': np.array(self._a).astype('U')})
        self.g = (tmp_df.classes + tmp_df.attributes).values.tolist() if (df['g'] == -1).all() else df['g'].tolist()
        self.c = self.a
        self.transform_ = transform

        self._count_groups()

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

    @staticmethod
    def label_attr_to_concept(y, a):
        y, a = np.array(y), np.array(a)
        c = np.zeros((len(y), len(np.unique(y)) * len(np.unique(a))))
        c[np.arange(len(y)), y] = 1
        c[np.arange(len(a)), a + len(np.unique(y))] = 1
        return c

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        feat = torch.tensor(0, dtype=torch.long)
        if self.feats is not None:
            feat = torch.tensor(self.feats[i], dtype=torch.float)
            x = torch.tensor(0, dtype=torch.long)
        else:
            x = self.transform(self.x[i])
        return i, x, y, a, feat

    def __len__(self):
        return len(self.idx)


def subsample(df: pd.DataFrame,
              target='y',
              target_split: int = 0,
              verbose=False,
              sort_idx=False,
              filter_mask=None,
              num_samples=None,
              dynamic_num_samples=False,
              stage=None) -> pd.DataFrame:
    """

    :param stage:
    :param filter_mask:
    :param sort_idx: sort the output df by their index
    :param verbose: print counts per target-group
    :param target: sub-sampling column, 'y' for class for 'subgroup' for subgroup
    :param df: metadata created by subpop repo
    :param target_split: 0: train, 1: val, 2: test
    :param num_samples: a fixed number of samples in each class
    :return: sub-sampled metadata
    """
    if target is None:
        return df

    if 'train' in set(df.split):  # special case with ImagenetBG
        splits = {0: 'train', 1: 'val', 2: 'test'}
        target_split = splits[target_split]

    all_num_samples = (df.groupby('split').get_group(target_split)[target].value_counts())
    if num_samples is None:
        num_samples = all_num_samples.min()

    indices = df[df.split != target_split].index.tolist()
    target_indices = []
    for i, subgroup in enumerate(all_num_samples.index):

        idx = (df.groupby('split')
               .get_group(target_split)
               .groupby(target)
               .get_group(subgroup)
               .index.tolist())

        if (target == 'subgroup') and dynamic_num_samples:  # equal number of samples in each subgroup, could be different among classes
            df_split = df.groupby('split').get_group(target_split)
            cls = df_split.loc[df_split.subgroup == subgroup].iloc[0].y
            num_samples = df_split.groupby('y').get_group(cls)[target].value_counts().min()

        if filter_mask is not None:
            idx_by_subgroup = (df[df.split == target_split].reset_index()
                               .groupby(target)
                               .get_group(subgroup)
                               .index.tolist())
            sampling_weights = filter_mask[idx_by_subgroup]
            idx = np.array(idx)[np.argsort(sampling_weights)[::-1]]  # [::-1]
        tmp = list(np.random.permutation(idx)[:num_samples])
        target_indices.append(tmp)

    target_indices = np.concatenate(target_indices).reshape((len(target_indices), -1)).T.flatten().tolist()
    indices.extend(target_indices)
    if sort_idx:
        indices = sorted(indices)

    assert len(np.unique(indices)) == len(indices)
    df = df.iloc[indices]

    if verbose:
        print(df[df.split == target_split][target].value_counts())

    return df


def get_transform_eval(data_type='images', resize_size=(256, 256), input_size=(224, 224)):
    match data_type:
        case 'images':
            transform = v2.Compose([
                v2.Resize(resize_size, antialias=True),
                v2.CenterCrop(input_size),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        case _:
            raise NotImplementedError
    return transform


class HAM10000(SubpopDataset):

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 augmentation=False, stage=None, transform_eval=None, **kwargs):
        root = os.path.join(data_path, "isic_2018", "ISIC2018_Task3_Training_Input")
        # metadata = os.path.join(data_path, "isic_2018", "metadata", "HAM10000_metadata_subpop_rs0.csv")
        metadata = os.path.join(data_path, "isic_2018", "metadata", "HAM10000_metadata_subpop.csv")
        self.data_type = "images"
        if augmentation:
            transform = transforms.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # v2.RandomPhotometricDistort(p=0.5),
                # v2.RandomRotation((-15, 15)),
                # v2.Resize((256, 256)),
                # v2.CenterCrop((224, 224)),
                v2.RandomResizedCrop(size=(224, 224)),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = get_transform_eval(self.data_type) if transform_eval is None else transform_eval(self.data_type)
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates, stage=stage,
                         **kwargs)
        # self.c = self.label_attr_to_concept(self.y, self._a)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class ISIC2018(SubpopDataset):

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 augmentation=False, stage=None, transform_eval=None, **kwargs):
        target_folder = {'va': 'Validation', 'tr': 'Training', 'te': 'Test'}[split]
        root = os.path.join(data_path, "isic_2018", f"ISIC2018_Task3_{target_folder}_Input")
        metadata = os.path.join(data_path, "isic_2018", "metadata", "isic2018_metadata_subpop.csv")
        self.data_type = "images"
        if augmentation:
            transform = transforms.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation((-15, 15)),
                # v2.Resize((256, 256)),
                # v2.CenterCrop((224, 224)),
                v2.RandomResizedCrop(size=(224, 224)),
                v2.RandomPhotometricDistort(p=0.5),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = get_transform_eval(self.data_type) if transform_eval is None else transform_eval(self.data_type)
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates, stage=stage,
                         **kwargs)
        # self.c = self.label_attr_to_concept(self.y, self._a)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class Papila(SubpopDataset):

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 augmentation=False, stage=None, transform_eval=None, **kwargs):
        root = os.path.join(data_path, "PapilaDB-PAPILA", "FundusImages")
        metadata = os.path.join(data_path, "PapilaDB-PAPILA", "ClinicalData", "papila_metadata_subpop.csv")
        self.data_type = "images"
        if augmentation:
            transform = transforms.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation((-15, 15)),
                # v2.Resize((256, 256)),
                # v2.CenterCrop((224, 224)),
                v2.RandomResizedCrop(size=(224, 224)),
                v2.RandomPhotometricDistort(p=0.5),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = get_transform_eval(self.data_type) if transform_eval is None else transform_eval(self.data_type)
        super().__init__(root, split, metadata, transform, train_attr, subsample_type, duplicates, stage=stage,
                         **kwargs)
        # self.c = self.label_attr_to_concept(self.y, self._a)

    def transform(self, x):
        return self.transform_(Image.open(x).convert("RGB"))


class BK(SubpopDataset):

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 augmentation=False, stage=None, transform_eval=None, metadata_path=None, **kwargs):

        root = data_path
        # metadata = os.path.join(data_path, "isic_2018", "metadata", "HAM10000_metadata_subpop_rs0.csv")
        if metadata_path is None:
            metadata_path = os.path.join(data_path, "bk_metadata_subpop.csv")
        self.data_type = "images"
        self.force_return_images = False
        if augmentation:
            transform = transforms.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                # v2.RandomPhotometricDistort(p=0.5),
                # v2.RandomRotation((-15, 15)),
                # v2.Resize((256, 256)),
                # v2.CenterCrop((224, 224)),
                v2.RandomResizedCrop(size=(224, 224)),
                v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = get_transform_eval(self.data_type) if transform_eval is None else transform_eval(self.data_type)
        super().__init__(root, split, metadata_path, transform, train_attr, subsample_type, duplicates, stage=stage,
                         **kwargs)
        split_map = {'tr': 'train', 'te': 'test', 'va': 'val'}
        rf_a_path = f'{data_path}/{split_map[split]}/rf_a.npy'
        self.rf_a = None if not os.path.exists(rf_a_path) else np.load(rf_a_path, mmap_mode='r')

    def transform(self, x):
        return self.transform_(np.load(x, mmap_mode='r'))

    def __getitem__(self, i):
        if self.force_return_images:
           return i, self.rf_a[i].squeeze()

        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        feat = torch.tensor(0, dtype=torch.long)
        if self.feats is not None:
            feat = torch.tensor(self.feats[i], dtype=torch.float)
            x = torch.tensor(0, dtype=torch.long)
        else:
            x = self.transform(self.x[i])
        return i, x, y, a, feat