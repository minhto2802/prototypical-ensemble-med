import os
import regex as re
from glob import glob
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import find_classes

__all__ = [
    'WaterBirds',
    'CelebA',
    'get_dataset',
]


class SubpopDataset:
    def __init__(self,
                 transform=None,
                 dataset_dir='/scratch/ssd004/scratch/minht/datasets',
                 dataset_folder: str = None,
                 metadata_file: str = None,
                 subsample_target=None,
                 subsample_split=0,
                 split_mask: list = None,
                 filter_mask: list = None,
                 num_samples: int = None,
                 *args, **kwargs,
                 ):
        """

        :param transform:
        :param dataset_dir:
        :param dataset_folder:
        :param metadata_file:
        :param num_samples: a fixed number of samples in each class
        :param subsample_target: either 'subgroup' or 'y' to get balanced subgroups or classes
        :param subsample_split: target split for subsampling [0, 1, 2], default [0]: training set
        :param args:
        :param kwargs:
        """
        self.num_samples = num_samples
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.subsample_target = subsample_target
        self.subsample_split = subsample_split
        self.dataset_folder = f'{self.dataset_dir}/{dataset_folder}'
        self.metadata_file = f'{self.dataset_dir}/{metadata_file}'
        self.metadata = self.load_metadata(split_mask, filter_mask)
        self.groups_to_attributes = None

    @staticmethod
    def insert_subgroup(df):
        df['subgroup'] = df.y.astype(str) + df.a.astype(str)
        return df

    def get_files_splits(self, set_split_dict: dict):
        files = {}
        for set_name, split_idx in set_split_dict.items():
            files[set_name] = (
                    self.dataset_folder + '/'
                    + self.metadata.filename[self.metadata.split == split_idx]).to_list()
        return files

    def get_dataset_splits(self, set_split_dict: dict):

        def filter_files(_set_name, _hash_split):
            def is_valid(filename):
                try:
                    return _hash_split[filename] == _set_name
                except KeyError:
                    return False

            return is_valid

        datasets = {}
        files_splits = self.get_files_splits(set_split_dict)

        # hash_split = {}
        # for k in files_splits.keys():
        #     hash_split.update({f: k for f in files_splits[k]})

        set_names = []
        file_names = []
        for k in files_splits.keys():
            set_names.extend([k] * len(files_splits[k]))
            file_names.extend(files_splits[k])
        hash_split = dict(zip(file_names, set_names))

        metadata = self.metadata
        group_unique = np.unique(metadata.subgroup)
        group_dict = dict(zip(group_unique, range(len(group_unique))))
        name_group_dict = dict(zip(metadata.filename.apply(os.path.basename),
                                   metadata.subgroup.map(group_dict)))
        name_class_dict = dict(zip(metadata.filename.apply(os.path.basename),
                                   metadata.y))

        for set_name, split_idx in set_split_dict.items():
            transform = None
            if self.transform is not None:
                transform = self.transform['train'] if set_name == 'train' else self.transform['eval']

            folders_in_use = (metadata[metadata.split == split_idx]
                              .filename.apply(os.path.dirname)
                              .unique().tolist())

            if set_name == 'train':
                breakpoint()
            ds = CustomImageFolder(self.dataset_folder,
                                   transform=transform,
                                   is_valid_file=filter_files(set_name, hash_split),
                                   folders_in_use=folders_in_use)

            ds.samples = [(_[0], name_class_dict[os.path.basename(_[0])]) for _ in ds.samples]
            ds.g = np.array([name_group_dict[os.path.basename(_[0])] for _ in ds.samples])
            ds.g2a = self.groups_to_attributes
            ds.y = np.array([_[1] for _ in ds.samples])
            ds.a = ds.g2a[ds.g]
            ds.subsample_target = self.subsample_target if split_idx == self.subsample_split else None
            datasets[set_name] = ds

        return datasets

    def gen_datasets(self, stage, set_split_dict=None):
        if set_split_dict is None:
            set_split_dict = {'train': 0, 'val': 1, 'test': 2} if stage == 0 else {'train': 1, 'test': 2}
        datasets = self.get_dataset_splits(set_split_dict)
        return datasets

    def load_metadata(self, split_mask, filter_mask):
        metadata = pd.read_csv(self.metadata_file)
        metadata = self.insert_subgroup(metadata)
        if split_mask is not None:
            if not isinstance(split_mask, list):
                split_mask = [split_mask]
            metadata = metadata[metadata.split.isin(split_mask)]

        metadata = subsample(metadata,
                             target=self.subsample_target,
                             target_split=self.subsample_split,
                             num_samples=self.num_samples,
                             filter_mask=filter_mask)
        return metadata


class WaterBirds(SubpopDataset):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(transform,
                         dataset_folder='waterbirds/waterbird_complete95_forest2water2',
                         metadata_file='waterbirds/metadata_waterbirds.csv',
                         *args, **kwargs)
        self.groups_to_attributes = np.array([0, 1, 0, 1])


class CelebA(SubpopDataset):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(transform,
                         dataset_folder='celeba',
                         metadata_file='celeba/metadata_celeba.csv',
                         *args, **kwargs)
        self.metadata.filename = 'img_align_celeba/' + self.metadata.filename
        self.groups_to_attributes = np.array([0, 1, 0, 1])


class MetaShift(SubpopDataset):
    def __init__(self, transform=None, *args, **kwargs):
        super().__init__(transform,
                         dataset_folder='metashift',
                         metadata_file='metashift/metadata_metashift.csv',
                         *args, **kwargs)
        self.metadata.filename = 'img_align_celeba/' + self.metadata.filename
        self.groups_to_attributes = np.array([0, 1, 0, 1])


DATSET_NAME_DICT = {
    'waterbirds': WaterBirds,
    'celeba': CelebA,
    'metashift': MetaShift,
}


def get_dataset(dataset_name, stage=0, transform=None, set_split_dict=None, *args, **kwargs):
    base_dataset = DATSET_NAME_DICT[dataset_name.lower()](transform, *args, **kwargs)
    assert dataset_name.lower() in DATSET_NAME_DICT
    ds = base_dataset.gen_datasets(stage, set_split_dict)
    return ds


class CustomImageFolder(ImageFolder):
    """A simple custom dataset from ImageFolder dataset that allows having classes without any images
    This was done by removing those classes from 'classes' and 'class_to_idx' fields"""

    def __init__(self, *args, folders_in_use=None, **kwargs):
        self.folders_in_use = folders_in_use
        super().__init__(*args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = self.folders_in_use
        _class_to_idx = {}
        try:
            _, class_to_idx = find_classes(directory)
            for _class in classes:
                _class_to_idx[_class] = class_to_idx[_class]
        except:
            for i, _class in enumerate(classes):
                _class_to_idx[_class] = i
        return classes, _class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # a, g = self.a[index], self.g[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # return sample, target, a, g
        return sample, target


def subsample(df: pd.DataFrame,
              target='y',
              target_split: int = 0,
              verbose=False,
              sort_idx=False,
              filter_mask=None,
              num_samples=None) -> pd.DataFrame:
    """

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

    all_num_samples = (df.groupby('split').get_group(target_split)[target].value_counts())
    if num_samples is None:
        num_samples = all_num_samples.min()

    indices = df[df.split != target_split].index.tolist()
    target_indices = []
    for subgroup in all_num_samples.index:
        idx = (df.groupby('split')
               .get_group(target_split)
               .groupby(target)
               .get_group(subgroup)
               .index.tolist())

        if filter_mask is not None:
            idx_by_subgroup = (df[df.split == target_split].reset_index()
                               .groupby(target)
                               .get_group(subgroup)
                               .index.tolist())
            sampling_weights = filter_mask[idx_by_subgroup]
            # idx = np.random.choice(idx, size=1500, replace=False, p=sampling_weights/sampling_weights.sum())
            idx = np.array(idx)[np.argsort(sampling_weights)[::-1]]  # [::-1]
        tmp = list(np.random.permutation(idx)[:num_samples])
        # tmp = list(idx[:num_samples])
        target_indices.append(tmp)
        # print(np.array(tmp)[:10])
        # indices.extend(list(idx[:num_samples]))

    # target_indices = np.concatenate(
    #     target_indices).reshape((len(target_indices), -1)).T.flatten().tolist()
    target_indices = np.concatenate(target_indices).tolist()
    indices.extend(target_indices)
    if sort_idx:
        indices = sorted(indices)
    # print(indices)

    assert len(np.unique(indices)) == len(indices)
    # indices = np.random.permutation(indices).tolist()
    df = df.iloc[indices]

    if verbose:
        print(df[df.split == target_split][target].value_counts())
    assert len(df[df.split == target_split][target].value_counts().unique()) == 1

    return df
