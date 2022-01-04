import logging
from pathlib import Path
from PIL import Image
import random

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms.functional import crop
import numpy as np


__PERMUTATIONS__ = []


def _get_permutations(grid_size, jig_classes):
    permutations = []

    n_grids = grid_size ** 2
    unshuffled_p = list(range(n_grids))
    permutations.append(unshuffled_p)

    for _ in range(jig_classes - 1):
        p = unshuffled_p.copy()
        random.shuffle(p)
        permutations.append(p)

    return permutations


class JigsawDataset(Dataset):
    """dataset for jigsaw"""
    def __init__(self, dataset, grid_size, permutations):
        """
        :param dataset: torch image dataset
        :param grid_size: number of pieces on one side of the puzzle
        :param jig_classes: number of pre permutations
        """
        self.dataset = dataset

        self.grid_size = grid_size
        self.permutations = permutations

    def __getitem__(self, index):
        img, label = self.dataset.__getitem__(index)
        n_grids = self.grid_size ** 2

        patches = [self._get_patch(img, i) for i in range(n_grids)]

        pre_permutation_idx = np.random.randint(len(self.permutations))  # added 1 for class 0: unsorted
        permutation = self.permutations[pre_permutation_idx]

        data = []
        for i in range(n_grids):
            data.append(patches[permutation[i]])

        data = torch.stack(data, dim=0)
        data = torchvision.utils.make_grid(data, self.grid_size, padding=0)

        return data, pre_permutation_idx

    def __len__(self):
        return len(self.dataset)

    def _get_patch(self, img, n):
        assert img.size(1) % self.grid_size == 0

        w = int(img.size(1) / self.grid_size)
        y = int(n / self.grid_size)
        x = n % self.grid_size
        patch = crop(img,
                     top=x * w,
                     left=y * w,
                     height=w,
                     width=w)
        return patch

    def _generate_permutations(self):
        permutations = []

        n_grids = self.grid_size ** 2
        unshuffled_p = list(range(n_grids))
        permutations.append(unshuffled_p)

        for _ in range(self.jig_classes - 1):
            p = unshuffled_p.copy()
            random.shuffle(p)
            permutations.append(p)

        return permutations


def get_jigsaw(dataset,
               split,
               grid_size,
               jig_classes,
               batch_size,
               num_workers=8):
    logging.info(f'get_jigsaw - split:{split}, grid_size:{grid_size}, jig_classes:{jig_classes}')

    global __PERMUTATIONS__
    if len(__PERMUTATIONS__) == 0:
        __PERMUTATIONS__ = _get_permutations(grid_size, jig_classes)

    permutations = __PERMUTATIONS__

    jigsaw_dataset = JigsawDataset(dataset,
                                   grid_size,
                                   permutations)

    dataloader = DataLoader(
        dataset=jigsaw_dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True if split == 'train' else False
    )

    return dataloader
