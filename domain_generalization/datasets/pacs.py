import logging
from pathlib import Path
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ._utils import Subset, get_transform


class PacsDataset(Dataset):
    DOMAINS = ['photo', 'art_painting', 'cartoon', 'sketch']
    def __init__(self, root, domain, split, transformer):
        domain = self._check_domain(domain)

        if split == 'val':
            split = 'crossval'

        self.img_path = Path(root) / 'kfold'
        train_val_path = Path(root) / 'train_val_splits' / f'{domain}_{split}_kfold.txt'

        names, labels = self._dataset_info(train_val_path)

        self.names = names
        self.labels = labels
        self.transformer = transformer

    def __getitem__(self, index):
        img_path = Path(self.img_path) / self.names[index]
        img = Image.open(img_path).convert('RGB')
        return self.transformer(img), int(self.labels[index] - 1)

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _dataset_info(train_val_path):
        with open(train_val_path, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            file_names.append(row[0])
            labels.append(int(row[1]))

        return file_names, labels

    def _check_domain(self, domain):
        if domain in self.DOMAINS:
            return domain

        if domain == 'P':
            domain = 'photo'
        elif domain == 'A':
            domain = 'art_painting'
        elif domain == 'C':
            domain = 'cartoon'
        elif domain == 'S':
            domain = 'sketch'
        else:
            raise ValueError()

        return domain


def get_pacs(root,
             batch_size,
             source_domains,
             target_domain,
             split='train',
             num_workers=8,
             aug=True,
             img_size=225,
             limit=None):

    logging.info(f'get_pacs - split:{split}, source_domains:{source_domains}, target_domain:{target_domain}, aug: {aug}')

    transform = get_transform(
        split=split,
        aug=aug,
        img_size=img_size,
    )

    if split == 'test':
        dataset = PacsDataset(
            root=root,
            domain=target_domain,
            split=split,
            transformer=transform,
        )
    else:
        datasets = []
        for dom in source_domains:
            dataset = PacsDataset(
                root=root,
                domain=dom,
                split=split,
                transformer=transform,
            )
            datasets.append(dataset)

        dataset = ConcatDataset(datasets)

    if limit is not None:
        dataset = Subset(dataset, limit)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True if split == 'train' else False
    )
    return dataloader
