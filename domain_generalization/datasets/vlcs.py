import logging
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

from ._utils import Subset, get_transform


class VLCSDataset(Dataset):
    DOMAINS = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    def __init__(self, root, domain, split, transformer):
        domain = self._check_domain(domain)

        if split == 'val':
            split = 'test'  # use test data of source domain for validation

        img_path = Path(root) / domain / split
        self.data = ImageFolder(img_path)

        self.transformer = transformer

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.transformer(img), label

    def __len__(self):
        return len(self.data)

    def _check_domain(self, domain):
        if domain in self.DOMAINS:
            return domain

        if domain == 'C':
            domain = 'CALTECH'
        elif domain == 'L':
            domain = 'LABELME'
        elif domain == 'P':
            domain = 'PASCAL'
        elif domain == 'S':
            domain = 'SUN'
        else:
            raise ValueError(f"domain must in {self.DOMAINS}")

        return domain


def get_vlcs(root,
             batch_size,
             source_domains,
             target_domain,
             split='train',
             num_workers=8,
             aug=True,
             img_size=224,
             limit=None):

    logging.info(f'get_vlcs - split:{split}, source_domains:{source_domains}, target_domain:{target_domain}, aug: {aug}')

    transform = get_transform(
        split=split,
        aug=aug,
        img_size=img_size
    )

    if split == 'test':
        dataset = VLCSDataset(
            root=root,
            domain=target_domain,
            split=split,
            transformer=transform,
        )
    else:
        datasets = []
        for dom in source_domains:
            dataset = VLCSDataset(
                root=root,
                domain=dom,
                split=split,
                transformer=transform,
            )
            datasets.append(dataset)

        dataset = ConcatDataset(datasets)

    if limit is not None:
        dataset = Subset(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True if split == 'train' else False
    )
    return dataloader

