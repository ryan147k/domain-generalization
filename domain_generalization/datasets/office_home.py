import logging
from pathlib import Path
import random

from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import numpy as np


class OfficeHomeDataset(Dataset):
    DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']
    def __init__(self, root, domain, split, transformer):
        domain = self._check_domain(domain)

        img_path = Path(root) / domain
        self.data = ImageFolder(img_path)

        if split == 'train' or split == 'val':
            indices = list(range(len(self.data)))
            random.shuffle(indices)

            split_num = int(len(self.data) * 0.8)

            if split == 'train':
                indices = indices[:split_num]
            else:
                indices = indices[split_num:]

            self.data.samples = np.array(self.data.samples)[indices].tolist()

        self.transformer = transformer

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.transformer(img), int(label)

    def __len__(self):
        return len(self.data)

    def _check_domain(self, domain):
        if domain in self.DOMAINS:
            return domain

        if domain == 'A':
            domain = 'Art'
        elif domain == 'C':
            domain = 'Clipart'
        elif domain == 'P':
            domain = 'Product'
        elif domain == 'R':
            domain = 'Real World'
        else:
            raise ValueError(f"domain must in {self.DOMAINS}")

        return domain


def get_office_home(root,
                    batch_size,
                    source_domains,
                    target_domain,
                    split='train',
                    num_workers=8,
                    aug=True,
                    img_size=224):
    logging.info(f'get_vlcs - split:{split}, source_domains:{source_domains}, target_domain:{target_domain}, aug: {aug}')

    if split == 'val' or split == 'test':
        transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    elif split == 'train':
        if aug:
            transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.8, 1.)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                T.RandomGrayscale(p=0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
    else:
        raise AttributeError("split must be one of [train, val, test]")

    if split == 'test':
        dataset = OfficeHomeDataset(
            root=root,
            domain=target_domain,
            split=split,
            transformer=transform,
        )
    else:
        datasets = []
        for dom in source_domains:
            dataset = OfficeHomeDataset(
                root=root,
                domain=dom,
                split=split,
                transformer=transform,
            )
            datasets.append(dataset)

        dataset = ConcatDataset(datasets)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True if split == 'train' else False
    )
    return dataloader
