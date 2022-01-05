import logging
from pathlib import Path

from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class NICODataset(Dataset):
    def __init__(self, root, class_, label, transformer):

        img_path = Path(root) / class_
        self.data = ImageFolder(img_path)
        self.label = label

        self.transformer = transformer

    def __getitem__(self, index):
        img, _ = self.data[index]
        return self.transformer(img), int(self.label)

    def __len__(self):
        return len(self.data)


def get_nico(root,
             class_,
             label,
             batch_size,
             split='test',
             num_workers=8,
             aug=True,
             img_size=225):
    logging.info(f'get_pacs - split:{split}, aug: {aug}')

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

    dataset = NICODataset(
        root=root,
        class_=class_,
        label=label,
        transformer=transform,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False
    )
    return dataloader

