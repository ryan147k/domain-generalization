import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class Subset(Dataset):
    def __init__(self, dataset, limit):
        """
        :param dataset:
        :param limit: num of samples in subset
        """
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_transform(split,
                  aug,
                  img_size):

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

    return transform
