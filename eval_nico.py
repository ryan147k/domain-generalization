import argparse
import logging
import os
from pathlib import Path

import torch

from domain_generalization.datasets.nico import get_nico
from domain_generalization.utils.utils import AverageMeter, accuracy
from domain_generalization.utils.trainer import get_model, load_model


NUM_CLASSES = 7


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=7)

    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=True, help='data augmentation')

    parser.add_argument('--model', type=str, default='res18')
    parser.add_argument('--pretrain', type=bool, default=True, help='if True, use pretrained parameters')
    parser.add_argument('--resume', type=bool, default=True, help='if True, resume the model form last stop')
    parser.add_argument('--load_path', type=str, default=True)

    opt = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def validate(val_loader, model):
    model.eval()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            bsz = labels.shape[0]

            output = model(images)

            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

    return top1.avg


def main():
    opt = parse_option()

    root = '../dataset/NICO'

    test_loader = get_nico(
        root,
        batch_size=opt.bs,
        split='test',
        aug=False
    )

    logging.info('test size: {}'.format(len(test_loader.dataset)))

    # model for PACS
    model = get_model(opt.model, opt.pretrain, 7)

    if True:
        from collections.abc import Iterable
        from torch import nn
        module = list(model.children())[-1]
        while isinstance(module, Iterable):
            module = module[-1]

        model.jigsaw_fc = nn.Linear(
            in_features=module.in_features,
            out_features=32
        )

    if opt.resume:
        load_path = Path(opt.load_path) / 'checkpoints' / 'best_valid.pth'
        state = torch.load(load_path)
        _, model, _, _ = load_model(state, model)

    model = model.cuda()

    acc = validate(test_loader, model)
    print(acc.item())


if __name__ == '__main__':
    main()
