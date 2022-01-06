import argparse
import datetime
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from domain_generalization.datasets.pacs import get_pacs
from domain_generalization.utils.logging import set_logging
from domain_generalization.utils.utils import AverageMeter, accuracy, set_seed, pretty_dict, pretty_list
from domain_generalization.utils.trainer import get_model, save_model, load_model, get_optimizer


NUM_CLASSES = 7


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=True, help='data augmentation')

    parser.add_argument('--model', type=str, default='res18')
    parser.add_argument('--pretrain', type=bool, default=True, help='if True, use pretrained parameters')
    parser.add_argument('--resume', type=bool, default=False, help='if True, resume the model form last stop')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--sd', nargs='+', default=['A', 'C', 'P'], help='source domains')
    parser.add_argument('--td', type=str, default='S', help='target domain')

    opt = parser.parse_args()
    opt.sd = pretty_list(opt.sd)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def set_model(opt):
    model = get_model(
        opt.model,
        opt.pretrain,
        NUM_CLASSES,
    ).cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(
        opt.optim,
        model.parameters(),
        opt.lr,
    )

    decay_epochs = [opt.epochs // 2, opt.epochs * 2 // 3]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    return model, criterion, optimizer, scheduler


def train_epoch(train_loader, model, criterion, optimizer):
    model.train()
    avg_loss = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        bsz = labels.shape[0]

        images_flip = torch.flip(images, (3,)).detach().clone()
        images = torch.cat((images, images_flip))
        labels = torch.cat((labels, labels))

        preds = model(images)

        loss = criterion(preds, labels)

        avg_loss.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss.avg


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

    exp_name = f"{opt.exp_name}-sd_{opt.sd}-td'{opt.td}'-model_{opt.model}-bs{opt.bs}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    logging.info(f'Set seed: {opt.seed}')
    set_seed(opt.seed)
    logging.info(f'save_path: {save_path}')

    np.set_printoptions(precision=3)
    torch.set_printoptions(precision=3)

    root = '../dataset/PACS'

    train_loader = get_pacs(
        root,
        batch_size=opt.bs,
        source_domains=opt.sd,
        target_domain=None,
        split='train',
        aug=opt.aug
    )

    val_loader = get_pacs(
        root,
        batch_size=opt.bs,
        source_domains=opt.sd,
        target_domain=None,
        split='val',
        aug=False
    )

    test_loader = get_pacs(
        root,
        batch_size=opt.bs,
        source_domains=None,
        target_domain=opt.td,
        split='test',
        aug=False
    )

    logging.info('train size: {} validation size: {} test size: {}'.format(
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset))
    )

    model, criterion, optimizer, scheduler = set_model(opt)

    if opt.resume:
        state = torch.load(save_path / 'checkpoints' / 'last.pth')
        opt, model, optimizer, epoch = load_model(state, model, optimizer)

    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {}
    start_time = time.time()

    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}')
        loss = train_epoch(train_loader, model, criterion, optimizer)
        logging.info(f'[{epoch} / {opt.epochs}] Loss: {loss}')

        scheduler.step()

        stats = pretty_dict(epoch=epoch)

        val_acc = validate(val_loader, model)
        test_acc = validate(test_loader, model)
        stats['valid/acc'] = val_acc.item()
        stats['test/acc'] = test_acc.item()


        logging.info(f'[{epoch} / {opt.epochs}] {stats}')

        for tag in ['valid', 'test']:
            if stats[f'{tag}/acc'] > best_accs[tag]:
                best_accs[tag] = stats[f'{tag}/acc']
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(**{f'{k}': v for k, v in stats.items()})

                save_file = save_path / 'checkpoints' / f'best_{tag}.pth'
                save_model(model, optimizer, opt, epoch, save_file)

            logging.info(
                f'\t best {tag} accuracy: {best_accs[tag]:.3f} at epoch {best_epochs[tag]} \t best_{tag}_stats: {best_stats[tag]}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')

    save_file = save_path / 'checkpoints' / f'last.pth'
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
