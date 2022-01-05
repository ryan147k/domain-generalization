import argparse
import datetime
import logging
import os
import time
from pathlib import Path
from collections.abc import Iterable

import torch
import torch.nn as nn

from domain_generalization.datasets.pacs import get_pacs
from domain_generalization.datasets.jigsaw_dataset import get_jigsaw
from domain_generalization.losses.jigen_loss import JiGenLoss
from domain_generalization.utils.logging import set_logging
from domain_generalization.utils.utils import AverageMeter, accuracy, set_seed, pretty_dict, pretty_list
from domain_generalization.utils.trainer import get_model, save_model, load_model, get_optimizer


NUM_CLASSES = 7
features = None


def _hook(module, input, output):
    """hook function for getting features of input from model"""
    global features
    features = input[0]


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).split('.')[0])
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=True, help='data augmentation')

    parser.add_argument('--model', type=str, default='res18')
    parser.add_argument('--pretrain', type=bool, default=True, help='if True, use pretrained parameters')
    parser.add_argument('--resume', type=bool, default=False, help='if True, resume the model form last stop')
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--lr', type=float, default=1e-2)

    parser.add_argument('--sd', nargs='+', default=['A', 'C', 'P'], help='source domains')
    parser.add_argument('--td', type=str, default='S', help='target domain')
    parser.add_argument('--njc', type=int, default=32, help='number of jigsaw classes')
    parser.add_argument('--weight', type=float, default=0.1, help='weight of jigsaw loss')

    opt = parser.parse_args()
    opt.sd = pretty_list(opt.sd)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    return opt


def set_model(opt):
    model = get_model(
        opt.model,
        opt.pretrain,
        NUM_CLASSES,
    )

    module = list(model.children())[-1]
    while isinstance(module, Iterable):
        module = module[-1]

    module.register_forward_hook(_hook)

    model.jigsaw_fc = nn.Linear(
        in_features=module.in_features,
        out_features=opt.njc
    )

    model = model.cuda()

    criterion = JiGenLoss(opt.weight)

    optimizer = get_optimizer(
        opt.optim,
        model.parameters(),
        opt.lr,
    )

    decay_epochs = [opt.epochs * 2 // 3, opt.epochs * 4 // 5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epochs, gamma=0.1)

    return model, criterion, optimizer, scheduler


def train_epoch(train_loader, train_jigsaw_loader, model, criterion, optimizer, epoch, opt):
    jigsaw_classifier = model.jigsaw_fc

    model.train()
    avg_img_loss = AverageMeter()
    avg_jigsaw_loss = AverageMeter()
    avg_joint_loss = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
        try:
            j_images, j_labels = next(train_jigsaw_loader)
        except TypeError:
            train_jigsaw_loader = iter(train_jigsaw_loader)
            j_images, j_labels = next(train_jigsaw_loader)

        images = images.cuda()
        labels = labels.cuda()
        j_images = j_images.cuda()
        j_labels = j_labels.cuda()

        bsz = labels.shape[0]

        preds = model(images)
        _ = model(j_images) # run hook to get jigsaw features

        global features
        i_loss, j_loss, all_loss = criterion(preds,
                                             labels,
                                             features,
                                             j_labels,
                                             jigsaw_classifier)

        avg_img_loss.update(i_loss.item(), bsz)
        avg_jigsaw_loss.update(j_loss.item(), bsz)
        avg_joint_loss.update(all_loss.item(), bsz)

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

    return avg_img_loss.avg, avg_jigsaw_loss.avg, avg_joint_loss.avg


def validate(val_loader, model, jigsaw=False):
    model.eval()
    top1 = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images, labels = images.cuda(), labels.cuda()
            bsz = labels.shape[0]

            output = model(images)

            if jigsaw:
                global features
                output = model.jigsaw_fc(features)

            acc1, = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

    return top1.avg


def main():
    opt = parse_option()

    exp_name = f"{opt.exp_name}-sd_{opt.sd}-td_{opt.td}-model_{opt.model}-seed{opt.seed}"
    opt.exp_name = exp_name

    output_dir = f'exp_results/{exp_name}'
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    set_logging(exp_name, 'INFO', str(save_path))
    logging.info(str(opt))
    set_seed(opt.seed)
    logging.info(f'Set seed: {opt.seed}')

    root = '../dataset/PACS'

    train_loader = get_pacs(
        root,
        batch_size=opt.bs,
        source_domains=opt.sd,
        target_domain=None,
        split='train',
        aug=opt.aug,
        img_size=225
    )

    train_jigsaw_loader = get_jigsaw(
        train_loader.dataset,
        split='train',
        grid_size=3,
        jig_classes=32,
        batch_size=opt.bs
    )

    val_loader = get_pacs(
        root,
        batch_size=opt.bs,
        source_domains=opt.sd,
        target_domain=None,
        split='val',
        aug=False,
        img_size=225
    )

    val_jigsaw_loader = get_jigsaw(
        val_loader.dataset,
        split='val',
        grid_size=3,
        jig_classes=32,
        batch_size=opt.bs
    )

    test_loader = get_pacs(
        root,
        batch_size=opt.bs,
        source_domains=None,
        target_domain=opt.td,
        split='test',
        aug=False,
        img_size=225
    )

    test_jigsaw_loader = get_jigsaw(
        test_loader.dataset,
        split='test',
        grid_size=3,
        jig_classes=32,
        batch_size=opt.bs
    )

    logging.info('train size: {} validation size: {} test size: {} [batch size: {}]'.format(
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
        opt.bs
    ))

    model, criterion, optimizer, scheduler = set_model(opt)

    if opt.resume:
        state = torch.load(save_path / 'checkpoints'/ 'last.pth')
        opt, model, optimizer, epoch = load_model(state, model, optimizer)

    (save_path / 'checkpoints').mkdir(parents=True, exist_ok=True)

    best_accs = {'valid': 0, 'test': 0}
    best_epochs = {'valid': 0, 'test': 0}
    best_stats = {}
    start_time = time.time()

    for epoch in range(1, opt.epochs + 1):
        logging.info(f'[{epoch} / {opt.epochs}] Learning rate: {scheduler.get_last_lr()[0]}')
        i_loss, j_loss, all_loss = train_epoch(train_loader, train_jigsaw_loader, model, criterion, optimizer, epoch, opt)
        logging.info(f'[{epoch} / {opt.epochs}] Images Loss: {i_loss} Jigsaw Loss: {j_loss} Loss: {all_loss}')

        scheduler.step()

        save_file = save_path / 'checkpoints' / f'last.pth'
        save_model(model, optimizer, opt, opt.epochs, save_file)

        stats = pretty_dict(epoch=epoch)

        val_acc = validate(val_loader, model)
        val_jigsaw_acc = validate(val_jigsaw_loader, model, jigsaw=True)
        test_acc = validate(test_loader, model)
        stats['valid/acc'] = val_acc.item()
        stats['test/acc'] = test_acc.item()
        stats['valid/jigsaw_acc'] = val_jigsaw_acc.item()

        logging.info(f'[{epoch} / {opt.epochs}] {stats}')

        for tag in ['valid', 'test']:
            if stats[f'{tag}/acc'] > best_accs[tag]:
                best_accs[tag] = stats[f'{tag}/acc']
                best_epochs[tag] = epoch
                best_stats[tag] = pretty_dict(**{f'{k}': v for k, v in stats.items()})

                save_file = save_path / 'checkpoints' / f'best_{tag}.pth'
                save_model(model, optimizer, opt, epoch, save_file)

            corresponding_tag = 'test' if tag == 'valid' else 'valid'

            logging.info(
                f'\t best {tag} accuracy: {best_accs[tag]:.3f} '
                f'at epoch {best_epochs[tag]} '
                f'corr. {corresponding_tag} accuracy: {best_stats[tag][corresponding_tag + "/acc"]:.3f}'
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f'Total training time: {total_time_str}')


if __name__ == '__main__':
    main()
