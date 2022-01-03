import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def get_model(model_name, pretrain, num_classes):
    if model_name == 'res18':
        model = models.resnet18(pretrained=pretrain)
        model.fc = nn.Linear(512, num_classes)
    elif model_name == 'res50':
        model = models.resnet50(pretrained=pretrain)
        model.fc = nn.Linear(2048, num_classes)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrain)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    else:
        raise NotImplementedError

    return model


def save_model(model, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def load_model(state_dict, model, optimizer):
    opt = state_dict['opt']
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    epoch = state_dict['epoch']

    return opt, model, optimizer, epoch


def get_optimizer(name, parameters, lr):
    if name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=1e-4)
    elif name == 'sgd':
        optimizer = optim.SGD(parameters, weight_decay=.0005, momentum=.9, lr=lr)
    else:
        raise ValueError()

    return optimizer
