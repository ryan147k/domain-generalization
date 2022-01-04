import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F


class JiGenLoss(nn.Module):
    """Jigsaw Loss
    """
    def __init__(self, weight):
        super(JiGenLoss, self).__init__()
        self.weight = weight

    def forward(self, preds, labels, j_features, j_labels, j_classifier):
        j_preds = j_classifier(j_features)

        image_loss = F.cross_entropy(preds, labels)
        jigsaw_loss = F.cross_entropy(j_preds, j_labels)

        loss = image_loss + self.weight * jigsaw_loss

        return image_loss, jigsaw_loss, loss
