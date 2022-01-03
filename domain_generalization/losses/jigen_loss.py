import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F


class JiGenLoss(nn.Module):
    """Representation Self-Challenging Loss
    """
    def __init__(self):
        super(JiGenLoss, self).__init__()

    def forward(self):
        pass
