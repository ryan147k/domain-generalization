import random
import math

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RSCLoss(nn.Module):
    """Representation Self-Challenging Loss"""
    def __init__(self, drop_f, drop_b):
        super(RSCLoss, self).__init__()

        self.drop_f = drop_f
        self.drop_b = drop_b

    # def forward(self, features, preds, labels, oh_labels, classifier):
    #     B, C, H, W = features.shape
    #
    #     # Equation (1): compute gradients with respect to representation
    #     grad = autograd.grad((preds * oh_labels).sum(), features)[0]
    #
    #     # Equation (2): compute top-gradient-percentile mask
    #     if np.random.randint(0, 9) <= 4:
    #         # 1. spatial wise
    #         spatial_grad_mean = grad.mean(dim=1).view(B, H * W)
    #         percentiles = np.percentile(spatial_grad_mean.cpu(), self.drop_f * 100, axis=1)
    #         percentiles = torch.Tensor(percentiles)
    #         percentiles = percentiles.unsqueeze(1).repeat(1, H * W)
    #         mask_f = spatial_grad_mean.lt(percentiles.cuda()).float()
    #         mask_f = mask_f.view(-1, 1, H, W).repeat(1, C, 1, 1)
    #     else:
    #         # 2. channel wise
    #         channel_grad_mean = grad.mean(dim=(2, 3)).view(B, C)
    #         percentiles = np.percentile(channel_grad_mean.cpu(), self.drop_f * 100, axis=1)
    #         percentiles = torch.Tensor(percentiles)
    #         percentiles = percentiles.unsqueeze(1).repeat(1, C)
    #         mask_f = channel_grad_mean.lt(percentiles.cuda()).float()
    #         mask_f = mask_f.view(-1, C, 1, 1).repeat(1, 1, H, W)
    #
    #     # Equation (3): mute top-gradient-percentile activations
    #     f_muted = features * mask_f
    #
    #     # Equation (4): compute muted predictions
    #     p_muted = classifier(f_muted)
    #
    #     # Section 3.3: Batch Percentage
    #     soft = torch.softmax(preds, dim=1)
    #     s_muted = torch.softmax(p_muted, dim=1)
    #     changes = (soft * oh_labels).sum(1) - (s_muted * oh_labels).sum(1)
    #     percentile = np.percentile(changes.detach().cpu(), self.drop_b * 100)
    #     mask_b = changes.lt(percentile).float().view(-1, 1, 1, 1).repeat(1, C, H, W)
    #     mask = torch.logical_or(mask_f, mask_b).float()
    #
    #     # Equations (3) and (4) again, this time mutting over examples
    #     p_muted_again = classifier(features * mask)
    #
    #     loss = F.cross_entropy(p_muted_again, labels)
    #
    #     return loss

    def forward(self, features, preds, labels, oh_labels, classifier):
        percent = self.drop_b

        x_new = features
        output = preds

        class_num = output.shape[1]
        index = labels
        num_rois = x_new.shape[0]
        num_channel = x_new.shape[1]
        H = x_new.shape[2]
        HW = x_new.shape[2] * x_new.shape[3]
        sp_i = torch.ones([2, num_rois]).long()
        sp_i[0, :] = torch.arange(num_rois)
        sp_i[1, :] = index
        sp_v = torch.ones([num_rois])
        one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()

        one_hot = torch.sum(output * one_hot_sparse)
        grads_val = autograd.grad(one_hot, x_new)[0]

        grad_channel_mean = torch.mean(grads_val.view(num_rois, num_channel, -1), dim=2)
        channel_mean = grad_channel_mean
        grad_channel_mean = grad_channel_mean.view(num_rois, num_channel, 1, 1)
        spatial_mean = torch.sum(x_new * grad_channel_mean, 1)
        spatial_mean = spatial_mean.view(num_rois, HW)

        choose_one = random.randint(0, 9)
        if choose_one <= 4:
            # ---------------------------- spatial -----------------------
            spatial_drop_num = math.ceil(HW * 1 / 3.0)
            th18_mask_value = torch.sort(spatial_mean, dim=1, descending=True)[0][:, spatial_drop_num]
            th18_mask_value = th18_mask_value.view(num_rois, 1).expand(num_rois, HW)
            mask_all_cuda = torch.where(spatial_mean > th18_mask_value, torch.zeros(spatial_mean.shape).cuda(),
                                        torch.ones(spatial_mean.shape).cuda())
            mask_all = mask_all_cuda.reshape(num_rois, H, H).view(num_rois, 1, H, H)
        else:
            # -------------------------- channel ----------------------------
            vector_thresh_percent = math.ceil(num_channel * 1 / 3.2)
            vector_thresh_value = torch.sort(channel_mean, dim=1, descending=True)[0][:, vector_thresh_percent]
            vector_thresh_value = vector_thresh_value.view(num_rois, 1).expand(num_rois, num_channel)
            vector = torch.where(channel_mean > vector_thresh_value,
                                 torch.zeros(channel_mean.shape).cuda(),
                                 torch.ones(channel_mean.shape).cuda())
            mask_all = vector.view(num_rois, num_channel, 1, 1)

        # ----------------------------------- batch ----------------------------------------
        cls_prob_before = F.softmax(output, dim=1)
        x_new_view_after = x_new * mask_all
        x_new_view_after = classifier(x_new_view_after)
        cls_prob_after = F.softmax(x_new_view_after, dim=1)

        sp_i = torch.ones([2, num_rois]).long()
        sp_i[0, :] = torch.arange(num_rois)
        sp_i[1, :] = index
        sp_v = torch.ones([num_rois])
        one_hot_sparse = torch.sparse.FloatTensor(sp_i, sp_v, torch.Size([num_rois, class_num])).to_dense().cuda()
        before_vector = torch.sum(one_hot_sparse * cls_prob_before, dim=1)
        after_vector = torch.sum(one_hot_sparse * cls_prob_after, dim=1)
        change_vector = before_vector - after_vector - 0.0001
        change_vector = torch.where(change_vector > 0, change_vector, torch.zeros(change_vector.shape).cuda())
        th_fg_value = torch.sort(change_vector, dim=0, descending=True)[0][int(round(float(num_rois) * percent))]
        drop_index_fg = change_vector.gt(th_fg_value).long()
        ignore_index_fg = 1 - drop_index_fg
        not_01_ignore_index_fg = ignore_index_fg.nonzero()[:, 0]
        mask_all[not_01_ignore_index_fg.long(), :] = 1

        mask_all = Variable(mask_all, requires_grad=True)
        x = features * mask_all
        x = classifier(x)

        loss = F.cross_entropy(x, labels)

        return loss