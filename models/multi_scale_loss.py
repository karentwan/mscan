import torch.nn as nn
import torch.nn.functional as F


class MultiScaleLoss(nn.Module):
    '''
    multi scale loss
    '''

    def __init__(self):
        super(MultiScaleLoss, self).__init__()

    def forward(self, pred, gt):
        loss = 0
        for item in pred:
            [b, c, h, w] = item.shape
            gt_i = F.interpolate(gt, size=(h, w), mode='bilinear')
            loss += F.mse_loss(item, gt_i)
        return loss
