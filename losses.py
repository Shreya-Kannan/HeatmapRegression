"""
Some code is from repos https://github.com/voxelmorph/voxelmorph and https://github.com/xueh2/CMR_LandMark_Detection.
Check them out!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KLD(nn.Module):
    def __init__(self, reduction='batchmean', debug_mode=False):
        super(KLD, self).__init__()
                    
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction)
        self.debug_mode = debug_mode
        
    def forward(self, outputs, targets):     
        log_prob = torch.log_softmax(outputs, dim=1)
        targets = torch.clamp(targets, min=1e-7, max=1.0 - 1e-7)
        loss = self.kldiv_loss(log_prob, targets)

        if(self.debug_mode):
            print('KLD loss is ', loss)
            
        return loss

class Dice(nn.Module):
    def __init__(self, debug_mode=False):
        super(Dice, self).__init__()
        self.debug_mode = debug_mode
        
    def forward(self, outputs, targets):
        outputs = torch.softmax(outputs, dim=1)
        ndims = len(list(targets.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (targets * outputs).sum(dim=vol_axes)
        bottom = torch.clamp((targets + outputs).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        if self.debug_mode:
            print('Dice loss is: ', -dice)
        return -dice

class MSE(nn.Module):
    def __init__(self, debug_mode=False):
        super(MSE, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.debug_mode = debug_mode
        
    def forward(self, outputs, targets):
        outputs = torch.softmax(outputs, dim=1)
        loss = self.mse(outputs, targets)
        if self.debug_mode:
            print('MSE loss is:',loss)
        return loss


class Criterion(nn.Module):
    def __init__(self, kld_weight=0, dice_weight=0, mse_weight=1.0, reduction='batchmean', debug_mode=False):
        super(Criterion, self).__init__()

        self.kld_weight = kld_weight
        self.dice_weight = dice_weight
        self.mse_weight = mse_weight

        self.kld = KLD(reduction=reduction, debug_mode=debug_mode)
        self.dice = Dice(debug_mode=debug_mode)
        self.mse = MSE(debug_mode=debug_mode)

        if dice_weight==0 and mse_weight==0 and kld_weight==0:
            raise ValueError('All loss weights are 0. Please initialize a non-zero weight to a loss function.')

        print(f'Initializing loss as: {mse_weight}*MSE+ {dice_weight}* Dice + {kld_weight}* KLD')

    def forward(self, outputs, targets):
        """
        targets: [B,C,H,W], 
        outputs: [B,C,H,W], logits
        """

        total_loss = 0

        if self.kld_weight>0:
            total_loss += self.kld_weight * self.kld(outputs, targets)
        
        if self.dice_weight>0:
            total_loss += self.dice_weight * self.dice(outputs, targets)
        
        if self.mse_weight>0:
            total_loss += self.mse_weight * self.mse(outputs, targets)

        return total_loss
        