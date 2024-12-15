import os, argparse, time
import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F


class MSELoss(torch.nn.Module):

    def __init__(self, **kwargs):
        super(MSELoss, self).__init__()
        self.l2_w_c = 1
        self.l2_w_f = 1


    def forward(self, preds, gts):
        preds = preds.view(-1)
        gts = gts.view(-1)
        l2_loss = F.mse_loss(preds, gts) 
    
        return l2_loss
    
