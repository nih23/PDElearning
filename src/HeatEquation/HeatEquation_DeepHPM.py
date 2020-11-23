import numpy as np
import time
import torch
import torch.nn as nn
import torch.autograd
import h5py
import torch.optim as optim
import scipy.io
from torch.autograd import Variable
import torch.optim as optim
from enum import Enum

from UKDDataset_segm_upd_norm import HeatEquationHPMDataset
from HeatEquation_baseline_nohvd_segm_upd_norm import  valLoss, save_checkpoint, load_checkpoint, HeatEquationBaseNet, check, model_snapshot, real_snapshot, temp_comp, mse
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F

import os
import h5py

class HeatEquationHPMNet(HeatEquationBaseNet):
    def __init__(self, numLayers, numFeatures, numLayers_hpm, numFeatures_hpm, lb, ub, samplingX, samplingY,
                 activation=torch.tanh, activation_hpm=F.relu):
        """
        This function creates the components of the Neural Network and saves the datasets
        :param x0: Position x at time zero
        :param u0: Real Part of the solution at time 0 at position x
        :param v0: Imaginary Part of the solution at time 0 at position x
        :param tb: Time Boundary
        :param X_f: Training Data for partial differential equation
        :param layers: Describes the structure of Neural Network
        :param lb: Value of the lower bound in space
        :param ub: Value of the upper bound in space
        """
        torch.manual_seed(1234)
        super().__init__(lb, ub, samplingX, samplingY, activation, noLayers=numLayers, noFeatures=numFeatures,
                         use_singlenet=True, ssim_windowSize=9, initLayers=True, useGPU=True)

        self.noLayers_hpm = numLayers_hpm
        self.noFeatures_hpm = numFeatures_hpm
        self.lin_layers_hpm = nn.ModuleList()
        self.activation_hpm = activation_hpm

        self.lb = torch.Tensor(lb).float().cuda()
        self.ub = torch.Tensor(ub).float().cuda()

        # build HPM
        self.init_layers_hpm()

    def init_layers_hpm(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """

        self.lin_layers_hpm.append(nn.Linear(7, self.noFeatures_hpm))
        for _ in range(self.noLayers_hpm):
            self.lin_layers_hpm.append(nn.Linear(self.noFeatures_hpm, self.noFeatures_hpm))
        self.lin_layers_hpm.append(nn.Linear(self.noFeatures_hpm, 1))

        for m in self.lin_layers_hpm:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward_hpm(self, x):
        for i in range(0, len(self.lin_layers_hpm) - 1):
            x = self.lin_layers_hpm[i](x)
            x = self.activation_hpm(x)
        x = self.lin_layers_hpm[-1](x)

        return x

    def net_pde(self, x, y, t, gamma=1.):
        """
        Calculates the quality of the pde estimation
        :param x postion x
        :param y postion y
        :param t time t
        """
        u, u_x, u_y, u_xx, u_yy, u_t = self.net_uv(x, y, t) #netuv for derivatives

        x = x.view(-1)
        y = y.view(-1)
        
        u_x = (u_x - (-54))/(49 - (-54))
        u_y = (u_y - (-73))/(61 - (-73))
        u_t = (u_t - (-5.5))/(9.5 - (-5.5))

        X = torch.stack([x,y,u,u_x,u_y,u_xx,u_yy], 1) #change parameters, Temperature, coordinates 
        f = u_t - self.forward_hpm(X)

        return u, f

    def hpm_loss(self, x, y, t, Ex_u):

        """
        Returns the quality HPM net
        """
        
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u, f_u = self.net_pde(x, y, t)

        Ex_u = Ex_u.view(-1)

        hpmLoss = torch.mean(f_u ** 2) 
        interpolLoss = torch.mean((u - Ex_u) ** 2)

        return hpmLoss, interpolLoss