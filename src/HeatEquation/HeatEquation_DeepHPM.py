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
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F
import os
import h5py

class HeatEquationHPMNet(HeatEquationBaseNet):
    def __init__(self, numLayers, numFeatures, numLayers_alpha, numFeatures_alpha, numLayers_hs, numFeatures_hs, lb, ub, samplingX, samplingY, activation=torch.tanh, activation_alpha=F.leaky_relu , activation_hs=F.leaky_relu, scalingConstant = 200):
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

        self.noLayers_alpha = numLayers_alpha
        self.noFeatures_alpha = numFeatures_alpha
        self.lin_layers_alpha = nn.ModuleList()
        self.activation_alpha = activation_alpha
        
        self.noLayers_hs = numLayers_hs
        self.noFeatures_hs = numFeatures_hs
        self.lin_layers_hs = nn.ModuleList()
        self.activation_hs = activation_hs
        self.scalingConstant = scalingConstant
        
        self.lb = torch.Tensor(lb).float().cuda()
        self.ub = torch.Tensor(ub).float().cuda()
        #self.dlb = torch.Tensor(dlb).float().cuda()
        #self.dub = torch.Tensor(dub).float().cuda()

        # build HPM
        self.init_layers_alpha()
        self.init_layers_hs()

    def init_layers_alpha(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """

        self.lin_layers_alpha.append(nn.Linear(3, self.noFeatures_alpha))
        for _ in range(self.noLayers_alpha):
            self.lin_layers_alpha.append(nn.Linear(self.noFeatures_alpha, self.noFeatures_alpha))
        self.lin_layers_alpha.append(nn.Linear(self.noFeatures_alpha, 1))

        for m in self.lin_layers_alpha:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def init_layers_hs(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """

        self.lin_layers_hs.append(nn.Linear(3, self.noFeatures_hs))
        for _ in range(self.noLayers_hs):
            self.lin_layers_hs.append(nn.Linear(self.noFeatures_hs, self.noFeatures_hs))
        self.lin_layers_hs.append(nn.Linear(self.noFeatures_hs, 1))

        for m in self.lin_layers_hs:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward_alpha(self, t_in):
        t_in = (t_in - self.lb)/(self.ub - self.lb)
        for i in range(0, len(self.lin_layers_alpha) - 1):
            t_in = self.lin_layers_alpha[i](t_in)
            t_in = self.activation_alpha(t_in)
        t_in = self.lin_layers_alpha[-1](t_in)
        
        #t_in = torch.sigmoid(t_in)
        #t_in = t_in * self.scalingConstant
        
        return t_in
        
    def forward_hs(self, x):
        t_in = (x - self.lb)/(self.ub - self.lb)
        for i in range(0, len(self.lin_layers_hs) - 1):
            t_in = self.lin_layers_hs[i](t_in)
            t_in = self.activation_hs(t_in)
        t_in = self.lin_layers_hs[-1](t_in)
        #t_in = torch.tanh(t_in)
        return t_in

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
       
        X = torch.stack([x,y,t,u,u_x,u_y,u_xx,u_yy], 1) #change parameters, Temperature, coordinates 
        f = u_t - self.forward_hpm(X)

        return u, f
    """
    def hpm_loss(self, x, y, t, Ex_u):
       
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u, f_u = self.net_pde(x, y, t)

        Ex_u = Ex_u.view(-1)

        hpmLoss = torch.mean(f_u ** 2) 
        interpolLoss = torch.mean((u - Ex_u) ** 2)

        return hpmLoss, interpolLoss
    
    def hpm_loss2(self, x, y, t, u, u_x, u_y, u_xx, u_yy, u_t0):

        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        u = u.view(-1)
        u_x = u_x.view(-1)
        u_y = u_y.view(-1)
        u_xx = u_xx.view(-1)
        u_yy = u_yy.view(-1)
        
        u_t0 = u_t0.view(-1)
                     
        X = torch.stack([x,y,u,u_x,u_y,u_xx,u_yy], 1)
        u_t = self.forward_hpm(X)

        #hpmLoss = (torch.mean((u_t0 - u_t) ** 2)) 
        hpmLoss = (torch.mean(torch.abs(u_t0 - u_t))) 

        return hpmLoss
    """
    def hpm_loss(self, x, y, t, u_xx, u_yy, exact_u_t):
        
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u_xx = u_xx.view(-1)
        u_yy = u_yy.view(-1)
        
        exact_u_t = exact_u_t.view(-1)
        
        X_hs = torch.stack([x,y,t], 1)        
        hs = (self.forward_hs(X_hs)).view(-1) #heat source net 
        
        X_alpha = torch.stack([x,y,t], 1) 
        alpha = (self.forward_alpha(X_alpha)).view(-1) #alpha net
        
        pred_u_t = alpha*(u_xx+u_yy) + hs
        
        hpmLoss = (torch.mean((exact_u_t - pred_u_t) ** 2))

        return hpmLoss
