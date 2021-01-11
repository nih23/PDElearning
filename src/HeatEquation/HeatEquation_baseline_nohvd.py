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
import torch.nn.functional as F
from enum import Enum
from UKDDataset import SchrodingerEquationDataset
import matplotlib.pyplot as plt
import torch.utils.data.distributed
from argparse import ArgumentParser
import os
import sys
import pathlib

def model_snapshot(model, pData, timeStep, dataset):
    seg_matrix = dataset.segmentation(pData, 0)
    
    x,y,t = dataset.getInput(timeStep, dataset.coordinateSystem, pData)
    
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()        
    x = x.view(-1)
    y = y.view(-1)
    t = t.view(-1)

    X = torch.stack([x, y, t], 1)

    U = model.forward(X).detach().cpu().numpy()
    #U = U*(dataset.coordinateSystem['u_ub'] - dataset.coordinateSystem['u_lb']) + dataset.coordinateSystem['u_lb']
    
    return (U.reshape(480,640))*seg_matrix.T

def real_snapshot(pData, timeStep, dataset):

    seg_matrix = dataset.segmentation(pData, 0)
    U = dataset.loadFrame(pData, timeStep)[0]
    return U.reshape(640,480).T*seg_matrix.T

def temp_comp(model, pData, dataset, nt, frameStep):
    temp = []
    temp_pr = []
    for i in range(0,nt,int(frameStep)):
        mT = model_snapshot(model, pData, i, dataset).reshape(-1)
        rT = real_snapshot(pData, i, dataset).reshape(-1)
        
        mT = mT[mT != 0]
        rT = rT[rT != 0]
        
        t_pr = np.mean(mT)
        t = np.mean(rT)
        
        temp.append(t)
        temp_pr.append(t_pr)
        
    return temp, temp_pr

def mse(model, pData, dataset):
    data = np.array([])
    prediction = np.array([])
    for i in range(0,1000,10):
        mT = model_snapshot(model, pData, i, dataset).reshape(-1)
        rT = real_snapshot(pData, i, dataset).reshape(-1)

        mT = mT[mT != 0]
        rT = rT[rT != 0]

        data = np.append(data,rT)
        prediction = np.append(prediction,mT)

    mse = np.square(np.subtract(data, prediction)).mean()
    
    return mse

def valLoss(model, dataset, timeStep, pData):
    
    mT = model_snapshot(model, pData, timeStep, dataset).reshape(-1)
    rT = real_snapshot(pData, timeStep, dataset).reshape(-1)

    valLoss_u = np.max(abs(mT - rT)) 
    valSqLoss_u = np.sqrt(np.sum(np.power(mT - rT,2)))

    return valLoss_u, valSqLoss_u
    
def check(model, dataset, csystem, nt, frameStep):
    
    norms_sq = []
    norms_inf = []
    for i in range(0, nt, frameStep):
    #for i in range(args.nt):
        valLoss_u, valSqLoss_u = valLoss(model, dataset, i, pData)
        norms_sq.append(valSqLoss_u)
        norms_inf.append(valLoss_u)
    
    nframes = range(args.nt)
    _norms_sq_, nframes = zip(*sorted(zip(norms_sq, nframes), reverse = True))
    
    return norms_sq, norms_inf, nframes

def save_checkpoint(model, path, epoch):
    #print(model.state_dict().keys())    
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    state = {
        'model': model.state_dict(),
    }
    torch.save(state, path + 'model_' + str(epoch)+'.pt')
    print("saving model to ---> %s" % (path + 'model_' + str(epoch)+'.pt'))

def load_checkpoint(model, path):
    device = torch.device('cpu')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])

def getDefaults():
    # static parameter
    nx = 640
    ny = 480 
    nt = 1000
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    dt = 0.001
    tmax = 1
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx":nx , "ny":ny, "nt": nt, "dt": dt}

    return coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, tmax 

class HeatEquationBaseNet(nn.Module):
    def __init__(self, lb, ub, samplingX, samplingY, activation=torch.tanh, noLayers = 8, noFeatures = 100, use_singlenet = True, ssim_windowSize = 9, initLayers = True, useGPU = True):
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
        super(HeatEquationBaseNet, self).__init__()
        self.activation = activation
        self.useGPU = useGPU
        self.lb = torch.Tensor(lb).float()
        self.ub = torch.Tensor(ub).float()
        self.noFeatures = noFeatures
        self.noLayers = noLayers
        self.in_t = nn.ModuleList([])
        if(initLayers):
            self.init_layers()

    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
        self.in_t.append(nn.Linear(3, self.noFeatures)) # time
        for _ in range(self.noLayers):
            self.in_t.append(nn.Linear(self.noFeatures, self.noFeatures))
        self.in_t.append(nn.Linear(self.noFeatures,1))
       
        for m in self.in_t:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #scale spatiotemporal coordinates to [0,1]
        t_in = (x - self.lb)/(self.ub - self.lb)
        for i in range(len(self.in_t)-1):
            x = self.in_t[i](x)
            x = self.activation(x)
        output = self.in_t[-1](x)
        #scale to [0,1]
        output = output*(35.421-27.96) + 27.96
        return output
                
    def net_uv(self, x, y, t):
        """
        Function that calculates the nn output at postion (x,y) at time t
        :param x: position
        :param t: time
        :return: Approximated solutions and their gradients
        """

        #torch.cuda.empty_cache()
        dim = x.shape[0] #defines the shape of the gradient

        # save input in variabeles is necessary for gradient calculation
        x = Variable(x, requires_grad=True).cuda()
        y = Variable(y, requires_grad=True).cuda()
        t = Variable(t, requires_grad=True).cuda()

        X = torch.stack([x, y, t], 1)

        u = (self.forward(X)).reshape(-1)
        grads = torch.ones([dim]).cuda()

        # huge change to the tensorflow implementation this function returns all neccessary gradients
        J_U = torch.autograd.grad(u,[x,y,t], create_graph=True, grad_outputs=grads)
    
        u_x = J_U[0].reshape([dim])
        u_y = J_U[1].reshape([dim])
        u_t = J_U[2].reshape([dim])

        u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=grads)[0]

        u_yy = torch.autograd.grad(u_y, y, create_graph=True, grad_outputs=grads)[0]

        u_xx = u_xx.reshape([dim])
        u_yy = u_yy.reshape([dim])

        return u, u_x, u_y, u_xx, u_yy, u_t
    
    def loss_ic(self, x, y, t, u0):
        """
        Returns the quality of the net
        """
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        inputX = torch.stack([x, y, t], 1)
        u = self.forward(inputX).view(-1)
        u0 = u0.view(-1)

        loss = (torch.mean((u0 - u)**2))

        return loss
