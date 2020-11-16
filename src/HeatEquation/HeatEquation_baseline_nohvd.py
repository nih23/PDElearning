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
from UKDDataset_segm_upd import SchrodingerEquationDataset
import matplotlib.pyplot as plt
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib

def model_snapshot(model, args, timeStep, dataset, csystem):
    """
    Function returns approximation of the solution for a time step
    """
    seg_matrix = dataset.segmentation(args.pData, 0)
    x,y,t = dataset.getInput(timeStep, csystem, args)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()        
    x = x.view(-1)
    y = y.view(-1)
    t = t.view(-1)

    X = torch.stack([x, y, t], 1)

    UV = model.forward(X)
    
    return (UV.detach().cpu().numpy().reshape(480,640))*seg_matrix

def real_snapshot(args, timeStep, dataset):
    """
    Function returns exact data for a time step
    """
    seg_matrix = dataset.segmentation(args.pData, 0)
    return dataset.loadFrame(args.pData, timeStep)[0].reshape(640,480).T*seg_matrix

def temp_comp(model, args, dataset, csystem, nt, frameStep):
    """
    Function calculates arrays of average exact temperature as well as predicted one for multiple time steps
    """
    temp = [] # exact
    temp_pr = [] # predicted
    for i in range(0,nt,int(frameStep)):
        mT = model_snapshot(model, args, i, dataset, csystem).reshape(-1)
        rT = real_snapshot(args, i, dataset).reshape(-1)
        
        mT = mT[mT != 0]
        rT = rT[rT != 0]
        
        t_pr = np.mean(mT)
        t = np.mean(rT)
        
        temp.append(t)
        temp_pr.append(t_pr)
        
    return temp, temp_pr

def mse(model, args, dataset, csystem):
    """
    Function calculates mean square error for exact and predicted temperature
    """
    data = np.array([])
    prediction = np.array([])
    for i in range(0,3000,25):
        mT = model_snapshot(model, args, i, dataset, csystem).reshape(-1)
        rT = real_snapshot(args, i, dataset).reshape(-1)

        mT = mT[mT != 0]
        rT = rT[rT != 0]

        data = np.append(data,rT)
        prediction = np.append(prediction,mT)

    mse = np.square(np.subtract(data, prediction)).mean()
    
    return mse


def valLoss(model, dataset, timeStep, csystem, args):
    
    seg_matrix = dataset.segmentation(args.pData, timeStep)
    
    x, y, t = dataset.getInput(timeStep, csystem, args)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()
    uPred = UV[:, 0].reshape(-1)
    uPred = (uPred.reshape((480,640))*seg_matrix).reshape(-1)

    # load label data
    uVal,_ = dataset.loadFrame(args.pData, timeStep)
    uVal = np.array(uVal).reshape(-1)
    uVal = (uVal.reshape((640,480)).T*seg_matrix).reshape(-1)

    valLoss_u = np.max(abs(uVal - uPred)) 
    valSqLoss_u = np.sqrt(np.sum(np.power(uVal - uPred,2)))

    return valLoss_u, valSqLoss_u
    
def check(model, dataset, csystem, args):
    """
    Function returns square and infinity norm values for miltiple frames as well as frames sorted by square norm value
    Can be used for re-initialization of dataset
    """
    norms_sq = []
    norms_inf = []
    for i in range(0, args.nt, int(args.frameStep)):
    #for i in range(args.nt):
        valLoss_u, valSqLoss_u = valLoss(model, dataset, i,csystem, args)
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
        # scale spatiotemporral coordinates to [-1,1]
        t_in = (x - self.lb)/(self.ub - self.lb)
        t_in = 2.0 * t_in - 1.0
        for i in range(len(self.in_t)-1):
            t_in = self.in_t[i](t_in)
            t_in = self.activation(t_in)
        return self.in_t[-1](t_in)
                

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

        UV = self.forward(X)
        u = UV[:, 0]
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
    
    def loss_ic(self, x, y, t, u0, filewriter=None, epoch = 0, w_ssim = 0):
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
