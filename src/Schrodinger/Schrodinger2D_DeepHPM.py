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

from Schrodinger.Dataset.Schrodinger2DDatasets import SchrodingerHPMEquationDataset
from Schrodinger.Schrodinger2D_baseline_nohvd import getDefaults, writeIntermediateState, valLoss, save_checkpoint, load_checkpoint, writeValidationLoss, SchrodingerNet


import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F


class SchrodingerHPMNet(SchrodingerNet):
    def __init__(self, numLayers, numFeatures, numLayers_hpm, numFeatures_hpm, lb, ub, samplingX, samplingY, activation=torch.tanh, activation_hpm=F.relu):
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
        super().__init__(lb, ub, samplingX, samplingY, activation, noLayers = numLayers, noFeatures = numFeatures, use_singlenet = True, ssim_windowSize = 9, initLayers = True, useGPU = True)

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

        self.lin_layers_hpm.append(nn.Linear(11, self.noFeatures_hpm))
        for _ in range(self.noLayers_hpm):
            self.lin_layers_hpm.append(nn.Linear(self.noFeatures_hpm, self.noFeatures_hpm))
        self.lin_layers_hpm.append(nn.Linear(self.noFeatures_hpm, 2))

        
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
        u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = self.net_uv(x, y, t)
        x = x.view(-1)
        y = y.view(-1)

        X = torch.stack([x, y, t, u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t], 1)

        f = torch.stack([-1 * u_t, -1 * v_t], 1) - self.forward_hpm(X)
        f_u = f[:, 0]
        f_v = f[:, 1]

        return u, v, f_u, f_v


    def hpm_loss(self, x, y, t, Ex_u, Ex_v):
        """
        Returns the quality HPM net
        """

        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u, v, f_u, f_v = self.net_pde(x, y, t)

        Ex_u = Ex_u.view(-1)
        Ex_v = Ex_v.view(-1)

        hpmLoss = torch.mean((u - Ex_u) ** 2) + torch.mean((v - Ex_v) ** 2) + torch.mean(f_u ** 2) + torch.mean(f_v ** 2)
        return hpmLoss
   
    
if __name__ == "__main__":

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str, default="S2D_DeepHPM")
    parser.add_argument("--pData", dest="pData", type=str, default="data/Schrodinger/S2D_-3_to_3/")
    parser.add_argument("--batchsize", dest="batchsize", type=int, default=10000)
    parser.add_argument("--numbatches", dest="numBatches", type=int, default=500)
    parser.add_argument("--numlayers", dest="numLayers", type=int, default=8)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int, default=8)
    parser.add_argument("--numlayers_hpm", dest="numLayers_hpm", type=int, default=3)
    parser.add_argument("--numfeatures_hpm", dest="numFeatures_hpm", type=int, default=100)
    parser.add_argument("--t_ic",dest="t_ic",type=float, default = 1e-4)
    parser.add_argument("--t_pde",dest="t_pde",type=float, default = 1e-5)
    parser.add_argument("--pretraining", dest="pretraining", type=int, default=1)
    args = parser.parse_args()

    if hvd.rank() == 0: 
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 +  args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
       
    #set constants for training
    coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, tmax = getDefaults()   
    modelPath = 'results/models/' + args.identifier + '/'
    logdir = 'results/experiments/' + args.identifier + '/'
    useGPU = True
    activation = torch.tanh

    
    #create modelpath
    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True) 
    # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None

    # create dataset
    ds = SchrodingerHPMEquationDataset(args.pData, coordinateSystem, args.numBatches, args.batchsize, shuffle = True, useGPU=True)
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)


    model = SchrodingerHPMNet(args.numLayers, args.numFeatures, args.numLayers_hpm, args.numFeatures_hpm, ds.lb, ds.ub, 5, 5, activation).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)

    # broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optim, root_rank=0)
    
    if args.pretraining:
        """
        approximate full simulation
        """
        epoch = 0
        l_loss = 1
        start_time = time.time()
        while(l_loss > args.t_ic):
            epoch+=1
            for x, y, t, Ex_u, Ex_v in train_loader:
                optimizer.zero_grad()
                # calculate loss
                loss = model.loss_ic(x, y, t, Ex_u, Ex_v)
                loss.backward()
                optimizer.step()
            l_loss = loss.item()
            
            if (epoch % 30 == 0) and log_writer:
                print("[%d] IC loss: %.4e [%.2fs]" % (epoch, loss.item(), time.time() - start_time))
                writeValidationLoss(0, model, epoch, log_writer, coordinateSystem, identifier = "PT")
                writeIntermediateState(0, model, epoch, log_writer, coordinateSystem, identifier = "PT")

    save_checkpoint(model, modelPath+"0_ic/", epoch)
                
    """
    learn non-linear operator N 
    """
    # we need to significantly reduce the learning rate [default: 9e-6]
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = 9e-6

    l_loss = 1
    start_time = time.time()
    while(l_loss > args.t_pde):
        epoch+=1
        for x, y, t, Ex_u, Ex_v in train_loader:
            optimizer.zero_grad()           
            loss = model.hpm_loss(x,
                                  y,
                                  t,
                                  Ex_u,
                                  Ex_v)
            loss.backward()
            optimizer.step()
        
        l_loss = loss.item()
        if (epoch % 30 == 0) and log_writer:
            print("[%d] PDE loss: %.4e [%.2fs]" % (epoch, loss.item(), time.time() - start_time))
            writeIntermediateState(0, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeIntermediateState(100, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeIntermediateState(200, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeIntermediateState(250, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeIntermediateState(300, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeIntermediateState(400, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeIntermediateState(500, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeValidationLoss(0, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeValidationLoss(250, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeValidationLoss(500, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            writeValidationLoss(1000, model, epoch, log_writer, coordinateSystem, identifier = "PDE")
            sys.stdout.flush()

            log_writer.add_histogram('First Layer Grads', model.lin_layers_hpm[0].weight.grad.view(-1, 1), epoch)
                
    save_checkpoint(model, modelPath+"1_pde/", epoch)
    print("--- converged ---")