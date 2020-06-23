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

from Schrodinger.Dataset.Baseline import SchrodingerEquationDataset

import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F


class SchrodingerNet(nn.Module):
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
        super(SchrodingerNet, self).__init__()

        self.numLayers = numLayers
        self.numFeatures = numFeatures
        self.lin_layers = nn.ModuleList()
        self.activation = activation

        self.numLayers_hpm = numLayers_hpm
        self.numFeatures_hpm = numFeatures_hpm
        self.lin_layers_hpm = nn.ModuleList()
        self.activation_hpm = activation_hpm

        self.lb = torch.Tensor(lb).float().cuda()
        self.ub = torch.Tensor(ub).float().cuda()
    
        W = np.zeros((samplingX, samplingY))
        W[0, 0] = 1
        W[0, samplingY - 1] = 1
        W[samplingX - 1, samplingY - 1] = 1
        W[samplingX - 1, 0] = 1

        for idx in range(1, samplingX - 1):
            W[idx, 0] = 2
            W[idx, samplingY - 1] = 2

        for idx in range(1, samplingY - 1):
            W[0, idx] = 2
            W[samplingX - 1, idx] = 2

        for i in range(1, samplingX - 1):
            for j in range(1, samplingY - 1):
                W[i, j] = 4

        W = W.reshape(-1)
        self.W = torch.Tensor(W).float().cuda()

        # building the neural network
        self.init_layers()

        
        
    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """

        self.lin_layers.append(nn.Linear(3, self.numFeatures))
        for _ in range(self.numLayers):
            inFeatures = self.numFeatures
            self.lin_layers.append(nn.Linear(inFeatures, self.numFeatures))
        self.lin_layers.append(nn.Linear(inFeatures, 2))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.lin_layers_hpm.append(nn.Linear(11, self.numFeatures))
        for _ in range(self.numLayers_hpm):
            inFeatures = self.numFeatures_hpm
            self.lin_layers_hpm.append(nn.Linear(inFeatures, self.numFeatures_hpm))
        self.lin_layers_hpm.append(nn.Linear(inFeatures, 2))

        for m in self.lin_layers_hpm:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def net_uv(self, x, y, t):
        """
        Function that calculates the nn output at postion x,y at time t
        :param x: position
        :param y: position
        :param t: time
        :return: Solutions and their gradients
        """

        dim = x.shape[0]
        x = Variable(x, requires_grad=True).cuda()
        y = Variable(y, requires_grad=True).cuda()
        t = Variable(t, requires_grad=True).cuda()

        X = torch.stack([x, y, t], 1)

        UV = self.forward(X)
        u = UV[:, 0]
        v = UV[:, 1]

        grads = torch.ones([dim]).cuda()
        # compute first-order partial derivative by automatic differentiation
        J_U = torch.autograd.grad(u,[x,y,t], create_graph=True, grad_outputs=grads)
        J_V = torch.autograd.grad(v,[x,y,t], create_graph=True, grad_outputs=grads)
    
        u_x = J_U[0].reshape([dim])
        u_y = J_U[1].reshape([dim])
        u_t = J_U[2].reshape([dim])

        v_x = J_V[0].reshape([dim])
        v_y = J_V[1].reshape([dim])
        v_t = J_V[2].reshape([dim])

		# compute second order partial derivative
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        v_xx = torch.autograd.grad(v_x, x, create_graph=True, grad_outputs=grads)[0]

        u_yy = torch.autograd.grad(u_y, y, create_graph=True, grad_outputs=grads)[0]
        v_yy = torch.autograd.grad(v_y, y, create_graph=True, grad_outputs=grads)[0]

        u_xx = u_xx.reshape([dim])
        v_xx = v_xx.reshape([dim])
        u_yy = u_yy.reshape([dim])
        v_yy = v_yy.reshape([dim])

        return u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t


    def forward(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(0, len(self.lin_layers) - 1):
            x = self.lin_layers[i](x)
            x = self.activation(x)
        x = self.lin_layers[-1](x)

        return x


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


    def solution_loss(self, x, y, t, u0, v0):
        """
        Returns the quality of the net
        """
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        inputX = torch.stack([x, y, t], 1)
        UV = self.forward(inputX)
        u = UV[:, 0]
        v = UV[:, 1]

        u0 = u0.view(-1)
        v0 = v0.view(-1)

        loss = torch.mean((u0 - u) ** 2) + torch.mean(v ** 2)
        return loss


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


def writeIntermediateState(timeStep, model, epoch, nx, ny, fileWriter,csystem):
    if fileWriter:
        x, y, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
        x = torch.Tensor(x).float().cuda()
        y = torch.Tensor(y).float().cuda()
        t = torch.Tensor(t).float().cuda()

        inputX = torch.stack([x, y, t], 1)
        UV = model.forward(inputX).detach().cpu().numpy()

        u = UV[:, 0].reshape((nx, ny))
        v = UV[:, 1].reshape((nx, ny))

        h = u ** 2 + v ** 2

        fig = plt.figure()
        plt.imshow(u, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('Real_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(v, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('Imaginary_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(h, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('Norm_' + str(timeStep), fig, epoch)
        plt.close(fig)


def valLoss(model, timeStep, csystem):
    x, y, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()
    uPred = UV[:, 0]
    vPred = UV[:, 1]

    # load label data
    uVal, vVal = SchrodingerEquationDataset.loadFrame(pData, timeStep)
    uVal = np.array(uVal)
    vVal = np.array(vVal)

    valLoss = np.mean((uVal - uPred) ** 2) + np.mean((vVal - vPred) ** 2)
    return valLoss


def writeValidationLoss(model, writer, timeStep, epoch, csystem):
    if writer:
        loss = valLoss(model, timeStep,csystem)
        writer.add_scalar("ValidationLoss_" + str(timeStep), loss, epoch)


def save_checkpoint(model, optimizer, path, epoch):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path + 'model_' + str(epoch))


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == "__main__":

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    # static parameter
    nx = 200
    ny = 200
    nt = 1000
    xmin = -2.5
    xmax = 2.5
    ymin = -2.5
    ymax = 2.5
    dt = 0.001
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx": nx , "ny": ny, "nt": nt, "dt": dt}

    pData = '../data/'
    batchSizeInit = 2500  #for the balanced dataset is not needed

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str, default="S2D_DeepHPM")
    parser.add_argument("--batchsize", dest="batchsize", type=int, default=10000)
    parser.add_argument("--numbatches", dest="numBatches", type=int, default=500)
    parser.add_argument("--numlayers", dest="numLayers", type=int, default=8)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int, default=8)
    parser.add_argument("--numlayers_hpm", dest="numLayers_hpm", type=int, default=3)
    parser.add_argument("--numfeatures_hpm", dest="numFeatures_hpm", type=int, default=100)
    parser.add_argument("--t_ic_tanh",dest="t_ic",type=float, default = 1e-5)
    parser.add_argument("--t_pde",dest="t_pde",type=float, default = 1e-5)
    parser.add_argument("--pretraining", dest="pretraining", type=int, default=1)
    parser.add_argument("--alpha",dest="alpha",type=float, default=12)
    parser.add_argument("--lhs",dest="lhs",type=int, default=0)
    args = parser.parse_args()

    if hvd.rank() == 0: 
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 +  args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
    
    print("Rank",hvd.rank(),"Local Rank", hvd.local_rank())
    
    #adapter of commandline parameters

    modelPath = '/home/s4386479/projects/schrodinger/models/' + args.identifier + '/'
    logdir = '/home/s4386479/projects/schrodinger/runs/experiments/' + args.identifier
    batchSizePDE = args.batchsize
    useGPU = True
    numBatches = args.numBatches
    numLayers = args.numLayers
    numFeatures = args.numFeatures
    numLayers_hpm = args.numLayers_hpm
    numFeatures_hpm = args.numFeatures_hpm
    numEpochsSolution = args.epochsSolution
    numEpochsPDE = args.epochsPDE
    activateEnergyLoss = args.energyLoss
    pretraining = args.pretraining
    #postprocessing = args.postprocessing

    #create modelpath
    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True) 
    # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None

    # create dataset
    ds = SchrodingerEquationDataset(pData, coordinateSystem,  numBatches, batchSizePDE, shuffle=True, useGPU=True)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)

    activation = torch.tanh

    model = SchrodingerNet(numLayers, numFeatures, numLayers_hpm, numFeatures_hpm, ds.lb, ds.ub, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, torch.tanh).cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)

    if pretraining:
        """
        approximate full simulation
        """
    	epoch = 0
       	l_loss = 1
        while(l_loss > args.t_ic):
        	epoch+=1
            for x, y, t, Ex_u, Ex_v in train_loader:
                optimizer.zero_grad()
                # calculate loss
                loss = model.solution_loss(x, y, t, Ex_u, Ex_v)
                loss.backward()
                optimizer.step()
            l_loss = loss.item()
            if epoch % 30 == 0:
                print("Loss at Epoch " + str(epoch) + ": " + str(loss.item()))

    """
    learn non-linear operator N 
    """
	# we need to significantly reduce the learning rate [default: 9e-6]
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = 1e-6

    l_loss = 1
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
        if epoch % 30 == 0:
            writeIntermediateState(0, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(100, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(200, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(250, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(300, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(400, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(500, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeValidationLoss(model, log_writer, 250, epoch,coordinateSystem)
            writeValidationLoss(model, log_writer, 500, epoch,coordinateSystem)
            sys.stdout.flush()

            print("PDE Loss at Epoch: ", epoch + 1, loss.item())
            if log_writer:
                log_writer.add_histogram('First Layer Grads', model.lin_layers[0].weight.grad.view(-1, 1), epoch)
                save_checkpoint(model, optimizer, modelPath, epoch)
