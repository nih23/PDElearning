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

from HeatEquation.Dataset.Baseline import HeatEquationDataset
from Schrodinger.Schrodinger2D_DeepHPM import SchrodingerNet

import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F


class HeatEquationNet(SchrodingerNet):
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
        super(HeatEquationNet, self).__init__(numLayers, numFeatures, numLayers_hpm, numFeatures_hpm, lb, ub, samplingX, samplingY, activation, activation_hpm)


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
        self.lin_layers.append(nn.Linear(inFeatures, 1))

        for m in self.lin_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.lin_layers_hpm.append(nn.Linear(6, self.numFeatures))
        for _ in range(self.numLayers_hpm):
            inFeatures = self.numFeatures_hpm
            self.lin_layers_hpm.append(nn.Linear(inFeatures, self.numFeatures_hpm))
        self.lin_layers_hpm.append(nn.Linear(inFeatures, 1))

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
        u = self.forward(X)

        grads = torch.ones([dim]).cuda()

        # huge change to the tensorflow implementation this function returns all neccessary gradients
        u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=grads)[0]
        u_y = torch.autograd.grad(u, y, create_graph=True, grad_outputs=grads)[0]
        u_x = u_x.reshape([dim])
        u_y = u_x.reshape([dim])
        u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=grads)[0]
        u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        u_yy = torch.autograd.grad(u_y, y, create_graph=True, grad_outputs=grads)[0]

        u_t = u_t.reshape([dim])
        u_xx = u_xx.reshape([dim])
        u_yy = u_yy.reshape([dim])

        return u, u_yy, u_xx, u_t


    def net_pde(self, x, y, t, gamma=1.):
        """
        Calculates the quality of the pde estimation
        :param x postion x
        :param y postion y
        :param t time t
        """
        u, u_yy, u_xx, u_t = self.net_uv(x, y, t)
        x = x.view(-1)
        y = y.view(-1)

        X = torch.stack([x, y, t, u, u_yy, u_xx], 1)

        f = -1 * u_t - self.forward_hpm(X)

        return u, f


    def pde_loss(self, x0, y0, t0, u0, v0, xf, yf, tf):
        """
        Returns the quality of the net
        """

        x0 = x0.view(-1)
        y0 = y0.view(-1)
        t0 = t0.view(-1)
        xf = xf.view(-1)
        yf = yf.view(-1)
        tf = tf.view(-1)

        n0 = x0.shape[0]

        inputX = torch.cat([x0, xf])
        inputY = torch.cat([y0, yf])
        inputT = torch.cat([t0, tf])

        u, f = self.net_pde(inputX, inputY, inputT)

        solU = u[:n0]

        u0 = u0.view(-1)

        pdeLoss = torch.mean((solU - u0) ** 2) + torch.mean(f ** 2)
        return pdeLoss


    def hpm_loss(self, x, y, t, Ex_u):
        """
        Returns the quality HPM net
        """

        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u, f = self.net_pde(x, y, t)

        Ex_u = Ex_u.view(-1)

        return torch.mean((u - Ex_u) ** 2) + torch.mean(f ** 2)


def writeIntermediateState(timeStep, model, epoch, nx, ny, fileWriter,csystem):
    if fileWriter:
        x, y, t = HeatEquationDataset.getInput(timeStep,csystem)
        x = torch.Tensor(x).float().cuda()
        y = torch.Tensor(y).float().cuda()
        t = torch.Tensor(t).float().cuda()

        inputX = torch.stack([x, y, t], 1)
        u = model.forward(inputX).detach().cpu().numpy()

        fig = plt.figure()
        plt.imshow(u, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('u_' + str(timeStep), fig, epoch)
        plt.close(fig)


def valLoss(model, timeStep, csystem):
    x, y, t = HeatEquationDataset.getInput(timeStep,csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    uPred = model.forward(inputX).detach().cpu().numpy()

    # load label data
    uVal, vVal = HeatEquationDataset.loadFrame(pData, timeStep)
    uVal = np.array(uVal)

    return np.mean((uVal - uPred) ** 2)


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
    nx = 480 
    ny = 640
    nt = 1000
    xmin = -2.5
    xmax = 2.5
    ymin = -2.5
    ymax = 2.5
    dt = 0.016 # roughly 60 Hz

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx": nx , "ny": ny, "nt": nt, "dt": dt}

    pData = 'data/UKD/2018_002/20180115_114053'

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str)
    parser.add_argument("--batchsize", dest="batchsize", type=int)
    parser.add_argument("--numbatches", dest="numBatches", type=int)
    parser.add_argument("--initsize", dest="initSize", type=int)
    parser.add_argument("--numlayers", dest="numLayers", type=int)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int)
    parser.add_argument("--numlayers_hpm", dest="numLayers_hpm", type=int)
    parser.add_argument("--numfeatures_hpm", dest="numFeatures_hpm", type=int)
    parser.add_argument("--epochsPDE", dest="epochsPDE", type=int)
    parser.add_argument("--alpha",dest="alpha",type=float)

    args = parser.parse_args()

    if hvd.rank() == 0: 
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 +  args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
    
    print("Rank",hvd.rank(),"Local Rank", hvd.local_rank())
    
    #adapter of commandline parameters
    modelPath = 'runs/models/' + args.identifier + '/'
    logdir = 'runs/logs/' + args.identifier
    batchSizePDE = args.batchsize
    useGPU = True
    numBatches = args.numBatches
    initSize = args.initSize
    numLayers = args.numLayers
    numFeatures = args.numFeatures
    numLayers_hpm = args.numLayers_hpm
    numFeatures_hpm = args.numFeatures_hpm
    numEpochsPDE = args.epochsPDE
    activation = torch.tanh

    #create modelpath
    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True) 
    # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None

    # create dataset
    ds = HeatEquationDataset(pData, coordinateSystem,  numBatches, batchSizePDE, useGPU=True)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)


    model = HeatEquationNet(numLayers, numFeatures, numLayers_hpm, numFeatures_hpm, ds.lb, ds.ub, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, torch.tanh).cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)

    for epoch in range(numEpochsPDE):

        for x, y, t, Ex_u in train_loader:
            optimizer.zero_grad()

            # calculate loss
            
            loss = model.hpm_loss(x,
                                  y,
                                  t,
                                  Ex_u)
            loss.backward()
            optimizer.step()

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
