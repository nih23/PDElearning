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
from BaselineDataset import SchrodingerEquationDataset
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import pathlib


class SchrodingerNet(nn.Module):
    def __init__(self, numLayers, numFeatures, lb, ub, activation=torch.tanh):
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
        self.lb = torch.Tensor(lb).float().cuda()
        self.ub = torch.Tensor(ub).float().cuda()

        # building the neural network
        self.init_layers()

        # Creating Weight Matrix for energy conservation mechanism

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

    def net_uv(self, x, y, t):
        """
        Function that calculates the nn output at postion x at time t
        :param x: position
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

        # huge change to the tensorflow implementation this function returns all neccessary gradients
        u_x = torch.autograd.grad(u, x, create_graph=True, grad_outputs=grads)[0]
        v_x = torch.autograd.grad(v, x, create_graph=True, grad_outputs=grads)[0]

        u_y = torch.autograd.grad(u, y, create_graph=True, grad_outputs=grads)[0]
        v_y = torch.autograd.grad(v, y, create_graph=True, grad_outputs=grads)[0]

        u_x = u_x.reshape([dim])
        v_x = v_x.reshape([dim])
        u_y = u_x.reshape([dim])
        v_y = v_x.reshape([dim])

        u_t = torch.autograd.grad(u, t, create_graph=True, grad_outputs=grads)[0]
        v_t = torch.autograd.grad(v, t, create_graph=True, grad_outputs=grads)[0]

        u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        v_xx = torch.autograd.grad(v_x, x, create_graph=True, grad_outputs=grads)[0]

        u_yy = torch.autograd.grad(u_y, y, create_graph=True, grad_outputs=grads)[0]
        v_yy = torch.autograd.grad(v_y, y, create_graph=True, grad_outputs=grads)[0]

        u_t = u_t.reshape([dim])
        v_t = v_t.reshape([dim])

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
            # x = torch.sin(x)
            # x = F.tanh(x)
        x = self.lin_layers[-1](x)

        return x

    def net_pde(self, x, y, t):
        """
        Calculates the quality of the pde estimation
        :param x postion x
        :param t time t
        """
        u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = self.net_uv(x, y, t)
        x = x.view(-1)
        y = y.view(-1)

        f_u = -1 * u_t - 0.5 * v_xx - 0.5 * v_yy + 0.5 * (x ** 2) * v + 0.5 * (y ** 2) * v
        f_v = -1 * v_t + 0.5 * u_xx + 0.5 * u_yy - 0.5 * (x ** 2) * u - 0.5 * (y ** 2) * u
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

        u, v, f_u, f_v = self.net_pde(inputX, inputY, inputT)

        solU = u[:n0]
        solV = v[:n0]

        u0 = u0.view(-1)
        v0 = v0.view(-1)

        pdeLoss = torch.mean((solU - u0) ** 2) + torch.mean((solV - v0) ** 2) + torch.mean(f_u ** 2) + torch.mean(f_v ** 2)
        return pdeLoss

    def supervised_loss(self, x, y, t, u, v):
    
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        exact_u = u.view(-1)
        exact_v = v.view(-1)

        pred_u, pred_v, f_u, f_v = self.net_pde(x, y, t)

        solution_loss = torch.mean((exact_u - pred_u) ** 2) + \
                        torch.mean((exact_v - pred_u) ** 2) 

        pdeLoss = solution_loss + \
                  torch.mean(f_u ** 2) + \
                  torch.mean(f_v ** 2)

        if epoch % 30 == 0:
            if log_writer:
                log_writer.add_scalar('Solution U', torch.mean((exact_u - pred_u) ** 2), epoch)
                log_writer.add_scalar('Solution V', torch.mean((exact_v - pred_v) ** 2), epoch)
                log_writer.add_scalar('Real PDE', torch.mean(f_u ** 2), epoch)
                log_writer.add_scalar('Imaginary PDE', torch.mean(f_v ** 2), epoch)
                log_writer.add_scalar('PDE Loss', pdeLoss, epoch)

        return solution_loss 


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


def valLoss(model, timeStep,csystem):
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
    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10
    dt = 0.001
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx": nx , "ny": ny, "nt": nt, "dt": dt}

    pData = '/projects/p_electron/stiller/schrodinger/data/polynomial_sampled/'
    batchSizeInit = 2500  #for the balanced dataset is not needed

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str)
    parser.add_argument("--batchsize", dest="batchsize", type=int)
    parser.add_argument("--numbatches", dest="numBatches", type=int)
    parser.add_argument("--numlayers", dest="numLayers", type=int)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int)
    parser.add_argument("--epochsPDE", dest="epochsPDE", type=int)


    #parser.add_argument("--postprocessing", dest='postprocessing', type=int)

    args = parser.parse_args()

    if hvd.rank() == 0: 
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 +  args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        

    #adapter of commandline parameters

    modelPath = '/projects/p_electron/stiller/schrodinger/models/' + args.identifier + '/'
    logdir = '/projects/p_electron/stiller/schrodinger/runs/experiments/' + args.identifier
    batchsize = args.batchsize
    useGPU = True
    numBatches = args.numBatches
    numLayers = args.numLayers
    numFeatures = args.numFeatures
    numEpochsPDE = args.epochsPDE

    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True) 

    # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None

    # create dataset
    ds = SchrodingerEquationDataset(pData, coordinateSystem, numBatches, batchsize, shuffle=True, useGPU=True)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)

    activation = torch.tanh

    model = SchrodingerNet(numLayers, numFeatures, ds.lb, ds.ub, torch.tanh).cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)


    for epoch in range(numEpochsPDE):

        for x,y,t,u,v in train_loader:
            optimizer.zero_grad()
            # calculate loss
            loss = model.supervised_loss(x,y,t,u,v)
            loss.backward()
            optimizer.step()

        if epoch % 30 == 0:
            writeIntermediateState(0, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(250, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(500, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(750, model, epoch, nx, ny, log_writer,coordinateSystem)
            writeValidationLoss(model, log_writer, 250, epoch,coordinateSystem)
            writeValidationLoss(model, log_writer, 500, epoch,coordinateSystem)
            writeValidationLoss(model, log_writer, 750, epoch,coordinateSystem)

            print("PDE Loss at Epoch: ", epoch + 1, loss.item())
            if log_writer:
                log_writer.add_histogram('First Layer Grads', model.lin_layers[0].weight.grad.view(-1, 1), epoch)
                save_checkpoint(model, optimizer, modelPath, epoch)