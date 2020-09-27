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
from Schrodinger2DDatasets import SchrodingerEquationDataset
from Schrodinger2D_functions import *
import matplotlib.pyplot as plt
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib


class SchrodingerNet(nn.Module):
    def __init__(self, lb, ub, samplingX, samplingY, activation=torch.tanh, noLayers=8, noFeatures=100, use_singlenet=True, ssim_windowSize=9, initLayers=True, useGPU=False):
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
        self.activation = activation
        self.useGPU = useGPU
        self.lb = torch.Tensor(lb).float()
        self.ub = torch.Tensor(ub).float()
        if(self.useGPU):
            self.lb = self.lb.cuda()
            self.ub = self.ub.cuda()
        self.noFeatures = noFeatures
        self.noLayers = noLayers
        self.use_singlenet = use_singlenet

        #self.ssim_loss = ssim.MSSSIM(window_size = ssim_windowSize)
        # Creating Weight Matrix for energy conservation mechanism
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
        self.W = torch.Tensor(W).float()
        if(self.useGPU):
            self.W = self.W.cuda()

        self.in_t = nn.ModuleList([])
        if(initLayers):
            self.init_layers()

    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
        print("layers", self.noLayers)
        self.in_t.append(nn.Linear(3, self.noFeatures))  # time
        for _ in range(self.noLayers):
            self.in_t.append(nn.Linear(self.noFeatures, self.noFeatures))
        self.in_t.append(nn.Linear(self.noFeatures, 2))

        for m in self.in_t:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # scale spatiotemporral coordinate to [-1,1]
        t_in = (x - self.lb)/(self.ub - self.lb)

        # only normalize to [-1,1] in case of tanh
        # if(self.activation == torch.tanh):
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

        # torch.cuda.empty_cache()
        dim = x.shape[0]  # defines the shape of the gradient

        # save input in variabeles is necessary for gradient calculation
        x = Variable(x, requires_grad=True).cuda()
        y = Variable(y, requires_grad=True).cuda()
        t = Variable(t, requires_grad=True).cuda()

        X = torch.stack([x, y, t], 1)

        UV = self.forward(X)
        u = UV[:, 0]
        v = UV[:, 1]
        grads = torch.ones([dim]).cuda()

        # huge change to the tensorflow implementation this function returns all neccessary gradients
        J_U = torch.autograd.grad(
            u, [x, y, t], create_graph=True, grad_outputs=grads)
        J_V = torch.autograd.grad(
            v, [x, y, t], create_graph=True, grad_outputs=grads)

        u_x = J_U[0].reshape([dim])
        u_y = J_U[1].reshape([dim])
        u_t = J_U[2].reshape([dim])

        v_x = J_V[0].reshape([dim])
        v_y = J_V[1].reshape([dim])
        v_t = J_V[2].reshape([dim])

        u_xx = torch.autograd.grad(
            u_x, x, create_graph=True, grad_outputs=grads)[0]
        v_xx = torch.autograd.grad(
            v_x, x, create_graph=True, grad_outputs=grads)[0]

        u_yy = torch.autograd.grad(
            u_y, y, create_graph=True, grad_outputs=grads)[0]
        v_yy = torch.autograd.grad(
            v_y, y, create_graph=True, grad_outputs=grads)[0]

        u_xx = u_xx.reshape([dim])
        v_xx = v_xx.reshape([dim])
        u_yy = u_yy.reshape([dim])
        v_yy = v_yy.reshape([dim])

        return u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t

    def net_pde(self, x, y, t, omega=1.):
        """
        Calculates the quality of the pde estimation
        :param x postion x
        :param t time t
        :param omega frequency of the harmonic oscillator 
        """
        # get predicted solution and the gradients
        u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = self.net_uv(x, y, t)
        x = x.view(-1)
        y = y.view(-1)

        # calculate loss for real and imaginary part seperatly
        # fu is the real part of the schrodinger equation
        f_u = -1 * u_t - 0.5 * v_xx - 0.5 * v_yy + omega * \
            0.5 * (x ** 2) * v + omega * 0.5 * (y ** 2) * v
        # fv is the imaginary part of the schrodinger equation
        f_v = -1 * v_t + 0.5 * u_xx + 0.5 * u_yy - omega * \
            0.5 * (x ** 2) * u - omega * 0.5 * (y ** 2) * u
        return u, v, f_u, f_v

    def loss_ic(self, x, y, t, u0, v0, filewriter=None, epoch=0, w_ssim=0):
        """
        Returns the quality of the net
        """
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        inputX = torch.stack([x, y, t], 1)
        UV = self.forward(inputX)
        u = UV[:, 0].view(-1, 1)
        v = UV[:, 1].view(-1, 1)

        u0 = u0.view(-1, 1)
        v0 = v0.view(-1, 1)

        loss = (torch.mean((u0 - u) ** 2) + torch.mean((v0 - v) ** 2))

        return loss

    def loss_uv(self, x, y, t, u0, v0, filewriter=None, epoch=0, w_ssim=0):
        """
        Returns the quality of the net
        """
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        inputX = torch.stack([x, y, t], 1)
        UV = self.forward(inputX)
        u = UV[:, 0].view(-1, 1)
        v = UV[:, 1].view(-1, 1)

        u0 = u0.view(-1, 1)
        v0 = v0.view(-1, 1)

        loss_u = (torch.mean((u0 - u) ** 2))
        loss_v = (torch.mean((v0 - v) ** 2))

        return loss_u, loss_v

    def loss_pde(self, x0, y0, t0, u0, v0, xf, yf, tf, xe, ye, te, c, samplingX, samplingY, activateEnergyLoss=True, alpha=1.):
        # reshape all inputs into correct shape
        x0 = x0.view(-1)
        y0 = y0.view(-1)
        t0 = t0.view(-1)
        xf = xf.view(-1)
        yf = yf.view(-1)
        tf = tf.view(-1)
        xe = xe.view(-1)
        ye = ye.view(-1)
        te = te.view(-1)
        n0 = x0.shape[0]
        nf = xf.shape[0]
        inputX = torch.cat([x0, xf, xe])
        inputY = torch.cat([y0, yf, ye])
        inputT = torch.cat([t0, tf, te])

        u, v, f_u, f_v = self.net_pde(inputX, inputY, inputT)

        solU = u[:n0]
        solV = v[:n0]

        pdeLoss = alpha * torch.mean((solU - u0) ** 2) + \
            alpha * torch.mean((solV - v0) ** 2) + \
            torch.mean(f_u ** 2) + \
            torch.mean(f_v ** 2)

        if activateEnergyLoss:
            eU = u[n0 + nf:]
            eV = v[n0 + nf:]
            eH = eU ** 2 + eV ** 2

            lowerX = self.lb[0]
            higherX = self.ub[0]

            lowerY = self.lb[1]
            higherY = self.ub[1]

            disX = (higherX - lowerX) / samplingX
            disY = (higherY - lowerY) / samplingY

            u0 = u0.view(-1)
            v0 = v0.view(-1)
            integral = 0.25 * disX * disY * torch.sum(eH * self.W)
            # calculte integral over field for energy conservation
            eLoss = (integral - c) ** 2
            pdeLoss = pdeLoss + eLoss

        return pdeLoss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier",
                        type=str, default="S2D_baseline")
    parser.add_argument("--batchsize", dest="batchsize",
                        type=int, default=10000)
    parser.add_argument("--numbatches", dest="numBatches",
                        type=int, default=300)
    parser.add_argument("--epochsIC", dest="epochsSolution",
                        type=int, default=2000)
    parser.add_argument("--epochsPDE", dest="epochsPDE",
                        type=int, default=50000)
    parser.add_argument("--initsize", dest="initSize", type=int, default=40000)
    parser.add_argument("--energyloss", dest="energyLoss", type=int, default=1)
    parser.add_argument("--pretraining", dest="pretraining",
                        type=int, default=1)
    parser.add_argument("--alpha", dest="alpha", type=float, default=12.)
    parser.add_argument("--noFeatures", dest="noFeatures",
                        type=int, default=300)
    parser.add_argument("--noLayers", dest="noLayers", type=int, default=8)
    parser.add_argument("--pModel", dest="pModel", type=str, default="")
    args = parser.parse_args()

    # grab constants
    coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, tmax = getDefaults()

    # adapter of commandline parameters
    modelPath = 'results/models/' + args.identifier + '/'
    logdir = 'results/logs/' + args.identifier
    batchSizePDE = args.batchsize
    useGPU = True
    numBatches = args.numBatches
    initSize = args.initSize
    numEpochsSolution = args.epochsSolution
    numEpochsPDE = args.epochsPDE
    activateEnergyLoss = args.energyLoss
    pretraining = args.pretraining
    activation = torch.tanh

    # create modelpath
    pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True)
    # create logWriter
    log_writer = SummaryWriter(logdir)
    # create dataset
    ds = SchrodingerEquationDataset(coordinateSystem, numOfEnergySamplingPointsX,
                                    numOfEnergySamplingPointsY, 40000, numBatches, batchSizePDE, useGPU=True)

    # Partition dataset among workers using DistributedSampler
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1)

    model = SchrodingerNet(ds.lb, ds.ub, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY,
                           activation=activation, noFeatures=args.noFeatures, noLayers=args.noLayers).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    """
    optimize PINN on initial condition
    """
    if pretraining:
        for epoch in range(numEpochsSolution):
            start_time = time.time()
            for x0, y0, t0, Ex_u, Ex_v, xf, yf, tf, xe, ye, te in train_loader:
                optimizer.zero_grad()
                loss = model.loss_ic(x0, y0, t0, Ex_u, Ex_v, log_writer, epoch)
                loss.backward()
                optimizer.step()
                break

            if ((epoch % 30 == 0)):
                print("[%d] IC loss: %.4e [%.2fs]" %
                      (epoch, loss.item(), time.time() - start_time))
                log_writer.add_scalar("L_IC", loss.item(), epoch)
                writeIntermediateState(
                    0, model, epoch, log_writer, coordinateSystem, identifier="PT")
        save_checkpoint(model, modelPath+"1_pretraining/", epoch)

    """
    optimize PINN on whole domain
    """
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = 9e-6
    # create dataset
    ds = SchrodingerEquationDataset(coordinateSystem, numOfEnergySamplingPointsX,
                                    numOfEnergySamplingPointsY, initSize, numBatches, batchSizePDE, useGPU=True)

    # Partition dataset among workers using DistributedSampler
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1)

    for epoch in range(numEpochsPDE):
        start_time = time.time()
        for x0, y0, t0, Ex_u, Ex_v, xf, yf, tf, xe, ye, te in train_loader:
            optimizer.zero_grad()
            loss = model.loss_pde(x0,
                                  y0,
                                  t0,
                                  Ex_u,
                                  Ex_v,
                                  xf,
                                  yf,
                                  tf,
                                  xe,
                                  ye,
                                  te,
                                  1.,
                                  numOfEnergySamplingPointsX,
                                  numOfEnergySamplingPointsY,
                                  activateEnergyLoss,
                                  args.alpha)
            loss.backward()
            optimizer.step()

        if ((epoch % 30 == 0)):
            print("[%d] PDE loss: %.4e [%.2fs]" %
                  (epoch, loss.item(), time.time() - start_time))
            log_writer.add_scalar("L_PDE", loss.item(), epoch)
            writeIntermediateState(
                0, model, epoch, log_writer, coordinateSystem)
            writeIntermediateState(
                250, model, epoch, log_writer, coordinateSystem)
            writeIntermediateState(
                500, model, epoch, log_writer, coordinateSystem)
            writeIntermediateState(
                750, model, epoch, log_writer, coordinateSystem)
            writeIntermediateState(
                1000, model, epoch, log_writer, coordinateSystem)
            # track_coefficient(model,log_writer)
            writeValidationLoss(model, log_writer, 0, epoch, coordinateSystem)
            writeValidationLoss(model, log_writer, 250,
                                epoch, coordinateSystem)
            writeValidationLoss(model, log_writer, 500,
                                epoch, coordinateSystem)
            writeValidationLoss(model, log_writer, 750,
                                epoch, coordinateSystem)
            writeValidationLoss(model, log_writer, 1000,
                                epoch, coordinateSystem)
            save_checkpoint(model, modelPath+"3_pde/",
                            epoch + numEpochsSolution)

            sys.stdout.flush()

    print("--- converged ---")
    if log_writer:
        hParams = {'numLayers': numLayers,
                   'numFeatures': numFeatures,
                   'ResidualPoints': numBatches * batchSizePDE,
                   'alpha': args.alpha,
                   'args': args}

        valLoss0 = valLoss(model, 0, coordinateSystem)
        valLoss250 = valLoss(model, 250, coordinateSystem)
        valLoss500 = valLoss(model, 500, coordinateSystem)
        valLoss750 = valLoss(model, 750, coordinateSystem)
        valLoss1000 = valLoss(model, 1000, coordinateSystem)

        metric = {'hparam/SimLoss': loss.item(),
                  'hparam/valLoss0': valLoss0,
                  'hparam/valLoss250': valLoss250,
                  'hparam/valLoss500': valLoss500,
                  'hparam/valLoss500': valLoss750,
                  'hparam/valLoss1000': valLoss1000
                  }

        log_writer.add_hparams(hParams, metric)
