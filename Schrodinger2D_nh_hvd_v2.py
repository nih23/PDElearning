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
from SchrodingerBalancedECDataset import SchrodingerEquationDataset
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib
import ssim

class SchrodingerNet(nn.Module):
    def __init__(self, lb, ub, samplingX, samplingY, activation=torch.tanh, noLayers = 8, noFeatures = 100, use_singlenet = True, ssim_windowSize = 9):
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
        self.lb = torch.Tensor(lb).float().cuda()
        self.ub = torch.Tensor(ub).float().cuda()
        self.noFeatures = noFeatures
        self.noLayers = noLayers
        self.use_singlenet = use_singlenet
        self.ssim_loss = ssim.MSSSIM(window_size = ssim_windowSize)
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
        self.W = torch.Tensor(W).float().cuda()

        # building the neural network
        self.init_layers()


    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
     
        params = []

        self.in_t0 = nn.ModuleList()
        self.in_t0.append(nn.Linear(2, self.noFeatures)) # time
        for _ in range(self.noLayers-4):
            self.in_t0.append(nn.Linear(self.noFeatures, self.noFeatures))
        self.in_t0.append(nn.Linear(self.noFeatures,2))

        if(self.use_singlenet):
            print("Initialize single net")
            self.in_t = nn.ModuleList()
            self.in_t.append(nn.Linear(3, self.noFeatures)) # time
            for _ in range(self.noLayers):
                self.in_t.append(nn.Linear(self.noFeatures, self.noFeatures))
            self.in_t.append(nn.Linear(self.noFeatures,2))
            params=[self.in_t, self.in_t0]

        else:
            print("Initialize bi-net")
            self.in_tu = nn.ModuleList()
            self.in_tu.append(nn.Linear(3, self.noFeatures)) # time
            for _ in range(self.noLayers):
                self.in_tu.append(nn.Linear(self.noFeatures, self.noFeatures))
            self.in_tu.append(nn.Linear(self.noFeatures,1))

            self.in_tv = nn.ModuleList()
            self.in_tv.append(nn.Linear(3, self.noFeatures)) # time
            for _ in range(self.noLayers):
                self.in_tv.append(nn.Linear(self.noFeatures, self.noFeatures))
            self.in_tv.append(nn.Linear(self.noFeatures,1))
            params=[self.in_tu, self.in_tv, self.in_t0]
        
        for m in params:
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

        UV = self.forward(X).view(-1,2) 
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
        v_yy = torch.autograd.grad(v_y, y, retain_graph=True, grad_outputs=grads)[0]

        u_t = u_t.reshape([dim])
        v_t = v_t.reshape([dim])

        u_xx = u_xx.reshape([dim])
        v_xx = v_xx.reshape([dim])
        u_yy = u_yy.reshape([dim])
        v_yy = v_yy.reshape([dim])

        return u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t


    def forward_tin0(self, x_in):
        for i in range(0, len(self.in_t0)-1):
            x_in = self.in_t0[i](x_in)
            x_in = self.activation(x_in)
        return self.in_t0[-1](x_in)


    def forward_tin(self, t_in):
        for i in range(0, len(self.in_t)-1):
            t_in = self.in_t[i](t_in)
            t_in = self.activation(t_in)
        return self.in_t[-1](t_in)
 
    
    def forward_tinu(self, t_inu):
        for i in range(0, len(self.in_tu)-1):
            t_inu = self.in_tu[i](t_inu)
            t_inu = self.activation(t_inu)
        return self.in_tu[-1](t_inu)


    def forward_tinv(self, t_inu):
        for i in range(0, len(self.in_tv)-1):
            t_inu = self.in_tv[i](t_inu)
            t_inu = self.activation(t_inu)
        return self.in_tv[-1](t_inu)


    def forward_t0(self,x):
        """
        Calculated the prediction of the net.
        :param x is an array [:,x,t] with positions x and time t
        """
        # scale spatiotemporral coordinate to [-1,1]
        x = 2.0*(x - self.lb[0:2])/(self.ub[0:2] - self.lb[0:2]) - 1.0
        
        # recover solution
        Y0 = self.forward_tin0(x)
        return Y0


    def forward_tall(self,x):
        """
        Calculated the prediction of the net.
        :param x is an array [:,x,t] with positions x and time t
        """
        # scale spatiotemporral coordinate to [-1,1]
        x = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        
        # recover solution
        if(self.use_singlenet):
            # one network to predict real and imaginary part
            Y = self.forward_tin(x)

        else:
            # two network for real and imaginary part
            t_inu = self.forward_tinu(x)
            t_inv = self.forward_tinv(x)

            Y = torch.stack([t_inu, t_inv], 1).view(-1,2)

        return Y 


    def forward(self, x):
        y_t0 = self.forward_t0(x[:,0:2])
        y_tall = self.forward_tall(x)
        x_test = torch.stack([x[:,2], x[:,2]], 1)
        
        Y = torch.where(x_test>0,y_t0+y_tall, y_t0)
        
        return Y


    def net_pde(self, x, y, t, gamma=1., storeEigenfunctions = False):
        """
        Calculates the quality of the pde estimation
        :param x postion x
        :param t time t
        """
        u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = self.net_uv(x, y, t)
        x = x.view(-1)
        y = y.view(-1)

        f_u = -1 * u_t - 0.5 * v_xx - 0.5 * v_yy + gamma* 0.5 * (x ** 2) * v + gamma * 0.5 *  (y ** 2) * v
        f_v = -1 * v_t + 0.5 * u_xx + 0.5 * u_yy - gamma* 0.5 * (x ** 2) * u - gamma * 0.5 * (y ** 2) * u
        return u, v, f_u, f_v

    
    def solution_loss(self, x, y, t, u0, v0, filewriter=None, epoch = 0, w_ssim = 0):
        """
        Returns the quality of the net
        """
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        inputX = torch.stack([x, y, t], 1)
        UV = self.forward(inputX)
        u = UV[:, 0].view(-1,1)
        v = UV[:, 1].view(-1,1)

        u0 = u0.view(-1,1)
        v0 = v0.view(-1,1)

        loss = 0


        #SSIM
#        loss = loss + w_ssim * (-self.ssim_loss(v, v0) + -self.ssim_loss(u, u0))

        #L2
        loss = loss + (1-w_ssim) * (torch.mean((u0 - u) ** 2) + torch.mean(v ** 2))

        #L1
        #loss = loss + (1-w_ssim) * (torch.mean(torch.abs(u0 - u)) + torch.mean(torch.abs(v)))

        if(False and filewriter and ((epoch % 30) == 0)):
            u0 = u0.view(200,200).cpu().detach().numpy()
            v0 = v0.view(200,200).cpu().detach().numpy()
            h = u0**2 + v0**2
            fig = plt.figure()
            plt.imshow(h, cmap='jet')
            plt.colorbar()
            filewriter.add_figure('Hpt_gt', fig, epoch)
            plt.close(fig)

            u0 = u.view(200,200).cpu().detach().numpy()
            v0 = v.view(200,200).cpu().detach().numpy()
            h = u0**2 + v0**2
            fig = plt.figure()
            plt.imshow(h, cmap='jet')
            plt.colorbar()
            filewriter.add_figure('Hpt_pred', fig, epoch)
            plt.close(fig)
        
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

    def ec_pde_loss(self, x0, y0, t0, u0, v0, xf, yf, tf, xe, ye, te, c, samplingX, samplingY,activateEnergyLoss=True, alpha=1.):
    
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

        u, v, f_u, f_v = self.net_pde(inputX, inputY, inputT, storeEigenfunctions = True)
        solU = u[:n0]
        solV = v[:n0]

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

        #ef_orthogonality_loss = self.approximateFunctionLosses(self.ef_at_x0[:n0,])

        pdeLoss = alpha * torch.mean((solU - u0) ** 2) + \
                  torch.mean((solV - v0) ** 2) + \
                  torch.mean(f_u ** 2) + \
                  torch.mean(f_v ** 2) 

        if activateEnergyLoss:
            pdeLoss = pdeLoss + eLoss

        if epoch % 30 == 0:
            if log_writer:
                log_writer.add_scalar('u-u_gt', torch.mean((solU - u0) ** 2), epoch)
                log_writer.add_scalar('v-v_gt', torch.mean((solV - v0) ** 2), epoch)
                log_writer.add_scalar('f_u', torch.mean(f_u ** 2), epoch)
                log_writer.add_scalar('f_v', torch.mean(f_v ** 2), epoch)
                log_writer.add_scalar('L_Energy', eLoss, epoch)
                log_writer.add_scalar('L_all', pdeLoss, epoch)
                log_writer.add_scalar('Integral', integral, epoch)
#                log_writer.add_scalar('L_EF_orth', ef_orthogonality_loss.item(), epoch)

        return pdeLoss


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

        h = np.sqrt(u ** 2 + v ** 2)

        UV_t0 = model.forward_t0(torch.stack([x, y], 1)).detach().cpu().numpy()
        u_t0 = UV_t0[:, 0].reshape((nx, ny))
        v_t0 = UV_t0[:, 1].reshape((nx, ny))
        h_t0 = np.sqrt(u_t0 ** 2 + v_t0 ** 2)

        fig = plt.figure()
        plt.imshow(u, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('u_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(v, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('v_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(h, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('h_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(h_t0, cmap='jet')
        plt.colorbar()
        fileWriter.add_figure('ht0_' + str(timeStep), fig, epoch)
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
    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10
    dt = 0.001
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx": nx , "ny": ny, "nt": nt, "dt": dt}

    pData = 'data/'
    batchSizeInit = 2500  #for the balanced dataset is not needed

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str)
    parser.add_argument("--batchsize", dest="batchsize", type=int)
    parser.add_argument("--numbatches", dest="numBatches", type=int)
    parser.add_argument("--initsize", dest="initSize", type=int)
    parser.add_argument("--epochssolution", dest="epochsSolution", type=int)
    parser.add_argument("--epochsPDE", dest="epochsPDE", type=int)
    parser.add_argument("--energyloss", dest="energyLoss",type=int)
    parser.add_argument("--pretraining", dest="pretraining", type=int)
    parser.add_argument("--alpha",dest="alpha",type=float, default=1.)
    parser.add_argument("--lhs",dest="lhs",type=int)
    parser.add_argument("--noFeatures",dest="noFeatures",type=int, default=100)
    parser.add_argument("--noLayers",dest="noLayers",type=int, default=8)
    parser.add_argument("--binet",dest="singlenet",action='store_false')
    parser.set_defaults(singlenet=True)
    args = parser.parse_args()
    if hvd.rank() == 0: 
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 +  args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
    
    print("Rank",hvd.rank(),"Local Rank", hvd.local_rank())
    
    #adapter of commandline parameters
    print(args)
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
    useSingleNet = args.singlenet 
   #postprocessing = args.postprocessing

    #create modelpath
    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True) 
    # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None
    # create dataset
    ds = SchrodingerEquationDataset(pData, coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, 40000, numBatches, batchSizePDE, shuffle=False, useGPU=True,do_lhs=args.lhs)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)


    model = SchrodingerNet(ds.lb, ds.ub, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, activation=activation, noFeatures = args.noFeatures, noLayers = args.noLayers, use_singlenet = useSingleNet).cuda()
    #optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = optim.Adam(list(model.in_t0.parameters()), lr=1e-5)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    if pretraining:
        for epoch in range(numEpochsSolution):
            for x0, y0, t0, Ex_u, Ex_v, xf, yf, tf, xe, ye, te in train_loader:
                optimizer.zero_grad()
                # calculate loss
                loss = model.solution_loss(x0, y0, t0, Ex_u, Ex_v, log_writer, epoch)
                loss.backward()
                optimizer.step()
            if epoch % 30 == 0:
                print("Loss at Epoch " + str(epoch) + ": " + str(loss.item()))
                writeIntermediateState(0, model, epoch, nx, ny, log_writer,coordinateSystem)
#        model.fix_eigenfunction_weights()

#    for paramGroup in optimizer.param_groups:
#        paramGroup['lr'] = 1e-6

    # create dataset
    ds = SchrodingerEquationDataset(pData, coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, initSize, numBatches, batchSizePDE, shuffle=False, useGPU=True,do_lhs=args.lhs)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)

    if(useSingleNet):
        optimizer_pde = optim.Adam(list(model.in_t.parameters()), lr=1e-6)
    else:
        optimizer_pde = optim.Adam(list(model.in_tu.parameters()) + list(model.in_tv.parameters()), lr=1e-6)


    optimizer_pde = hvd.DistributedOptimizer(optimizer_pde,
                      named_parameters=model.named_parameters(),
                      backward_passes_per_step=1)

    for param in model.in_t0.parameters():
        param.requires_grad = False

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    for epoch in range(numEpochsPDE):

        for x0, y0, t0, Ex_u, Ex_v, xf, yf, tf, xe, ye, te in train_loader:
            optimizer_pde.zero_grad()

            # calculate loss
            loss = model.ec_pde_loss(x0,
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
            optimizer_pde.step()

        if epoch % 30 == 0:
            epoch_l = epoch + numEpochsSolution
            writeIntermediateState(1, model, epoch_l, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(100, model, epoch_l, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(200, model, epoch_l, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(250, model, epoch_l, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(300, model, epoch_l, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(400, model, epoch_l, nx, ny, log_writer,coordinateSystem)
            writeIntermediateState(500, model, epoch_l, nx, ny, log_writer,coordinateSystem)
            #writeValidationLoss(model, log_writer, 250, epoch_l,coordinateSystem)
            #writeValidationLoss(model, log_writer, 500, epoch_l,coordinateSystem)
            sys.stdout.flush()

            print("PDE Loss at Epoch: ", epoch + 1, loss.item())
            if log_writer:
            #    log_writer.add_histogram('First EF Layer Grads', model.in_x[0].weight.grad.view(-1, 1), epoch)
            #    log_writer.add_histogram('First EC Layer Grads', model.in_t[0].weight.grad.view(-1, 1), epoch)
                save_checkpoint(model, optimizer, modelPath, epoch_l)

    if log_writer:
        hParams = {'numLayers': numLayers,
                   'numFeatures': numFeatures,
                   'ResidualPoints': numBatches * batchSizePDE,
                   'alpha':args.alpha,
                   'ELoss': activateEnergyLoss,
                   'args': args}

        valLoss0 = valLoss(model, 0, coordinateSystem)
        valLoss250 = valLoss(model, 250, coordinateSystem)
        valLoss500 = valLoss(model, 500, coordinateSystem)

        metric = {'hparam/SimLoss': loss.item(),
                  'hparam/valLoss0': valLoss0,
                  'hparam/valLoss250': valLoss250,
                  'hparam/valLoss500': valLoss500}

        log_writer.add_hparams(hParams, metric)
