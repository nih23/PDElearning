from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.autograd
import h5py
import torch.optim as optim
import scipy.io
from torch.autograd import Variable
import torch.optim as optim
from enum import Enum

import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F

from tensorboardX import SummaryWriter


class SchrodingerEquationDataset(Dataset):

    @staticmethod
    def get2DGrid(nx, ny):
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)

        xGrid, yGrid = np.meshgrid(x, y)

        posX = xGrid.reshape(-1)
        posY = yGrid.reshape(-1)

        return posX, posY

    @staticmethod
    def get3DGrid(nx, ny, nt):
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)
        t = np.arange(0, nt, 1)

        xGrid, yGrid, tGrid = np.meshgrid(x, y, t)

        posX = xGrid.reshape(-1)
        posY = yGrid.reshape(-1)
        posT = tGrid.reshape(-1)
        return posX, posY, posT

    @staticmethod
    def getInput(t, csystem):
        posX, posY = SchrodingerEquationDataset.get2DGrid(csystem["nx"],csystem["ny"])
        size = posX.shape[0]
        posT = np.zeros(size) + t

        posX, posY, posT = SchrodingerEquationDataset.pixelToCoordinate(posX, posY,posT,csystem)

        return posX, posY, posT
    
    @staticmethod
    def pixelToCoordinate(x, y, t, csystem):
        dx = (csystem["x_ub"] - csystem["x_lb"]) / csystem["nx"]
        dy = (csystem["y_ub"] - csystem["y_lb"]) / csystem["ny"]
        disX = x * dx + csystem["x_lb"]
        disY = y * dy + csystem["y_lb"]
        disT = t * csystem["dt"]

        return disX, disY, disT


    @staticmethod
    def loadFrame(pFile, discreteT):
        """

        :param pFile: place of the h5 files ending up with '/'
        :param discretT: discrete time position
        :return: returns real and imaginary part of the solution at discete time step discretT
        """
        # generate filename from parameters
        filePath = pFile + 'step-' + str(discreteT) + '.h5'

        if not os.path.exists(filePath):
            raise FileNotFoundError('Could not find file' + filePath)

        hf = h5py.File(filePath, 'r')
        real = np.array(hf['real'][:])
        imag = np.array(hf['imag'][:])

        hf.close()
        return real, imag

    def __init__(self, pData, cSystem, numBatches, batchSize, shuffle=True, useGPU=True):

        # Load data for t0
        self.lb = np.array([cSystem["x_lb"], cSystem["y_lb"], 0.])
        self.ub = np.array([cSystem["x_ub"], cSystem["y_ub"], cSystem["nt"] * cSystem["dt"]])

        self.batchSize= batchSize
        self.numBatches = numBatches
        self.numSamples = numBatches * batchSize

        self.u = []
        self.v = []
        self.t = []
        self.x = []
        self.y = []
        for step in range(1):
            Exact_u, Exact_v = self.loadFrame(pData, step)
            Exact_u = Exact_u.reshape(cSystem["nx"], cSystem["ny"]).T.reshape(-1)
            Exact_v = Exact_v.reshape(cSystem["nx"], cSystem["ny"]).T.reshape(-1)
            posX, posY, posT = SchrodingerEquationDataset.getInput(step, cSystem)
            self.u.append(Exact_u)
            self.v.append(Exact_v)
            self.x.append(posX)
            self.y.append(posY)
            self.t.append(posT)
            
        self.u = np.array(self.u).reshape(-1)
        self.v = np.array(self.v).reshape(-1)
        self.x = np.array(self.x).reshape(-1)
        self.y = np.array(self.y).reshape(-1)
        self.t = np.array(self.t).reshape(-1)

        self.randomState = np.random.RandomState(seed=1234)
        # Domain bounds

        if (useGPU):
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor

        if shuffle:
            # this function shuffles the whole dataset

            # generate random permutation idx

            randIdx = self.randomState.permutation(self.x.shape[0])

            # use random index
            self.x = self.x[randIdx]
            self.y = self.y[randIdx]
            self.t = self.t[randIdx]
            self.u = self.u[randIdx]
            self.v = self.v[randIdx]


        # sclice the array for training
        self.x = self.x[:self.numSamples]
        self.y = self.y[:self.numSamples]
        self.t = self.t[:self.numSamples]
        self.u = self.u[:self.numSamples]
        self.v = self.v[:self.numSamples]

        #convert grids in physical coordinate systen
        self.x, self.y, self.t = self.pixelToCoordinate(self.x, self.y, self.t, cSystem)
 

    def __getitem__(self, index):
        # generate batch for inital solution

        x = self.dtype(self.x[index * self.batchSize: (index + 1) * self.batchSize])
        y = self.dtype(self.y[index * self.batchSize: (index + 1) * self.batchSize])
        t = self.dtype(self.t[index * self.batchSize: (index + 1) * self.batchSize])
        u = self.dtype(self.u[index * self.batchSize: (index + 1) * self.batchSize])
        v = self.dtype(self.v[index * self.batchSize: (index + 1) * self.batchSize])
        return x, y, t, u, v

    def __len__(self):
        return self.numBatches


pData = './data_3/init/'

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

        f = torch.stack([u_t, v_t], 1) - self.forward_hpm(X)
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

        loss = torch.mean((u0 - u) ** 2) + torch.mean((v0 - v) ** 2)
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


def writeIntermediateState(timeStep, model, epoch, nx, ny, fileWriter,csystem, pretraining = False):
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
        if not pretraining:
            fileWriter.add_figure('Real_' + str(timeStep), fig, epoch)
        else: 
            fileWriter.add_figure('Real_pretraining_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(v, cmap='jet')
        plt.colorbar()
        if not pretraining:
            fileWriter.add_figure('Imaginary_' + str(timeStep), fig, epoch)
        else: 
            fileWriter.add_figure('Imaginary_pretraining_' + str(timeStep), fig, epoch)
        plt.close(fig)

        fig = plt.figure()
        plt.imshow(h, cmap='jet')
        plt.colorbar()
        if not pretraining:
            fileWriter.add_figure('Norm_' + str(timeStep), fig, epoch)
        else: 
            fileWriter.add_figure('Norm_pretraining_' + str(timeStep), fig, epoch)
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
    uPred = uPred.reshape([csystem['nx']*csystem['ny'],])
    vPred = vPred.reshape([csystem['nx']*csystem['ny'],])
    # load label data
    uVal, vVal = SchrodingerEquationDataset.loadFrame(pData, timeStep)
    uVal = np.array(uVal).reshape([csystem['nx']*csystem['ny'],])
    vVal = np.array(vVal).reshape([csystem['nx']*csystem['ny'],])

    valLoss = np.mean((uVal - uPred) ** 2) + np.mean((vVal - vPred) ** 2)
    return valLoss
    
def norms(model, timeStep, csystem):
    x, y, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()
    uPred = UV[:, 0]
    vPred = UV[:, 1]

    # load label data
    uPred = uPred.reshape([csystem['nx']*csystem['ny'],])
    vPred = vPred.reshape([csystem['nx']*csystem['ny'],])
    # load label data
    uVal, vVal = SchrodingerEquationDataset.loadFrame(pData, timeStep)
    uVal = np.array(uVal).reshape([csystem['nx']*csystem['ny'],])
    vVal = np.array(vVal).reshape([csystem['nx']*csystem['ny'],])

    norm2 = np.linalg.norm(uVal - uPred,2)
    norminf = np.linalg.norm(uVal - uPred,np.inf)
    
    return norm2, norminf

def writeValidationLoss(model, writer, timeStep, epoch, csystem, pretraining = False):
    if writer:
        loss = valLoss(model, timeStep,csystem)
        if not pretraining:
            writer.add_scalar("ValidationLoss_" + str(timeStep), loss, epoch)
        else: 
            writer.add_scalar("ValidationLoss_pretraining_" + str(timeStep), loss, epoch)
        
def writeNorm(model, writer, timeStep, epoch, csystem, pretraining = False):
    if writer:
        norm2, norminf = norms(model, timeStep,csystem)
        if pretraining:
            writer.add_scalar("Norm_2_pretraining_" + str(timeStep), norm2, epoch)
            writer.add_scalar("Norm_inf_pretraining_" + str(timeStep), norminf, epoch)
        else: 
            writer.add_scalar("Norm_2_" + str(timeStep), norm2, epoch)
            writer.add_scalar("Norm_inf_" + str(timeStep), norminf, epoch)



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


def runNN(identifier = "init_S2D_DeepHPM", batchsize = 10000, numBatches = 900, numLayers = 8, numFeatures = 300, numLayers_hpm = 3, t_ic = 1e-5, t_pde = 1e-5, pretraining = True, alpha = 12, lhs = 0):

    # Initialize Horovod
    hvd.init()
    numFeatures_hpm = numFeatures
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    # static parameter
    nx = 200
    ny = 200
    nt = 1000
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    dt = 0.001
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx": nx , "ny": ny, "nt": nt, "dt": dt}

    pData = './data_3/init/'

    if hvd.rank() == 0: 
        print("-" * 10 + "-" * len(identifier) + "-" * 10)
        print("-" * 10 + identifier + "-" * 10)
        print("-" * 10 + "-" * len(identifier) + "-" * 10)
    
    print("Rank",hvd.rank(),"Local Rank", hvd.local_rank())
    
    #adapter of commandline parameters

    modelPath = './models/' + str((identifier)) + '/'
    logdir = './experiments/' + str((identifier)) + '/'  

    batchSizePDE = batchsize
    useGPU = True

    #create modelpath
    if hvd.rank() == 0:
    	pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True) 
    # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None

    # create dataset
    ds = SchrodingerEquationDataset(pData, coordinateSystem,  numBatches, batchSizePDE, shuffle=True, useGPU=True)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler = train_sampler)
    activation = torch.tanh

    model = SchrodingerNet(numLayers, numFeatures, numLayers_hpm, numFeatures_hpm, ds.lb, ds.ub, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, torch.tanh).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)

    if pretraining:

      epoch = 0
      l_loss = 1
      while(epoch < 100000):
        epoch+=1
        for x, y, t, u, v in train_loader:
            optimizer.zero_grad()
                # calculate loss
            loss = model.solution_loss(x, y, t, u, v)
            loss.backward()
            optimizer.step()
        l_loss = loss.item()
        if epoch % 500 == 0:
            
            writeIntermediateState(0, model, epoch, nx, ny, log_writer,coordinateSystem, pretraining)
            writeValidationLoss(model, log_writer, 0, epoch,coordinateSystem, pretraining)
            writeNorm(model, log_writer, 0, epoch, coordinateSystem, pretraining)
            
            print("Loss at Epoch " + str(epoch) + ": " + str(loss.item()))

    """
    learn non-linear operator N 
    """
	# we need to significantly reduce the learning rate [default: 9e-6]
    return 0 
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = 1e-5
    
    l_loss = 1
    epoch = 0
    writeNorm(model, log_writer, 0, 0, coordinateSystem)

    while(l_loss > t_pde):
      epoch += 1
      
      for x, y, t, u, v in train_loader:
          optimizer.zero_grad()           
          loss = model.hpm_loss(x,
                                y,
                                t,
                                u,
                                v)
          loss.backward()
          optimizer.step()

      l_loss = loss.item()

      if epoch % 200 == 0:

          writeIntermediateState(0, model, epoch, nx, ny, log_writer,coordinateSystem)
          writeValidationLoss(model, log_writer, 0, epoch,coordinateSystem)
          writeNorm(model, log_writer, 0, epoch, coordinateSystem)

          sys.stdout.flush()

          print("PDE Loss at Epoch: ", epoch + 1, loss.item())
          if log_writer:
              log_writer.add_histogram('First Layer Grads', model.lin_layers[0].weight.grad.view(-1, 1), epoch)
              save_checkpoint(model, optimizer, modelPath, epoch)

runNN()
