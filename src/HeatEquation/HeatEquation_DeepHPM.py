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
from sklearn.linear_model import LinearRegression

from UKDDataset import HeatEquationHPMDataset
from HeatEquation_baseline_nohvd import  writeIntermediateState, valLoss, save_checkpoint, load_checkpoint, writeValidationLoss, HeatEquationBaseNet

import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F

import os
import h5py

class HeatEquationHPMNet(HeatEquationBaseNet):
    def __init__(self, numLayers, numFeatures, numLayers_hpm, numFeatures_hpm, lb, ub, samplingX, samplingY,
                 activation=torch.tanh, activation_hpm=F.relu):
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

        self.lin_layers_hpm.append(nn.Linear(8, self.noFeatures_hpm))
        for _ in range(self.noLayers_hpm):
            self.lin_layers_hpm.append(nn.Linear(self.noFeatures_hpm, self.noFeatures_hpm))
        self.lin_layers_hpm.append(nn.Linear(self.noFeatures_hpm, 1))

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
        u, u_yy, u_xx, u_t = self.net_uv(x, y, t) #netuv for derivatives

        x = x.view(-1)
        y = y.view(-1)

        X = torch.stack([u_yy, u_xx], 1) #change parameters, Temperature, coordinates 

        f = u_t - self.forward_hpm(X)

        return u, f

    def get_params(self, x, y, t): #not necessary

        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u, u_yy, u_xx, u_t = self.net_uv(x, y, t)
        # u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = self.net_uv(x, y, t)

        X = torch.stack([x, y, u, u_yy, u_xx], 1)
        # X = torch.stack([x, y, u, v, u_yy, v_yy, u_xx, v_xx], 1)

        f = self.forward_hpm(X)
        dudt = - f[:, 0]
        # dvdt = - f[:, 1]

        dudt = dudt.cpu().detach().numpy().reshape(-1, 1)
        # dvdt = dvdt.cpu().detach().numpy().reshape(-1, 1)
        # v_xx = v_xx.cpu().detach().numpy().reshape(-1, 1)
        # v_yy = v_yy.cpu().detach().numpy().reshape(-1, 1)
        u_xx = u_xx.cpu().detach().numpy().reshape(-1, 1)
        u_yy = u_yy.cpu().detach().numpy().reshape(-1, 1)
        u_t = u_t.cpu().detach().numpy().reshape(-1, 1)
        # v_t = v_t.cpu().detach().numpy().reshape(-1, 1)

        diff_u = np.linalg.norm(dudt - u_t, 2)
        # diff_v = np.linalg.norm(dvdt-v_t,2)

        reg_u = LinearRegression().fit([[u_xx[i][0], u_yy[i][0]] for i in range(len(u_xx))], dudt)  # ??
        # reg_u = LinearRegression().fit([[v_xx[i][0],v_yy[i][0]] for i in range(len(v_xx))], dudt)
        # reg_v = LinearRegression().fit([[u_xx[i][0],u_yy[i][0]] for i in range(len(u_xx))], dvdt)

        lambda_u_xx, lambda_u_yy = reg_u.coef_[0]
        # lambda_v_xx, lambda_v_yy = reg_u.coef_[0]
        # lambda_u_xx, lambda_u_yy = reg_v.coef_[0]

        return lambda_u_xx, lambda_u_yy, diff_u
        # return lambda_v_xx, lambda_v_yy, lambda_u_xx, lambda_u_yy, diff_u, diff_v

    def hpm_loss(self, x, y, t, Ex_u):
        # def hpm_loss(self, x, y, t, Ex_u, Ex_v):
        """
        Returns the quality HPM net
        """

        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u, f_u = self.net_pde(x, y, t)


        Ex_u = Ex_u.view(-1)
        # Ex_v = Ex_v.view(-1)

        hpmLoss = torch.mean(f_u ** 2) + torch.mean((u - Ex_u) ** 2)
        # hpmLoss = torch.mean(f_u ** 2) + torch.mean(f_v ** 2) + torch.mean((u - Ex_u) ** 2) + torch.mean((v - Ex_v) ** 2)
        return hpmLoss

def getDefaults():
    # static parameter
    nx = 640 
    ny = 480
    nt = 1000
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1
    dt = 1
    tmax = 1.9271e-04
    #numOfEnergySamplingPointsX = 100
    #numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx":nx , "ny":ny, "tmax": tmax, "dt": dt}

    return coordinateSystem


def loadTimesteps(pFile):

    if not os.path.exists(pFile):
        raise FileNotFoundError('Could not find file' + pFile)

    hf = h5py.File(pFile, 'r')
    timing = np.array(hf['timing'][:])

    hf.close()
    
    timing = timing - np.min(timing)
    
    return timing


if __name__ == "__main__":

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str, default="S2D_DeepHPM")
    parser.add_argument("--pData", dest="pData", type=str, default="/home/hoffma83/Code/PDElearning_Szymon/data/UKD/2014_022_rest.mat")  #"/home/h7/szch154b/2014_022_rest.mat")
    parser.add_argument("--batchsize", dest="batchsize", type=int, default=307200)
    parser.add_argument("--numbatches", dest="numBatches", type=int, default=500)
    parser.add_argument("--numlayers", dest="numLayers", type=int, default=8)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int, default=300)
    parser.add_argument("--numlayers_hpm", dest="numLayers_hpm", type=int, default=8)
    parser.add_argument("--numfeatures_hpm", dest="numFeatures_hpm", type=int, default=300)
    parser.add_argument("--t_ic", dest="t_ic", type=float, default=1e-7)
    parser.add_argument("--t_pde", dest="t_pde", type=float, default=2e-7)
    parser.add_argument("--pretraining", dest="pretraining", type=int, default=1)
    args = parser.parse_args()

    if hvd.rank() == 0:
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 + args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)

    # set constants for training
    coordinateSystem = getDefaults()
    
    maxTime = np.max(loadTimesteps(args.pData))
    coordinateSystem["tMax"] = maxTime
    print("tmax: %.2e" % (maxTime))
    # fix maximum time based on timestep of our data



    
    modelPath = 'results/models/' + args.identifier + '/'
    logdir = 'results/experiments/' + args.identifier + '/'
    useGPU = True
    activation = torch.tanh

    # create modelpath
    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True)
        # create logWriter
    log_writer = SummaryWriter(logdir) if hvd.rank() == 0 else None

    # create dataset
    ds = HeatEquationHPMDataset(args.pData, coordinateSystem, args.numBatches, args.batchsize, shuffle=False, useGPU=True)
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)

    model = HeatEquationHPMNet(args.numLayers, args.numFeatures, args.numLayers_hpm, args.numFeatures_hpm, ds.lb, ds.ub,
                              5, 5, activation).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)
    
    # broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    if args.pretraining:
        """
        approximate full simulation
        """
        print("Starting pretraining ..")
        epoch = 0
        l_loss = 1
        l_loss_0 = 10
        start_time = time.time()
        while (l_loss > args.t_ic or abs(l_loss_0 - l_loss) > args.t_ic):
            epoch += 1
            for x, y, t, Ex_u in train_loader:
                optimizer.zero_grad()
                # calculate loss
                loss = model.loss_ic(x, y, t, Ex_u)
                loss.backward()
                optimizer.step()
            l_loss_0 = l_loss
            l_loss = loss.item()

            if (epoch % 100 == 0) and log_writer:
                print("[%d] IC loss: %.4e [%.2fs]" % (epoch, l_loss, time.time() - start_time))
                log_writer.add_scalar("loss_ic", l_loss, epoch)

                writeValidationLoss(0, model, epoch, log_writer, coordinateSystem, identifier="PT")
                writeIntermediateState(0, model, epoch, log_writer, coordinateSystem, identifier="PT")
                writeValidationLoss(500, model, epoch, log_writer, coordinateSystem, identifier="PT")
                writeIntermediateState(500, model, epoch, log_writer, coordinateSystem, identifier="PT")
                writeValidationLoss(1000, model, epoch, log_writer, coordinateSystem, identifier="PT")
                writeIntermediateState(1000, model, epoch, log_writer, coordinateSystem, identifier="PT")

                save_checkpoint(model, modelPath + "0_ic/", epoch)

    if args.pretraining:
        save_checkpoint(model, modelPath + "0_ic/", epoch)
        

    """
    learn non-linear operator N 
    """
    # we need to significantly reduce the learning rate [default: 9e-6]
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = 1e-7

    l_loss = 1
    start_time = time.time()
    while (l_loss > args.t_pde):
        epoch += 1
        for x, y, t, Ex_u in train_loader:
            # for x, y, t, Ex_u, Ex_v in train_loader:
            optimizer.zero_grad()
            loss = model.hpm_loss(x,
                                  y,
                                  t,
                                  Ex_u)
            """
            loss = model.hpm_loss(x,
                                  y,
                                  t,
                                  Ex_u,
                                  Ex_v)
            """
            loss.backward()
            optimizer.step()

        l_loss = loss.item()

        if (epoch % 1000 == 0) and log_writer:
            lambda_u_xx, lambda_u_yy, diff_u = model.get_params(x, y, t)
            # lambda_v_xx, lambda_v_yy, lambda_u_xx, lambda_u_yy, diff_u, diff_v = model.get_params(x, y, t)
            # log_writer.add_scalar("lambda_v_xx", lambda_v_xx, epoch)
            # log_writer.add_scalar("lambda_v_yy", lambda_v_yy, epoch)
            log_writer.add_scalar("lambda_u_xx", lambda_u_xx, epoch)
            log_writer.add_scalar("diff_u_t", diff_u, epoch)
            # log_writer.add_scalar("diff_v_t", diff_v, epoch)
            log_writer.add_scalar("hpm_loss", l_loss, epoch)

            print("[%d] PDE loss: %.4e [%.2fs] saved" % (epoch, loss.item(), time.time() - start_time))

            writeIntermediateState(0, model, epoch, log_writer, coordinateSystem, identifier="PDE")
            writeIntermediateState(500, model, epoch, log_writer, coordinateSystem, identifier="PDE")
            writeIntermediateState(1000, model, epoch, log_writer, coordinateSystem, identifier="PDE")

            writeValidationLoss(0, model, epoch, log_writer, coordinateSystem, identifier="PDE")
            writeValidationLoss(500, model, epoch, log_writer, coordinateSystem, identifier="PDE")
            writeValidationLoss(1000, model, epoch, log_writer, coordinateSystem, identifier="PDE")

            sys.stdout.flush()

            log_writer.add_histogram('First Layer Grads', model.lin_layers_hpm[0].weight.grad.view(-1, 1), epoch)

            save_checkpoint(model, modelPath + "1_pde/", epoch)

    save_checkpoint(model, modelPath + "1_pde/", epoch)
    print("--- converged ---")
