import numpy as np
import time
import torch
import torch.nn as nn
import torch.autograd
import h5py
import torch.optim as optim
import scipy.io
import wandb
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
import os
import sys
import pathlib
import torch.nn.functional as F

from argparse import ArgumentParser
from torch.autograd import Variable
from enum import Enum
from sklearn.linear_model import LinearRegression
from UKDDataset_segm_upd import HeatEquationHPMDataset
from HeatEquation_baseline_nohvd_segm_upd import  valLoss, save_checkpoint, load_checkpoint, HeatEquationBaseNet, check, model_snapshot, real_snapshot, temp_comp, mse


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

        self.lin_layers_hpm.append(nn.Linear(7, self.noFeatures_hpm))
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
        u, u_x, u_y, u_xx, u_yy, u_t = self.net_uv(x, y, t) # netuv for derivatives

        x = x.view(-1)
        y = y.view(-1)

        X = torch.stack([x,y,u,u_x,u_y,u_xx,u_yy], 1) # input variables
        f = u_t - self.forward_hpm(X)

        return u, f

    def hpm_loss(self, x, y, t, Ex_u):

        """
        Returns the quality HPM net
        """

        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)

        u, f_u = self.net_pde(x, y, t)

        Ex_u = Ex_u.view(-1)

        hpmLoss = torch.mean(f_u ** 2) 
        interpolLoss = torch.mean((u - Ex_u) ** 2)

        return hpmLoss, interpolLoss
        
def getDefaults(args):
    
    if not os.path.exists(args.pData):
        raise FileNotFoundError('Could not find file' + args.pData)
        
    # static parameter
    nx = 640 
    ny = 480
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1
    dt = 1
    
    # tmax is afterwards used to normalize time range
    hf = h5py.File(args.pData + str(args.nt-1) + '.h5', 'r')
    tmax = np.array(hf['timing'][0])
    hf.close()
    
    print("%d time steps <-> tmax: %.1f seconds" % (args.nt, tmax))
    
    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx":nx , "ny":ny, "dt": dt, "tmax": tmax, "nt": args.nt, "tmax": tmax}

    return coordinateSystem


if __name__ == "__main__":

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str, default="UKD_DeepHPM")
    parser.add_argument("--pData", dest="pData", type=str, default="../data/")  #"/home/h7/szch154b/2014_022_rest.mat")
    parser.add_argument("--pModel", dest="pModel", type=str, default="")  #"/home/h7/szch154b/2014_022_rest.mat")
    parser.add_argument("--numBatches", dest = "numBatches", type = int, default = 200000)
    parser.add_argument("--batchSize", dest="batchSize", type=int, default=512)
    parser.add_argument("--numlayers", dest="numLayers", type=int, default=8)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int, default=500)
    parser.add_argument("--numlayers_hpm", dest="numLayers_hpm", type=int, default=8)
    parser.add_argument("--numfeatures_hpm", dest="numFeatures_hpm", type=int, default=500)
    parser.add_argument("--t_ic", dest="t_ic", type=float, default=4e-3)
    parser.add_argument("--t_pde", dest="t_pde", type=float, default=1e-5)
    parser.add_argument("--pretraining", dest="pretraining", type=int, default=1)
    parser.add_argument("--frameStep", dest="frameStep", type=float, default=3)
    parser.add_argument("--nt", dest="nt", type=int, default=3000)
    parser.add_argument("--lr", dest="lr", type=float, default=1e-5)
    args = parser.parse_args()
    
    if hvd.rank() == 0:
        
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 + args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)

    # set constants for training
    coordinateSystem = getDefaults(args)
    
    modelPath = 'results/models/' + args.identifier + '/'
    useGPU = True
    activation = torch.tanh

    # create modelpath
    if hvd.rank() == 0:
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True)
  
    # create dataset
    ds = HeatEquationHPMDataset(args.pData, coordinateSystem, args.batchSize, args.numBatches, shuffle=True, useGPU=True, frameStep = args.frameStep, frames = [])
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)

    model = HeatEquationHPMNet(args.numLayers, args.numFeatures, args.numLayers_hpm, args.numFeatures_hpm, ds.lb, ds.ub,
                              5, 5, activation).cuda()
    
    # Weights and Biases initialization
    if hvd.rank() == 0:
        wandb.init(project="thermal_hpm")
        wandb.run.name = args.identifier
        wandb.config.update(args, allow_val_change=True)
        wandb.watch(model)

    lr = args.lr
    nframes = int(args.nt//args.frameStep)
    
    frames = range(0,args.nt,int(args.frameStep))

    optimizer = optim.Adam(model.parameters(), lr = lr)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         backward_passes_per_step=1)
    
    epoch = 0

    if(args.pModel != ""):
        load_checkpoint(model, args.pModel)
        #epoch = int(args.pModel.split('_')[5].split('.')[0])

    # broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    
    #frequency of logging output: every N epochs
    log_freq = 50
    if args.pretraining:
        """
        approximate full simulation
        """
        print("Starting pretraining ..")
        l_loss = 1
        start_time = time.time()
        while (l_loss > args.t_ic):
            epoch += 1
            for x, y, t, Ex_u in train_loader:
                optimizer.zero_grad()
                # calculate loss
                loss = model.loss_ic(x, y, t, Ex_u)
                loss.backward()
                optimizer.step()
                
            l_loss = loss.item()
 
            if (epoch % log_freq == 0):
                
                if hvd.rank() == 0:
                    
                    print("[%d] IC loss: %.4e [%.2fs]" % (epoch, l_loss, time.time() - start_time))
                    
                    # arrays with average predicted and exact temperatures for multiple frames
                    temp, temp_pr = temp_comp(model, args, ds, coordinateSystem, args.nt, int(args.frameStep))
		    
		    # mean square error
                    mse1 = mse(model, args, ds, coordinateSystem)
                    
                    # model produces snapshots at the beginning, middle and end of time span 
                    for i in [0, len(frames)//2, -1]:
                        img = model_snapshot(model, args, frames[i], ds, coordinateSystem)
                        plot = plt.imshow(img)
                        wandb.log({"frame " + str(frames[i]): plot, "epoch": epoch})
                        plt.clf()
			
		    # table with average predicted and exact temperatures for multiple frames
                    data = [[label, val1, val2] for (label, val1, val2) in zip(frames, temp, temp_pr)]
                    table = wandb.Table(data=data, columns = ["frame", "Mean temperature exact", "Mean temperature predicted"])
                    
                    wandb.log({"My table": table, "number of frames": nframes, "MSE": mse1, "train loss": l_loss, "epoch": epoch, "learning rate": lr})
             
            if (epoch % 100 == 0) and hvd.rank() == 0:
	        wandb.save(args.identifier + '_ic_' + str(epoch) + '.pt')
    
    if args.pretraining:
        save_checkpoint(model, modelPath + "0_ic/", epoch)
        if hvd.rank() == 0:
            wandb.save(args.identifier + '_ic_' + str(epoch) + '.pt')
        
    #learn non-linear operator N 
    
    # we need to significantly reduce the learning rate [default: 9e-6]
    for paramGroup in optimizer.param_groups:
        paramGroup['lr'] = 5e-7

    l_loss = 1
    epoch = 0
    start_time = time.time()
    while (l_loss > args.t_pde):
        epoch += 1
        for x, y, t, Ex_u in train_loader:
            optimizer.zero_grad()
            hpmLoss,interpolLoss = model.hpm_loss(x,
                                  y,
                                  t,
                                  Ex_u)
            loss = hpmLoss + interpolLoss
            loss.backward()
            optimizer.step()

        l_loss = loss.item()
        h_loss = hpmLoss.item()
        i_loss = interpolLoss.item()
        if (epoch % log_freq == 0):
            if hvd.rank() == 0:

                mse1 = mse(model, args, ds, coordinateSystem)
                wandb.log({"Train loss HPM": l_loss, "hpmLoss": h_loss, "interpolLoss": i_loss, "epoch": epoch, "MSE": mse1})

                print("[%d] PDE loss: %.4e %.4e [%.2fs] saved" % (epoch, h_loss, i_loss, time.time() - start_time))

                if (epoch % 100 == 0):
                    save_checkpoint(model, modelPath + "1_pde/", epoch)

    save_checkpoint(model, modelPath + "1_pde/", epoch)
    
    if hvd.rank() == 0:
        print("--- converged ---")
