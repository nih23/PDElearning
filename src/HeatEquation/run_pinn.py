import wandb
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

from UKDDataset import *
from HeatEquation_baseline_nohvd import  valLoss, save_checkpoint, load_checkpoint, HeatEquationBaseNet, check, model_snapshot, real_snapshot, temp_comp, mse
from HeatEquation_DeepHPM import *
import matplotlib.pyplot as plt
import torch.utils.data.distributed
import horovod.torch as hvd
from argparse import ArgumentParser
import os
import sys
import pathlib
import torch.nn.functional as F

import os
import h5py

if __name__ == "__main__":

    # Initialize Horovod
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())

    parser = ArgumentParser()
    parser.add_argument("--identifier", dest="identifier", type=str, default="UKD_DeepHPM")
    parser.add_argument("--pData", dest="pData", type=str, default="../data/") 
    parser.add_argument("--pModel", dest="pModel", type=str, default="") 
    parser.add_argument("--numBatches", dest = "numBatches", type = int, default = 200000)
    parser.add_argument("--batchSize", dest="batchSize", type=int, default=512)
    parser.add_argument("--numlayers", dest="numLayers", type=int, default=8)
    parser.add_argument("--numfeatures", dest="numFeatures", type=int, default=500)
    parser.add_argument("--numlayers_alpha", dest="numLayers_alpha", type=int, default=8)
    parser.add_argument("--numfeatures_alpha", dest="numFeatures_alpha", type=int, default=500)
    parser.add_argument("--numlayers_hs", dest="numLayers_hs", type=int, default=8)
    parser.add_argument("--numfeatures_hs", dest="numFeatures_hs", type=int, default=500)
    parser.add_argument("--t_ic", dest="t_ic", type=float, default=5e-4)
    parser.add_argument("--t_pde", dest="t_pde", type=float, default=1e-5)
    parser.add_argument("--frameStep", dest="frameStep", type=int, default=25)
    parser.add_argument("--nt", dest="nt", type=int, default=3000)
    parser.add_argument("--lr", dest="lr", type=float, default=9e-5)
    args = parser.parse_args()
    
    if hvd.rank() == 0:
        
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 + args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        
    # set constants for training
    log_freq = 100
    useGPU = True
    activation = torch.tanh
    activation_alpha = F.leaky_relu
    activation_hs = F.leaky_relu
    
    # create modelpath
    if hvd.rank() == 0:
        modelPath = 'results/models/' + args.identifier + '/'
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True)
  
    # create dataset
    ds = HeatEquationHPMDataset(args.pData, args.batchSize, args.numBatches, frameStep = args.frameStep, nt = args.nt)
    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(ds, batch_size=1, sampler=train_sampler)
    # create model
    model = HeatEquationHPMNet(args.numLayers, args.numFeatures, args.numLayers_alpha, args.numFeatures_alpha, args.numLayers_hs, args.numFeatures_hs, ds.lb, ds.ub, 5, 5, activation = activation, activation_alpha = activation_alpha).cuda()
    # initialize weights and biases 
    if hvd.rank() == 0:
        wandb.init(project="thermal_hpm")
        wandb.run.name = args.identifier + '_pinn'
        wandb.config.update(args, allow_val_change=True)
        wandb.watch(model)
    # load pretrained model if it is given
    if(args.pModel != ""):
        load_checkpoint(model, args.pModel)
    # list of frames that are used for the dataset
    nframes = range(0,args.nt,int(args.frameStep))
    # optimize parameters of pinn model
    optimizer = optim.Adam(model.in_t.parameters(), lr = args.lr)
    optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.in_t.named_parameters(),
                                     backward_passes_per_step=1)
                                     
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    epoch = -1

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

        if (epoch % log_freq == 0 and hvd.rank() == 0):

            print("[%d] IC loss: %.4e [%.2fs]" % (epoch, l_loss, time.time() - start_time))
	    
	    """
            temp, temp_pr = temp_comp(model, args.pData, ds, args.nt, args.frameStep)

            mse1 = mse(model, args.pData, ds)

            for i in [0, len(nframes)//2, -1]:
                img = model_snapshot(model, args.pData, nframes[i], ds)
                plot = plt.imshow(img)
                wandb.log({"frame " + str(nframes[i]): plot, "epoch": epoch})
                plt.clf()

            data = [[label, val1, val2] for (label, val1, val2) in zip(nframes, temp, temp_pr)]
            table = wandb.Table(data=data, columns = ["frame", "Mean temperature exact", "Mean temperature predicted"])

            wandb.log({"My table": table, "MSE": mse1, "train loss": l_loss, "epoch": epoch})
            """
            
            wandb.log({"train loss": l_loss, "epoch": epoch})

            save_checkpoint(model, modelPath + "0_ic/", epoch)

    if hvd.rank() == 0:

        print("[%d] IC loss: %.4e [%.2fs]" % (epoch, l_loss, time.time() - start_time))
        wandb.save(args.identifier + 'model_' + str(epoch) + '.pt')

        temp, temp_pr = temp_comp(model, args.pData, ds, args.nt, args.frameStep)

        mse1 = mse(model, args.pData, ds)

        for i in [0, len(nframes)//2, -1]:
            img = model_snapshot(model, args.pData, nframes[i], ds)
            plot = plt.imshow(img)
            wandb.log({"frame " + str(nframes[i]): plot, "epoch": epoch})
            plt.clf()

        data = [[label, val1, val2] for (label, val1, val2) in zip(nframes, temp, temp_pr)]
        table = wandb.Table(data=data, columns = ["frame", "Mean temperature exact", "Mean temperature predicted"])

        wandb.log({"My table": table, "MSE": mse1, "train loss": l_loss, "epoch": epoch})

        save_checkpoint(model, modelPath + "0_ic/", 0)
