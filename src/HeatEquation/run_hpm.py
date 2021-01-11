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
    parser.add_argument("--t_pde", dest="t_pde", type=float, default=1e-6)
    parser.add_argument("--frameStep", dest="frameStep", type=int, default=25)
    parser.add_argument("--nt", dest="nt", type=int, default=3000)
    parser.add_argument("--lr", dest="lr", type=float, default=9e-5)
    args = parser.parse_args()
    
    if hvd.rank() == 0:
        
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        print("-" * 10 + args.identifier + "-" * 10)
        print("-" * 10 + "-" * len(args.identifier) + "-" * 10)
        
    # set constants for training
    log_freq = 1
    useGPU = True
    activation = torch.tanh
    activation_alpha = F.leaky_relu
    activation_hs = F.leaky_relu
    
    # create modelpath
    if hvd.rank() == 0:
        modelPath = 'results/models/' + args.identifier + '/'
        pathlib.Path(modelPath).mkdir(parents=True, exist_ok=True)
  
    # create dataset to initialize model
    ds = HeatEquationHPMDataset(args.pData, args.batchSize, args.numBatches, frameStep = args.frameStep, nt = args.nt)
    # create model
    model = HeatEquationHPMNet(args.numLayers, args.numFeatures, args.numLayers_alpha, args.numFeatures_alpha, args.numLayers_hs, args.numFeatures_hs, ds.lb, ds.ub, 5, 5, activation = activation, activation_alpha = activation_alpha).cuda()
    # delete the dataset in order to create new one containing derivatives
    del ds
    # initialize weights and biases
    if hvd.rank() == 0:
        wandb.init(project="thermal_hpm")
        wandb.run.name = args.identifier + '_hpm'
        wandb.config.update(args, allow_val_change=True)
        wandb.watch(model)
    # load pretrained model if it is given
    if(args.pModel != ""):
        load_checkpoint(model, args.pModel)
    # create dataset with derivatives   
    dds = dHeatEquationHPMDataset(model, args.pData, args.batchSize, args.numBatches, frameStep = args.frameStep, nt = args.nt)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dds, num_replicas=1, rank=0)
    train_loader = torch.utils.data.DataLoader(dds, batch_size=1, sampler=train_sampler)
    # optimize parameters of 2 networks: alpha and heat_source
    optParams = list(model.lin_layers_alpha.parameters()) + list(model.lin_layers_hs.parameters())
    optimizer = optim.Adam(optParams, lr=args.lr)
    
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), backward_passes_per_step=1)
    
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    epoch = -1

    print("Starting training ..")
    
    h_loss = 1
    epoch = 0
    start_time = time.time()
    while (h_loss > args.t_pde):
        epoch += 1
        for x, y, t, u, u_x, u_y, u_xx, u_yy, exact_u_t in train_loader:

            optimizer.zero_grad()
            # calculate loss
            hpmLoss = model.hpm_loss4(x, y, u_xx, u_yy, exact_u_t)
            hpmLoss.backward()
            optimizer.step()

        h_loss = hpmLoss.item()

        print("[%d] PDE loss: %.4e [%.2fs] saved" % (epoch, h_loss, time.time() - start_time))

        if (epoch % log_freq == 0 and hvd.rank() == 0):
            	"""
                x = x.view(-1)
                y = y.view(-1)

                u_xx = u_xx.view(-1)
                u_yy = u_yy.view(-1)

                exact_u_t = exact_u_t.view(-1)

                X = torch.stack([x,y], 1)         
                hs = (model.forward_hs(X)).view(-1) #heat source net 
                alpha = (model.forward_alpha(X)).view(-1) #alpha net

                pred_u_t = alpha*(u_xx+u_yy) + hs

                data = [[x, y, z] for (x, y, z) in zip(exact_u_t.cpu().detach().numpy().reshape(-1), pred_u_t.cpu().detach().numpy().reshape(-1), alpha.cpu().detach().numpy().reshape(-1))]
                table = wandb.Table(data=data, columns = ["exact u_t", "predicted u_t", "alpha"])

                wandb.log({"u_t exact vs u_t predicted": wandb.plot.scatter(table, "exact u_t", "predicted u_t"), "hpmLoss": h_loss, "epoch": epoch})
                """
		wandb.log({"hpmLoss": h_loss, "epoch": epoch})
                save_checkpoint(model, modelPath + "1_ic/", epoch)

    if hvd.rank() == 0:
        save_checkpoint(model, modelPath + "1_ic/", 0)
        print("--- converged ---")
