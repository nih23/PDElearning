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
from UKDDataset import SchrodingerEquationDataset
import matplotlib.pyplot as plt
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
import os
import sys
import pathlib

class HeatEquationBaseNet(nn.Module):
    def __init__(self, lb, ub, samplingX, samplingY, activation=torch.tanh, noLayers = 8, noFeatures = 100, use_singlenet = True, ssim_windowSize = 9, initLayers = True, useGPU = False):
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
        super(HeatEquationBaseNet, self).__init__()
        self.activation = activation
        self.useGPU = useGPU
        self.lb = torch.Tensor(lb).float()
        self.ub = torch.Tensor(ub).float()
        if(self.useGPU):
            self.lb = self.lb.cuda()
            self.ub = self.ub.cuda()
        self.noFeatures = noFeatures
        self.noLayers = noLayers
        self.in_t = nn.ModuleList([])
        if(initLayers):
            self.init_layers()


    def init_layers(self):
        """
        This function creates the torch layers and initialize them with xavier
        :param self:
        :return:
        """
        self.in_t.append(nn.Linear(3, self.noFeatures)) # time
        for _ in range(self.noLayers):
            self.in_t.append(nn.Linear(self.noFeatures, self.noFeatures))
        self.in_t.append(nn.Linear(self.noFeatures,1))

        
        for m in self.in_t:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # scale spatiotemporral coordinate to [-1,1]
        t_in = (x - self.lb)/(self.ub - self.lb)
        # only normalize to [-1,1] in case of tanh
        #if(self.activation == torch.tanh):
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

        #torch.cuda.empty_cache()
        dim = x.shape[0] #defines the shape of the gradient

        # save input in variabeles is necessary for gradient calculation
        x = Variable(x, requires_grad=True).cuda()
        y = Variable(y, requires_grad=True).cuda()
        t = Variable(t, requires_grad=True).cuda()

        X = torch.stack([x, y, t], 1)

        UV = self.forward(X)
        u = UV[:, 0]
        grads = torch.ones([dim]).cuda()

        # huge change to the tensorflow implementation this function returns all neccessary gradients
        J_U = torch.autograd.grad(u,[x,y,t], create_graph=True, grad_outputs=grads)
        #J_V = torch.autograd.grad(v,[x,y,t], create_graph=True, grad_outputs=grads)
    
        u_x = J_U[0].reshape([dim])
        u_y = J_U[1].reshape([dim])
        u_t = J_U[2].reshape([dim])

        #v_x = J_V[0].reshape([dim])
        #v_y = J_V[1].reshape([dim])
        #v_t = J_V[2].reshape([dim])

        u_xx = torch.autograd.grad(u_x, x, create_graph=True, grad_outputs=grads)[0]
        #v_xx = torch.autograd.grad(v_x, x, create_graph=True, grad_outputs=grads)[0]

        u_yy = torch.autograd.grad(u_y, y, create_graph=True, grad_outputs=grads)[0]
        #v_yy = torch.autograd.grad(v_y, y, create_graph=True, grad_outputs=grads)[0]

        u_xx = u_xx.reshape([dim])
        #v_xx = v_xx.reshape([dim])
        u_yy = u_yy.reshape([dim])
        #v_yy = v_yy.reshape([dim])

        return u, u_yy, u_xx, u_t
        #return u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t


    def net_pde(self, x, y, t, omega=1.):
        """
        Calculates the quality of the pde estimation
        :param x postion x
        :param t time t
        :param omega frequency of the harmonic oscillator 
        """
        #get predicted solution and the gradients 
        u, u_yy, u_xx, u_t = self.net_uv(x, y, t)
        #u, v, u_yy, v_yy, u_xx, v_xx, u_t, v_t = self.net_uv(x, y, t)
        x = x.view(-1)
        y = y.view(-1)

        #calculate loss for real and imaginary part seperatly 
        # fu is the real part of the schrodinger equation
        #f_u = -1 * u_t   + omega * 0.5 *  (y ** 2) 
        f_u = -1 * u_t - 0.5 * v_xx - 0.5 * v_yy + omega* 0.5 * (x ** 2) * v + omega * 0.5 *  (y ** 2) * v
        # fv is the imaginary part of the schrodinger equation 
        #f_v =  0.5 * u_xx + 0.5 * u_yy - omega* 0.5 * (x ** 2) * u - omega * 0.5 * (y ** 2) * u
        f_v = -1 * v_t + 0.5 * u_xx + 0.5 * u_yy - omega* 0.5 * (x ** 2) * u - omega * 0.5 * (y ** 2) * u
        return u, v, f_u #, f_v

    
    def loss_ic(self, x, y, t, u0, filewriter=None, epoch = 0, w_ssim = 0):
        """
        Returns the quality of the net
        """
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        inputX = torch.stack([x, y, t], 1)
        u = self.forward(inputX).view(-1)
        u0 = u0.view(-1)
        #v0 = v0.view(-1,1)

        #loss = (torch.mean((u0 - u) ** 2) + torch.mean((v0 - v) ** 2))
        loss = (torch.mean((u0 - u)**2))

        return loss


    def loss_pde(self, x0, y0, t0, u0, xf, yf, tf, xe, ye, te, c, samplingX, samplingY,activateEnergyLoss=True, alpha=1.):
    #def loss_pde(self, x0, y0, t0, u0, v0, xf, yf, tf, xe, ye, te, c, samplingX, samplingY,activateEnergyLoss=True, alpha=1.):
        #reshape all inputs into correct shape 
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

        u, f_u = self.net_pde(inputX, inputY, inputT)
        #u, v, f_u, f_v = self.net_pde(inputX, inputY, inputT)

        solU = u[:n0]
        #solV = v[:n0]

        pdeLoss = alpha * torch.mean((solU - u0) ** 2) + \
                  torch.mean(f_u ** 2) #+ \
                  #alpha * torch.mean((solV - v0) ** 2) + \
                  
                  #torch.mean(f_v ** 2)

        if activateEnergyLoss:
            eU = u[n0 + nf:]
            #eV = v[n0 + nf:]
            eH = eU ** 2 #+ eV ** 2

            lowerX = self.lb[0]
            higherX = self.ub[0]

            lowerY = self.lb[1]
            higherY = self.ub[1]

            disX = (higherX - lowerX) / samplingX
            disY = (higherY - lowerY) / samplingY

            u0 = u0.view(-1)
            #v0 = v0.view(-1)
            integral = 0.25 * disX * disY * torch.sum(eH * self.W) #weights???
            # calculte integral over field for energy conservation
            eLoss = (integral - c) ** 2
            pdeLoss = pdeLoss + eLoss
            
        return pdeLoss


def writeIntermediateState(timeStep, model, epoch, fileWriter,csystem, identifier = "PDE"):
    """
    Functions that write intermediate solutions to tensorboard
    """
    if fileWriter is None:
        return 
       
    nx = csystem['nx']
    ny = csystem['nx']

    x, y, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()

    u = UV[:, 0].reshape((nx,ny))
    
    
    #v = UV[:, 1].reshape((nx,ny))

    #h = u ** 2 + v ** 2

    fig = plt.figure()
    plt.imshow(u, cmap='jet')
    plt.colorbar()
    fileWriter.add_figure('%s-real/t%.2f' % (identifier, t[0].cpu().numpy()), fig, epoch)
    plt.close(fig)

#    fig = plt.figure()
#    plt.imshow(v, cmap='jet')
#    plt.colorbar()
#    fileWriter.add_figure('%s-imag/t%.2f' % (identifier, t[0].cpu().numpy()), fig, epoch)
#    plt.close(fig)

#    fig = plt.figure()
#    plt.imshow(h, cmap='jet')
#    plt.colorbar()
#    fileWriter.add_figure('%s-norm/t%.2f' % (identifier, t[0].cpu().numpy()), fig, epoch)
#    plt.close(fig)
    
    


def valLoss(model, timeStep, csystem):
    x, y, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
    x = torch.Tensor(x).float().cuda()
    y = torch.Tensor(y).float().cuda()
    t = torch.Tensor(t).float().cuda()

    inputX = torch.stack([x, y, t], 1)
    UV = model.forward(inputX).detach().cpu().numpy()
    uPred = UV[:, 0].reshape(-1)
    #vPred = UV[:, 1].reshape(-1)

    # load label data
    uVal, vVal = SchrodingerEquationDataset.getFrame(timeStep,csystem)
    uVal = np.array(uVal).reshape(-1)
    #vVal = np.array(vVal).reshape(-1)

    valLoss_u = np.max(abs(uVal - uPred)) 
    #valLoss_v = np.max(abs(vVal - vPred))
    valSqLoss_u = np.sqrt(np.sum(np.power(uVal - uPred,2)))
    #valSqLoss_v	= np.sqrt(np.sum(np.power(vVal - vPred,2)))

    return valLoss_u, valSqLoss_u
    #return valLoss_u, valLoss_v, valSqLoss_u, valSqLoss_v


def writeValidationLoss(timeStep, model, epoch, writer, csystem, identifier):
    if writer is None:
        return
        
    _, _, t = SchrodingerEquationDataset.getInput(timeStep,csystem)
    t = torch.Tensor(t).float().cuda()

    valLoss_u, valSqLoss_u = valLoss(model, timeStep,csystem)
    #valLoss_u, valLoss_v, valSqLoss_u, valSqLoss_v = valLoss(model, timeStep,csystem)
    #valLoss_uv = valLoss_u + valLoss_v
    #valSqLoss_uv = valSqLoss_u + valSqLoss_v
    writer.add_scalar("inf: L_%s/u/t%.2f" % (identifier, t[0].cpu().numpy()), valLoss_u, epoch)
    #writer.add_scalar("inf: L_%s/v/t%.2f" % (identifier, t[0].cpu().numpy()), valLoss_v, epoch)
    #writer.add_scalar("inf: L_%s/uv/t%.2f" % (identifier, t[0].cpu().numpy()), valLoss_uv, epoch)
    writer.add_scalar("2nd: L_%s/u/t%.2f" % (identifier, t[0].cpu().numpy()), valSqLoss_u, epoch)
    #writer.add_scalar("2nd: L_%s/v/t%.2f" % (identifier, t[0].cpu().numpy()), valSqLoss_v, epoch)
    #writer.add_scalar("2nd: L_%s/uv/t%.2f" % (identifier, t[0].cpu().numpy()), valSqLoss_uv, epoch)
    


def save_checkpoint(model, path, epoch):
    #print(model.state_dict().keys())    
    pathlib.Path(path).mkdir(parents=True, exist_ok=True) 
    state = {
        'model': model.state_dict(),
    }
    torch.save(state, path + 'model_' + str(epoch)+'.pt')
    print("saving model to ---> %s" % (path + 'model_' + str(epoch)+'.pt'))

    
def load_checkpoint(model, path):
    device = torch.device('cpu')
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])



def getDefaults():
    # static parameter
    nx = 640
    ny = 480 
    nt = 1000
    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3
    dt = 0.001
    tmax = 1
    numOfEnergySamplingPointsX = 100
    numOfEnergySamplingPointsY = 100

    coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx":nx , "ny":ny, "nt": nt, "dt": dt}

    return coordinateSystem, numOfEnergySamplingPointsX, numOfEnergySamplingPointsY, tmax 
