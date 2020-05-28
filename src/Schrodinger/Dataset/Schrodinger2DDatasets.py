from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py
from pyDOE import lhs
import src.Schrodinger.Dataset.SchrodingerAnalytical as SchrodingerAnalytical


# Calculates Schrodinger 2D at t=0 and its derivatives
class Schrodinger2DDerivatives():    

    def f0(self, z):
        return 1 / (np.pi ** (1 / 4)) * np.exp(-(((z) ** 2) / 2))

    def f00(self, z):
        return 1 / (np.pi ** (1 / 4)) * np.multiply(-(z), np.exp(-(((z) ** 2) / 2)))

    def f000(self, z):
        return 1 / (np.pi ** (1 / 4)) * (-np.exp(-(((z) ** 2) / 2)) + np.multiply((z) ** 2, np.exp(-(((z) ** 2) / 2))))

    def f1(self, z):
        return np.multiply(self.f0(z), (2 * z) / (np.pi ** (1 / 2)))

    def f11(self, z):
        return np.multiply(self.f00(z), (2 * z) / (np.pi ** (1 / 2))) + np.multiply(self.f0(z), 2 / (np.pi ** (1 / 2)))

    def f111(self, z):
        t = 1 / (np.pi ** (1 / 4)) * (-np.exp(-(((z) ** 2) / 2)) + np.multiply(((z) ** 2), np.exp(-(((z) ** 2) / 2))))
        a = np.multiply(t, (2 * z) / np.pi ** (1 / 2))
        b = a + (1 / (np.pi ** (1 / 4)) * np.multiply(-(z), np.exp(-(((z) ** 2) / 2)))) * 2 / (np.pi ** (1 / 2))
        c = b + (1 / (np.pi ** (1 / 4)) * np.multiply(-(z), np.exp(-(((z) ** 2) / 2)))) * 2 / (np.pi ** (1 / 2))
        return c

    # first x derivative
    def dx(self, x, y):
        return 1 / (2 ** (1 / 2)) * np.multiply(self.f0(y), self.f00(x)) + np.multiply(self.f1(y), self.f11(x))
    
    # second x derivative
    def dxx(self, x, y):
        return 1 / (2 ** (1 / 2)) * np.multiply(self.f0(y), self.f000(x)) + np.multiply(self.f1(y), self.f111(x))
    
    # first y derivative
    def dy(self, x, y):
        return 1 / (2 ** (1 / 2)) * np.multiply(self.f00(y), self.f0(x)) + np.multiply(self.f11(y), self.f1(x))
    
    # second y derivative
    def dyy(self, x, y):
        return 1 / (2 ** (1 / 2)) * np.multiply(self.f000(y), self.f0(x)) + np.multiply(self.f111(y), self.f1(x))
    
    # Schrodinger 2D at t=0
    def base(self, x, y):
        return 1 / (2 ** (1 / 2)) * np.multiply(self.f0(y), self.f0(x)) + np.multiply(self.f1(y), self.f1(x))


class SchrodingerEquationDataset(Dataset):

    @staticmethod
    def get2DGrid(nx, ny):
        """
        Create a vector with all postions of a 2D grid (nx X ny )
        """
        x = np.arange(0, nx, 1)
        y = np.arange(0, ny, 1)

        xGrid, yGrid = np.meshgrid(x, y)

        posX = xGrid.reshape(-1)
        posY = yGrid.reshape(-1)

        return posX, posY

    @staticmethod
    def get3DGrid(nx, ny, nt):
        """
        Create a vector with all postions of a 3D grid (nx X ny X nt)
        """
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
        """
        get the input for a specifiy point t 
        this function returns a list of grid points appended with time t
        """
        posX, posY = SchrodingerEquationDataset.get2DGrid(csystem["nx"],csystem["ny"])
        size = posX.shape[0]
        posT = np.zeros(size) + t
        posX, posY, posT = SchrodingerEquationDataset.pixelToCoordinate(posX, posY,posT,csystem)
        return posX, posY, posT
    
    @staticmethod
    def pixelToCoordinate(x, y, t, csystem):
        """
        Helper function for swapping between pixel and real coordinate system
        """
        dx = (csystem["x_ub"] - csystem["x_lb"]) / csystem["nx"]
        dy = (csystem["y_ub"] - csystem["y_lb"]) / csystem["ny"]
        disX = x * dx + csystem["x_lb"]
        disY = y * dy + csystem["y_lb"]
        disT = t * csystem["dt"]
        return disX, disY, disT
        
    @staticmethod
    def getFrame(t_pixel, csystem, omega = 1):
        posX, posY, posT = SchrodingerEquationDataset.getInput(t_pixel, csystem)
        Exact = SchrodingerAnalytical.Psi(posX, posY, posT, f=omega)
        return Exact.real, Exact.imag
                                  

    def __init__(self, cSystem, energySamplingX, energySamplingY, initSize, numBatches, batchSizePDE, useGPU=True, randomICpoints = True):
        self.lb = np.array([cSystem["x_lb"], cSystem["y_lb"], 0.])
        self.ub = np.array([cSystem["x_ub"], cSystem["y_ub"], cSystem["nt"] * cSystem["dt"]])

        self.batchSizePDE = batchSizePDE
        self.initSize = initSize
        self.numBatches = numBatches

        self.x0, self.y0 = self.get2DGrid(cSystem["nx"], cSystem["ny"])
        self.randomState = np.random.RandomState(seed=1234)
        nf = self.batchSizePDE * self.numBatches

        # Domain bounds
        self.xf, self.yf, self.tf = self.get3DGrid(cSystem["nx"], cSystem["ny"], cSystem["nt"])

        if (useGPU):
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor

        ## build static set
        idx_x = np.arange(initSize)
        if(randomICpoints):
            idx_x = np.random.choice(self.x0.shape[0], initSize, replace=False)
        self.fbx0 = self.dtype(self.x0[idx_x])
        self.fby0 = self.dtype(self.y0[idx_x])
        self.fbt0 = self.dtype(np.zeros(initSize))

        # generate energie grid
        idxX = np.arange(0, energySamplingX)
        idxY = np.arange(0, energySamplingY)

        h = (cSystem["x_ub"] - cSystem["x_lb"]) / energySamplingX
        k = (cSystem["y_ub"] - cSystem["y_lb"]) / energySamplingY

        x = cSystem["x_lb"]+ idxX * h
        y = cSystem["y_lb"]+ idxY * k

        X, Y = np.meshgrid(x, y)
        self.xe = self.dtype(X.reshape(-1))
        self.ye = self.dtype(Y.reshape(-1))

        #convert grids in physical coordinate systen
        self.xf, self.yf, self.tf = self.pixelToCoordinate(self.xf, self.yf, self.tf, cSystem)
        self.fbx0, self.fby0, self.fbt0 = self.pixelToCoordinate(self.fbx0, self.fby0, self.fbt0, cSystem)
        
        Exact = SchrodingerAnalytical.Psi(self.fbx0.cpu().numpy(), self.fby0.cpu().numpy(), self.fbt0.cpu().numpy(), f=1)
        self.fbu0 = self.dtype(Exact.real)
        self.fbv0 = self.dtype(Exact.imag)
      
        
    def __getitem__(self, index):
        # generate batch for inital solution

        xf = self.dtype(self.xf[index * self.batchSizePDE: (index + 1) * self.batchSizePDE])
        yf = self.dtype(self.yf[index * self.batchSizePDE: (index + 1) * self.batchSizePDE])
        tf = self.dtype(self.tf[index * self.batchSizePDE: (index + 1) * self.batchSizePDE])

        randT = self.randomState.uniform() * self.ub[2]
        te = (torch.zeros(self.xe.shape[0]) + randT).cuda()

        return self.fbx0, self.fby0, self.fbt0, self.fbu0, self.fbv0, xf, yf, tf, self.xe, self.ye, te

    def getFullBatch(self):
        return self.fbx0, self.fby0, self.fbt0, self.fbu0, self.fbv0

    def __len__(self):
        return self.numBatches


class Schrodinger2DTimeSliceDataset(Dataset):

    @staticmethod
    def get2DGrid(nx, ny, step = 1):
        x = np.arange(0, nx, step)
        y = np.arange(0, ny, step)

        xGrid, yGrid = np.meshgrid(x, y)

        posX = xGrid.reshape(-1)
        posY = yGrid.reshape(-1)

        return posX, posY


    @staticmethod
    def get3DGrid(nx, ny, nt, step = 1):
        x = np.arange(0, nx, step)
        y = np.arange(0, ny, step)
        t = np.arange(0, nt, step)

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


    def __init__(self, cSystem, numBatches, useGPU=True, tmax = 8):

        # Load data for t0
        self.lb = np.array([cSystem["x_lb"], cSystem["y_lb"], 0.])
        self.ub = np.array([cSystem["x_ub"], cSystem["y_ub"], tmax])
        print(self.ub)
        self.numBatches = numBatches

        Exact_u, Exact_v = self.loadFrame(pData, 0)
        Exact_u = Exact_u.reshape(cSystem["nx"], cSystem["ny"]).T.reshape(-1)
        Exact_v = Exact_v.reshape(cSystem["nx"], cSystem["ny"]).T.reshape(-1)

        self.x0, self.y0 = self.get2DGrid(cSystem["nx"], cSystem["ny"])
        self.dExactX = 200
        self.dExactY = 200
        
        if (useGPU):
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor

        # convert into pytorch datatypes
        self.Exact_u = self.dtype(Exact_u)
        self.Exact_v = self.dtype(Exact_v)

        
        self.fbx0 = self.dtype(self.x0)
        self.fby0 = self.dtype(self.y0)
        self.fbt0 = self.dtype(np.zeros(self.y0.shape[0]))
        
        self.fbu0 = self.Exact_u
        self.fbv0 = self.Exact_v
        
        self.tmax = float(tmax)
        self.cSystem = cSystem
        #convert grids in physical coordinate systen
        self.fbx0, self.fby0, self.fbt0 = self.pixelToCoordinate(self.fbx0, self.fby0, self.fbt0, cSystem)

        self.fbt0 = self.dtype(self.fbt0)

    def __getitem__(self, index):
        # generate batch for inital solution
        tBatch = self.tmax * (float(index+1) / float(self.numBatches))
        tBatch = self.cSystem["dt"] * float(index) * (float(self.cSystem["nt"]) / float(self.numBatches))  #* self.tmax * (float(index) / float(self.numBatches)) #- self.tmax / 2
        xt = self.dtype(self.fbx0)
        yt = self.dtype(self.fby0)
        t = (tBatch * (torch.ones(self.y0.shape[0])))

        return xt, yt, t, self.fbu0, self.fbv0

    def getFullBatch(self):
        return self.fbx0, self.fby0, self.fbt0, self.fbu0, self.fbv0

    def __len__(self):
        return self.numBatches


class Schrodinger2DLargeTimeSliceDataset(Schrodinger2DTimeSliceDataset):
    def __getitem__(self, index):
        # generate batch for inital solution
        x0 = self.dtype(self.fbx0)
        y0 = self.dtype(self.fby0)
        t0 = self.fbt0
        tPDEl = self.tmax * (float(index) / float(self.numBatches))
        tP_1 = self.fbt0 + (tPDEl)
        tP_2 = self.fbt0 + ((tPDEl + 0.01*torch.randn(1).cuda()))# * (torch.ones(self.y0.shape[0])))
        tP_3 = self.fbt0 + ((tPDEl + 0.01*torch.randn(1).cuda()))# * (torch.ones(self.y0.shape[0])))
        #tP_4 = self.fbt0 + ((tPDEl + 0.01*torch.randn(1).cuda()))# * (torch.ones(self.y0.shape[0])))
        #tP_5 = self.fbt0 + ((tPDEl + 0.01*torch.randn(1).cuda()))# * (torch.ones(self.y0.shape[0])))
        xf = torch.cat([x0 ,x0, x0])#, x0, x0])
        yf = torch.cat([y0 ,y0, y0])#, y0, y0])
        tf = torch.cat([tP_1, tP_2, tP_3])#, tP_4, tP_5])

        return x0, y0, t0, self.fbu0, self.fbv0, xf, yf, tf
