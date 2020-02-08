from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py
from pyDOE import lhs

"""

#create dataset, that gives you access to data

with this dataset you will be able to get a

- implementing the torch.utils.data.Dataset interface  
- holds the data in the class for less I0 usage 
- whole frame with positions and labels (for plotting)
- the dataset enables batch processing for xf and x0 (cpu and gpu frame)
- holds the grid for one position

"""

class SchrodingerEquationDataset(Dataset):

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

    def __init__(self, pData, cSystem, energySamplingX, energySamplingY, initSize, numBatches, batchSizePDE, shuffle=True, useGPU=True, do_lhs=False):

        # Load data for t0
        self.lb = np.array([cSystem["x_lb"], cSystem["y_lb"], 0.])
        self.ub = np.array([cSystem["x_ub"], cSystem["y_ub"], cSystem["nt"] * cSystem["dt"]])

        self.batchSizePDE = batchSizePDE
        self.initSize = initSize
        self.numBatches = numBatches

        Exact_u, Exact_v = self.loadFrame(pData, 0)
        Exact_u = Exact_u.reshape(cSystem["nx"], cSystem["ny"]).T.reshape(-1)
        Exact_v = Exact_v.reshape(cSystem["nx"], cSystem["ny"]).T.reshape(-1)

        self.x0, self.y0 = self.get2DGrid(cSystem["nx"], cSystem["ny"])
        self.randomState = np.random.RandomState(seed=1234)
        nf = self.batchSizePDE * self.numBatches
        # Domain bounds
        
        if do_lhs :
            X_f = self.lb + (self.ub - self.lb) * lhs(3,nf)
            self.xf = X_f[:,0]
            self.yf = X_f[:,1]
            self.tf = X_f[:,2]
        else:
            self.xf, self.yf, self.tf = self.get3DGrid(cSystem["nx"], cSystem["ny"], cSystem["nt"])

        if (useGPU):
            self.dtype = torch.cuda.FloatTensor
            self.dtype2 = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype2 = torch.LongTensor

        if shuffle:
            # this function shuffles the whole dataset
            print('[Warning] You should not shuffle')
    
            # generate random permutation idx

            randIdxInit = self.randomState.permutation(Exact_u.shape[0])

            # use random index
            self.x0 = self.x0[randIdxInit]
            self.y0 = self.y0[randIdxInit]
            Exact_u = Exact_u[randIdxInit]
            Exact_v = Exact_v[randIdxInit]

            randIdxPDE = self.randomState.permutation(self.xf.shape[0])
            self.xf = self.xf[randIdxPDE]
            self.yf = self.yf[randIdxPDE]
            self.tf = self.tf[randIdxPDE]

        # sclice the array for training
        self.Exact_u = self.dtype(Exact_u)
        self.Exact_v = self.dtype(Exact_v)

        # build static set (we dont need random sampling..)
        idx_x = np.random.choice(self.x0.shape[0], initSize, replace=False)
#        idx_x = np.arange(0,self.x0.shape[0])
#        initSize = len(idx_x)

        self.fbx0 = self.dtype(self.x0[idx_x])
        self.fby0 = self.dtype(self.y0[idx_x])

        self.fbt0 = self.dtype(np.zeros(initSize))
        self.fbu0 = self.Exact_u[idx_x]
        self.fbv0 = self.Exact_v[idx_x]

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
