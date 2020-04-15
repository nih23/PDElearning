from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py

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
        for step in range(cSystem["nt"]+1):
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
