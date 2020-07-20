from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py

#xmin = -10
#xmax = 10

class SchrodingerEquationDataset(Dataset):
    
    @staticmethod
    def get1DGrid(nx):
        return np.arange(0, nx, 1)

    @staticmethod
    def getInput(t, nx):
        posX = get1DGrid(nx)
        size = posX.shape[0]
        posT = np.zeros(size) + t
        return posX, posT

    @staticmethod
    def loadFrame(pFile):

        if not os.path.exists(pFile):
            raise FileNotFoundError('Could not find file' + pFile)

        hf = h5py.File(pFile, 'r')
        value = np.array(hf['seq'][:])
        t = np.array(hf['timing'][:])

        hf.close()
        return value, t

    @staticmethod
    def pixelToCoordinate(x, t, xmax, xmin, nx):
        dx = (xmax - xmin) / nx
        disX = x * dx + xmin
        return disX, t
    
    def __init__(self, filepath, xmin, xmax, numBatches, batchSize, shuffle=True, useGPU=True):

        self.batchSize= batchSize
        self.numBatches = numBatches
        self.numSamples = numBatches * batchSize

        self.u = []
        self.t = []
        self.x = []
        
        seq, t = loadFrame(filepath)
            
        nx = len(seq)
        nt = len(t)
        
        for step in range(nt):
            
            Exact_u,t = seq[step], t[step]
            posX, posT = getInput(t, len(Exact_u))
            
            [self.u.append(u) for u in Exact_u]
            [self.x.append(x) for x in posX]
            [self.t.append(t) for t in posT]
            
            
        del seq, t, Exact_u, posX, posT
        
        self.u = np.array(self.u)
        self.x = np.array(self.x)
        self.t = np.array(self.t)

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
            self.t = self.t[randIdx]
            self.u = self.u[randIdx]


        # sclice the array for training
        self.x = self.x[:self.numSamples]
        self.t = self.t[:self.numSamples]
        self.u = self.u[:self.numSamples]

        #convert grids in physical coordinate systen
        self.x, self.t = pixelToCoordinate(self.x, self.t, xmax, xmin, nx)
