from torch.utils.data import Dataset
import numpy as np
import torch
import os
import h5py
import SchrodingerAnalytical as SchrodingerAnalytical


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

    
class HeatEquationHPMDataset(SchrodingerEquationDataset):
    
    @staticmethod   
    def loadFrame(pFile):

        if not os.path.exists(pFile):
            raise FileNotFoundError('Could not find file' + pFile)

        hf = h5py.File(pFile, 'r')
        value = np.array(hf['seq'][:])
        timing = np.array(hf['timing'][:])

        hf.close()
        
        timing = (timing - np.min(timing)) / 1e-5 # convert into seconds + remove offset, we just care about relative time from beginning of acquisition
        
        return value, timing
    
    def __init__(self, pData, cSystem, batchSize, numBatches = 5000, maxFrames = 500, shuffle=True, useGPU=True, subsampleFactor = 1):

        # Load data for t0
        self.lb = np.array([cSystem["x_lb"], cSystem["y_lb"], 0.])
        self.ub = np.array([cSystem["x_ub"], cSystem["y_ub"], cSystem["tmax"]])

        #for step in range(cSystem["nt"]):            
        Exact_u, timing = self.loadFrame(pData) #shape of the dataa: 307.200x 3000
        Exact_u = Exact_u[0:maxFrames,:]
        Exact_u = Exact_u[0:-1:subsampleFactor,:]
        print("Eu shape", Exact_u.shape)
        noTimesteps, dxdy = Exact_u.shape
        Exact_u = Exact_u.reshape(noTimesteps, 640, 480)
        
        x,y,t = SchrodingerEquationDataset.get3DGrid(noTimesteps, 640,480)
        
        self.u = []
        self.x = []
        self.y = []
        self.t = []
        for ti in range(noTimesteps):
            for xi in range(640):
                for yi in range(480):
                    self.u.append(Exact_u[ti, xi,yi])
                    self.x.append(xi)
                    self.y.append(yi)
                    self.t.append(timing[ti])
        
        self.u = np.array(self.u).reshape(-1)
        self.x = np.array(self.x).reshape(-1)
        self.y = np.array(self.y).reshape(-1)
        self.t = np.array(self.t).reshape(-1)
        
        self.x,self.y,_ = SchrodingerEquationDataset.pixelToCoordinate(self.x, self.y, self.t, cSystem)
        
        # sometimes we are loading less files than we specified by batchsize + numBatches 
        # => adapt numBatches to real number of batches for avoiding empty batches
        self.batchSize = batchSize
        print("batchSize: %d" % (self.batchSize)) 
        self.numSamples = min( (numBatches * batchSize, len(self.x) ) ) 
        print("self.numSamples: %d" % (self.numSamples)) 
        self.numBatches = self.numSamples // self.batchSize
        print("numBatches: %d" % (self.numBatches)) 
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
            #self.v = self.v[randIdx]


        # sclice the array for training
        self.x = self.dtype(self.x[:self.numSamples])
        self.y = self.dtype(self.y[:self.numSamples])
        self.t = self.dtype(self.t[:self.numSamples])
        self.u = self.dtype(self.u[:self.numSamples])
        
#        print("Normalizing u to -1/1")
#        self.u = (self.u - torch.mean(self.u))/3


    def __getitem__(self, index):
        # generate batch for inital solution
        x = (self.x[index * self.batchSize: (index + 1) * self.batchSize])
        y = (self.y[index * self.batchSize: (index + 1) * self.batchSize])
        t = (self.t[index * self.batchSize: (index + 1) * self.batchSize])
        u = (self.u[index * self.batchSize: (index + 1) * self.batchSize])
        return x, y, t, u 

    def __len__(self):
        return self.numBatches
