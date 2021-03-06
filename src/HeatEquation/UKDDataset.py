from torch.utils.data import Dataset
import numpy as np
import torch
import os
from skimage.restoration import denoise_bilateral 
import h5py
import SchrodingerAnalytical as SchrodingerAnalytical
from skimage.filters import sobel
from skimage.segmentation import watershed
from scipy import ndimage as ndi


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
    def getInput(tpoint, csystem, pData):
        """
        get the input for a specifiy point t 
        this function returns a list of grid points appended with time t
        """
        hf = h5py.File(pData + str(tpoint) + '.h5', 'r')
        t = np.array(hf['timing'][0])
        hf.close()
        
        posX, posY = SchrodingerEquationDataset.get2DGrid(csystem["nx"],csystem["ny"])
        
        size = posX.shape[0]
        posT = np.zeros(size) + t
        
        #posX, posY, _ = SchrodingerEquationDataset.pixelToCoordinate(posX, posY,posT,csystem)
        posX = posX*0.25
        posY = posY*0.25
        
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
    def getFrame(timepoint, csystem, omega = 1):
        posX, posY, posT = SchrodingerEquationDataset.getInput(timepoint, csystem)
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
    def segmentation(pFile, step, threshold = 32.4):
    
        hf = h5py.File(pFile + str(step) + '.h5', 'r')
        value = np.array(hf['seq'][:])
        hf.close()

        value = np.array(value).reshape(-1)
        value = value.reshape(640,480)
        
        #bilateral filter + 32.4 threshold lead to 80 percent decrease of data actually used
        value = denoise_bilateral(value, sigma_color=5, sigma_spatial=5, multichannel=False)
        
        elevation_map = sobel(value)
        markers = np.zeros_like(value)
        markers[value > threshold] = 2
        markers[value <= threshold] = 1
        segmentation = watershed(elevation_map, markers)
        segmentation = ndi.binary_fill_holes(segmentation-1)
        segmentation = np.array(segmentation, dtype = np.int)
        
        return segmentation
    
    @staticmethod   
    def loadFrame(pFile, step):

        if not os.path.exists(pFile):
            raise FileNotFoundError('Could not find file' + pFile)

        hf = h5py.File(pFile + str(step) + '.h5', 'r')
        value = np.array(hf['seq'][:])
        
        timing = np.array(hf['timing'][:])
        hf.close()
       
        return value, timing
    
    def __init__(self, pData, batchSize, numBatches, frameStep, nt, spatialStep = 1, shuffle=True, useGPU=True):
                
        self.u = []
        self.x = []
        self.y = []
        self.t = []
        
        hf = h5py.File(pData + str(nt-1) + '.h5', 'r')
        tmax = np.array(hf['timing'][0])
        hf.close()
        
        nx = 640 
        ny = 480
        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
        dt = 1
                
        seg_matrix = self.segmentation(pData, 0)
        
        for step in range(0,nt,frameStep):
            
            Exact_u, timing = self.loadFrame(pData, step)
            Exact_u = Exact_u.reshape(nx,ny)*seg_matrix
            
            for xi in range(0,640,spatialStep):
                for yi in range(0,480,spatialStep):
                    if Exact_u[xi,yi] != 0:
                        self.u.append(Exact_u[xi,yi])
                        self.x.append(xi)
                        self.y.append(yi)
                        self.t.append(timing)          
         
        self.u = np.array(self.u).reshape(-1)
        self.x = np.array(self.x).reshape(-1)
        self.y = np.array(self.y).reshape(-1)
        self.t = np.array(self.t).reshape(-1)
                       
        # sometimes we are loading less files than we specified by batchsize + numBatches 
        # => adapt numBatches to real number of batches for avoiding empty batches
        self.batchSize = batchSize
        print("batchSize: %d" % (self.batchSize)) 
        self.numSamples = min( (numBatches * batchSize, len(self.x) ) ) 
        print("numSamples: %d" % (self.numSamples)) 
        self.numBatches = self.numSamples // self.batchSize
        print("numBatches: %d" % (self.numBatches)) 
        self.randomState = np.random.RandomState(seed=1234)
                
        self.coordinateSystem = {"x_lb": xmin, "x_ub": xmax, "y_lb": ymin, "y_ub" : ymax, "nx":nx , "ny":ny, "dt": dt, "nt": nt} #, 't_lb': 0, 't_ub': tmax} #, 'u_lb': min(self.u), 'u_ub': max(self.u)}
        
        # normalization of pixels to mm
        
        self.x = self.x*0.25
        self.y = self.y*0.25
        #self.u = (self.u - min(self.u))/(max(self.u) - min(self.u))
        
        self.lb = np.array([self.x.min(), self.y.min(), self.t.min()])
        self.ub = np.array([self.x.max(), self.y.max(), self.t.max()])

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
        
    def __getitem__(self, index):
        # generate batch for inital solution
        x = (self.x[index * self.batchSize: (index + 1) * self.batchSize])
        y = (self.y[index * self.batchSize: (index + 1) * self.batchSize])
        t = (self.t[index * self.batchSize: (index + 1) * self.batchSize])
        u = (self.u[index * self.batchSize: (index + 1) * self.batchSize])
        return x, y, t, u 

    def __len__(self):
        return self.numBatches
    
class dHeatEquationHPMDataset(HeatEquationHPMDataset):
    def __init__(self, model, pData, batchSize, numBatches, frameStep, nt, spatSize = 4, shuffle=True, useGPU=True):
        super().__init__(pData = pData, batchSize = batchSize, numBatches = numBatches, frameStep = frameStep, nt = nt, spatialStep = 1, shuffle = shuffle, useGPU = useGPU)
        
        size = self.x.size()[0]
        
        self.u_x = np.zeros(size)
        self.u_y = np.zeros(size)
        self.u_xx = np.zeros(size)
        self.u_yy = np.zeros(size)
        self.u_t = np.zeros(size)
              
        n = batchSize
        for i in range (size//n):

            _u, u_x, u_y, u_xx, u_yy, u_t = model.net_uv(self.x[(n*i):(n*(i+1))], self.y[(n*i):(n*(i+1))], self.t[(n*i):(n*(i+1))])
            
            self.u_x[(n*i):(n*(i+1))] = u_x.cpu().detach().numpy()
            self.u_y[(n*i):(n*(i+1))] = u_y.cpu().detach().numpy()
            self.u_xx[(n*i):(n*(i+1))] = u_xx.cpu().detach().numpy()
            self.u_yy[(n*i):(n*(i+1))] = u_yy.cpu().detach().numpy()
            self.u_t[(n*i):(n*(i+1))] = u_t.cpu().detach().numpy()
            
        self.x = self.x.cpu().detach().numpy()
        self.y = self.y.cpu().detach().numpy()
        self.t = self.t.cpu().detach().numpy()
        self.u = self.u.cpu().detach().numpy()
                    
        #self.dlb = np.array([self.x.min(), self.y.min(), self.t.min(), self.u.min(), self.u_x.min(), self.u_y.min(), self.u_xx.min(), self.u_yy.min()])
        #self.dub = np.array([self.x.max(), self.y.max(), self.t.max(), self.u.max(), self.u_x.max(), self.u_y.max(), self.u_xx.max(), self.u_yy.max()])
        
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
            self.u_x = self.u_x[randIdx]
            self.u_y = self.u_y[randIdx]
            self.u_t = self.u_t[randIdx]
            self.u_xx = self.u_xx[randIdx]
            self.u_yy = self.u_yy[randIdx]
            #self.v = self.v[randIdx]

        # sclice the array for training
        self.x = self.dtype(self.x[:self.numSamples])
        self.y = self.dtype(self.y[:self.numSamples])
        self.t = self.dtype(self.t[:self.numSamples])
        self.u = self.dtype(self.u[:self.numSamples])
        self.u_x = self.dtype(self.u_x[:self.numSamples])
        self.u_y = self.dtype(self.u_y[:self.numSamples])
        self.u_t = self.dtype(self.u_t[:self.numSamples])
        self.u_xx = self.dtype(self.u_xx[:self.numSamples])
        self.u_yy = self.dtype(self.u_yy[:self.numSamples])
           
    def __getitem__(self, index):

        x = (self.x[index * self.batchSize: (index + 1) * self.batchSize])
        y = (self.y[index * self.batchSize: (index + 1) * self.batchSize])
        t = (self.t[index * self.batchSize: (index + 1) * self.batchSize])
        u = (self.u[index * self.batchSize: (index + 1) * self.batchSize])
        u_x = (self.u_x[index * self.batchSize: (index + 1) * self.batchSize])
        u_y = (self.u_y[index * self.batchSize: (index + 1) * self.batchSize])
        u_xx = (self.u_xx[index * self.batchSize: (index + 1) * self.batchSize])
        u_yy = (self.u_yy[index * self.batchSize: (index + 1) * self.batchSize])
        u_t = (self.u_t[index * self.batchSize: (index + 1) * self.batchSize])     
        return x, y, t, u , u_x, u_y, u_xx, u_yy, u_t    
