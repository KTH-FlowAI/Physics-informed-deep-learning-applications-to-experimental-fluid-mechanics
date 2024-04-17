"""
A class for vanlia NN for PINN task  as comparison 
"""
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs

import  tensorflow as tf
from    tensorflow       import keras
from    keras.optimizers import Adam
from    keras.losses     import MeanSquaredError
from    tensorflow.keras import models, layers, optimizers, activations
from    ScipyOP import optimizer as SciOP

from PINNs import PINNs

from time import time
class nnSolver:
    def __init__(self, nn, nl, 
                 epoch = 1000, act = "tanh",lr = 1e-3,
                 s_w= 10, u_w = 1):
        self.epoch = epoch
        self.lr = lr
        
        self.s_w = s_w 
        self.u_w = u_w
        inp = layers.Input(shape = (2,))
        hl = inp
        for _ in range(nl):
            hl = layers.Dense(nn,
                      kernel_initializer='he_normal', 
                      activation = act)(hl)
            out = layers.Dense(4,
                   kernel_initializer='he_normal',
                   )(hl)

        self.model = models.Model(inp, out)
        print(f"The model has been bulited")
        print(self.model.summary())
    
    def fit(self,ic,cp):
        
        self.model.compile(optimizer = Adam(learning_rate= self.lr),
                           loss      = MeanSquaredError() )
        print(f"INFO: Compile Success")       
        
        st_time = time()
        hist = self.model.fit(x             = ic[:,:2], 
                              y             = ic[:,2:], 
                              batch_size    = len(ic[:,:2]),
                              epochs        =  self.epoch)
        en_time = time()

        comp_time = en_time - st_time
        print(f"INFO: Training end, time = {comp_time:.5f}")

        return hist, comp_time

    def pred(self,cp,gt,return_error = True):
        up = self.model.predict(cp)
        up_c = up[:,[0,1,2,3]]
        print(up_c.shape)
        print(gt.shape)

        if return_error:
            error = self.l2_error(up_c,gt)
            
            return up,error
        else:
            return up 


    def l2_error(self,p,g):
       
        error = np.linalg.norm((p-g),axis=0)/np.linalg.norm(g,axis=0)
        error = error * 100
        return np.round(error,4)


        