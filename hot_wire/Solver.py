"""
A class for solving PINNs and return all out puts
"""
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, activations
from ScipyOP import optimizer as SciOP

from PINNs import PINNs

from time import time
class Solver:
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
            out = layers.Dense(6,
                   kernel_initializer='he_normal',
                   )(hl)

        self.model = models.Model(inp, out)
        print(f"The model has been bulited")
        print(self.model.summary())
    
    def fit(self,ic,cp):
        
        opt = optimizers.Adam(self.lr)
        sopt = SciOP(self.model)

        
        self.pinn = PINNs(self.model, opt, sopt, self.epoch, self.s_w, self.u_w)
        st_time = time()
        hist = self.pinn.fit(ic, cp)
        en_time = time()
        comp_time = en_time - st_time

        return hist, comp_time

    def pred(self,cp,gt,return_error = True):
        up = self.pinn.predict(cp)
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


        