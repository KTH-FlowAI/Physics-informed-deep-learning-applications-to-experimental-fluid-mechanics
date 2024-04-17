#%%
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs
from nnSolver import nnSolver

from time import time
"""
Training NN without Physcial Infomation  

2D RANS for inflow data profile 

Without scaling 

Consider X direction derivatives    

We make a further downsampling which only use 4 points for interpolation

"""
#%%
import argparse
parser = argparse.ArgumentParser(description='PINN training')
parser.add_argument('--cp', default= 50, type=int, help='Number of grid point')
parser.add_argument('--nl', default= 4 , type=int, help='Number of layer')
parser.add_argument('--nn', default= 40 , type=int, help='Number of neuron')
parser.add_argument('--epoch', default=1000, type=int, help='Training Epoch')
parser.add_argument('--sw', default=10, type=int, help='Weight of supervise learning loss')
parser.add_argument('--uw', default=1, type=int, help='Weight of unsupervise learning loss')
parser.add_argument('--f', default=3, type=int, help='Sample Frequency of the reference data')
args = parser.parse_args()
SampleFreq = args.f
# %%
file_name ='01_data/inflow.dat'
ds = np.genfromtxt(file_name,skip_header=1)

y   = ds[:,0]
x   = np.ones(shape=y.shape) * 3

u   = ds[:,1] ;uv = -ds[:,2]; uu = ds[:,3]
vv  = ds[:,4] ;ww = ds[:,5]; 
gt  = np.array([u,uu,vv,uv]).T

print(f"The min U = {u.min()}, the max U = {u.max()}")
print(f"The min vv = {vv.min()}, the max vv = {vv.max()}")
print(f"The min uv = {uv.min()}, the max uv = {uv.max()}")
#%%
yall    = u.shape[0]
indx    = np.arange(0,yall,SampleFreq)
print(f"The index will be {indx}")
x       = x[indx]; y = y[indx]
u       = u[indx]; uv = uv[indx]; uu = uu[indx]; vv = vv[indx]
#%%
name = [
        'u','v','w',
        'uu','vv','uv']
for i,n in enumerate(name):
    print(f"The MIN and MAX value for {n} is {ds[:,i].min()}, {ds[:,i].max()}\n")
#%%
#%%
np.random.seed(24)
lb  = np.array([x.min(),y.min()])
ub  = np.array([x.max(),y.max()])
ncp = args.cp
cp  = lb + (ub-lb) * lhs(2, ncp)
print(cp.shape)

#%%

ic = np.array([
                x.flatten(),y.flatten(),
                u.flatten(), 
                uu.flatten(),vv.flatten(),uv.flatten(),
                 ]).T
print(ic.shape)
print(f"Collection point = {ic.flatten().shape} ")

#%%

nl = args.nl
nn = args.nn
epoch = args.epoch
s_w = args.sw
u_w = args.uw
solv = nnSolver(nn=nn,nl=nl,epoch=epoch,
              s_w=s_w,u_w=u_w)


case_name = f"SR_NoPI_cp{ncp}_nl{nl}_nn{nn}_epoch{epoch}_{s_w}S_{u_w}U_{SampleFreq}Sample"
print(f"INFO: Solver has been setup, case name is {case_name}")

hist, comp_time = solv.fit(ic=ic,cp=cp)
print(f"INFO: Training end, time cost: {np.round(comp_time,2)}s")
y           = ds[:,0]
x           = np.ones(shape=y.shape) * 3
cp          = np.array([ x.flatten(),y.flatten()]).T
up,error    = solv.pred(cp=cp,gt=gt)
print(f"The prediction error are {np.round(error,3)}%")
hist        = np.array(hist)
#%%
cp          = ds[:,0]
y_spine     = np.zeros(shape=cp.shape[0]*SampleFreq)
y_spine[::SampleFreq] = cp
# y_spine[1:-1:SampleFreq] = (cp[0:-1] + cp[1:])/2
y_spine[-1] = cp[-1]
x_spine     = np.ones(shape=y_spine.shape)*3
cp_spine    = np.array([x_spine.flatten(), y_spine.flatten()]).T
print(cp_spine.shape)
up_sp = solv.pred(cp_spine,gt=gt,return_error=False)
np.savez_compressed("02_pred/"+ case_name + ".npz", up = up,
                                                    up_sp=up_sp, # interpolated prediction
                                                    hist = hist,
                                                    comp_time = comp_time)
solv.model.save("03_model/"+case_name +".h5")

# %%
