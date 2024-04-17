"""
De-Noising task using 3-sigma criterion to add gaussian noise 
2024.01.04
@yuningw 
"""

import numpy as np 
from pyDOE import lhs
from Solver import Solver
from time import time
import argparse
import matplotlib.pyplot as plt 
# Manual random seed 
np.random.seed(24)
utau = 0.48 # in the paper

plt.rc("font",family = "serif")
plt.rc("font",size = 20)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 16)
names = [r"$U$"+" "+"[m/s]",
        r"$\overline{u^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{v^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{uv}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]" ]
parser = argparse.ArgumentParser(description='PINN training')
parser.add_argument('--cp', default= 2000, type=int, help='Number of grid point')
parser.add_argument('--nl', default= 4 , type=int, help='Number of layer')
parser.add_argument('--nn', default= 40 , type=int, help='Number of neuron')
parser.add_argument('--epoch', default=1000, type=int, help='Training Epoch')
parser.add_argument('--noise', default=1, type=int, help='Noise level in %')
parser.add_argument('--sw', default=10, type=int, help='Weight of supervise learning loss')
parser.add_argument('--uw', default=1, type=int, help='Weight of unsupervise learning loss')
args = parser.parse_args()

file_name ='01_data/inflow.dat'
ds = np.genfromtxt(file_name,skip_header=1)
# ds = ds / ds_max
ds_max = ds.max(1)
y = ds[:,0]
x = np.ones(shape=y.shape) * 3
u = ds[:,1] ;uv = -ds[0:,2]; uu = ds[0:,3]
vv = ds[0:,4] ;ww = ds[0:,5]; 
npts = y.shape[0]
u_max = u.max()
# gt = np.array([u/utau,uu/utau**2,vv/utau**2,uv/utau**2]).T

gt = np.array([u/u_max,uu,vv,uv]).T

# Add random noise by the 3-sigma criterion
# Denote mu = mean, sigma = std, sigma = noise/100 * U/3 
# Note that the value here are scaled as inner regin
noise_level = args.noise / 100 
mu          = 0
for il in range(npts): 
    for jl in range(1,5):
        if jl == 1:
            
            norm_val = ds[il,jl]/utau 
            # norm_val = ds[il,jl]/utau 
        else: 
            norm_val = ds[il,jl]/utau**2
        
        sigma = noise_level * (norm_val)/3
        noise = np.random.normal(loc=mu,scale=np.abs(sigma))
        print(f"At Sigma = {sigma:.3f}\t noise = {noise:.3f}")
        ds[il,jl] += noise

u = ds[:,1]/u_max   ;uv = -ds[0:,2]; uu = ds[0:,3]
vv = ds[0:,4] ;ww = ds[0:,5] ; 

gn = np.array([u,uu,vv,uv]).T

## Visualisation
fig, axs = plt.subplots(1,4, figsize=(24,5),sharey=True)
for i in range(4): 
    axs[i].plot(gt[:,i],y,"-ok")
    axs[i].plot(gn[:,i],y,"-^r")
    axs[i].set_xlabel(names[i])
axs[0].set_ylabel("y [m]")
plt.savefig('04_fig/' + f"3_sigma_noise={args.noise}%.jpg",bbox_inches='tight')
# quit()

name = [
        'u','v','w',
        'uu','vv','uv']
for i,n in enumerate(name):
    print(f"The MIN and MAX value for {n} is {ds[:,i].min()}, {ds[:,i].max()}\n")
#%%
lb = np.array([x.min(),y.min()])
ub = np.array([x.max(),y.max()])
ncp = args.cp
cp = lb + (ub-lb) * lhs(2, ncp)
print(cp.shape)

#%%
ic = np.array([
                x.flatten(),
                y.flatten(),
                u.flatten(),  
                uu.flatten(),
                vv.flatten(),
                uv.flatten(),
                ]).T
print(ic.shape)
print(f"Collection point = {ic.flatten().shape} ")
#%%
nl = args.nl
nn = args.nn
epoch = args.epoch
s_w = args.sw
u_w = args.uw
solv = Solver(nn=nn,nl=nl,epoch=epoch,
            s_w=s_w,u_w=u_w)

case_name = f"DN_3sigma_noise{args.noise}_cp{ncp}_nl{nl}_nn{nn}_epoch{epoch}_S{s_w}_U{u_w}"
print(f"INFO: Solver has been setup, case name is {case_name}")

hist, comp_time = solv.fit(ic=ic,cp=cp)
print(f"INFO: Training end, time cost: {np.round(comp_time,2)}s")
y = ds[:,0]
x = np.ones(shape=y.shape) * 3
cp = np.array([ x.flatten(),y.flatten()]).T
up,error = solv.pred(cp=cp,gt=gt)
print(f"The prediction error are {np.round(error,3)}%")
hist = np.array(hist)
#%%
hist = np.array(hist)
np.savez_compressed("02_pred/"+ case_name + ".npz", up = up,
                                                    gn = gn, # Noise reference
                                                    hist = hist,
                                                    comp_time = comp_time)
solv.model.save("03_model/"+case_name +".h5")