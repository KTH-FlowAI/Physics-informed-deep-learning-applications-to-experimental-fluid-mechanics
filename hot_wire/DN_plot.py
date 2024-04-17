#%%
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs
import argparse
from time import time

plt.rc("font",family = "serif")
plt.rc('text',usetex=True)
plt.rc("font",size = 22)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 16)
font_dict = {"fontsize":25,'weight':'bold'}

# plt.rc("xlabel",labelsize = 16)
# plt.rc("ylabel",labelsize = 16)\
parser = argparse.ArgumentParser(description='PINN training')
parser.add_argument('--cp', default= 2000, type=int, help='Number of grid point')
parser.add_argument('--nl', default= 4 , type=int, help='Number of layer')
parser.add_argument('--nn', default= 40 , type=int, help='Number of neuron')
parser.add_argument('--epoch', default=1000, type=int, help='Training Epoch')
parser.add_argument('--noise', default=1, type=int, help='Noise level in %')
parser.add_argument('--sw', default=10, type=int, help='Weight of supervise learning loss')
parser.add_argument('--uw', default=1, type=int, help='Weight of unsupervise learning loss')
args = parser.parse_args()
"""
Training PINNs 

3D RANS + continouity 
Add pressure gradient and viscous term
"""
# %%
def avg_data_inx(data):
    dd1 = data[:,:,3:]
    dy = data[:,0,1]
    ddx = np.mean(dd1,axis=0)
    print(ddx.shape)
    return ddx, dy

def l2_error(p,g):
    error = np.linalg.norm((p-g),axis=0)/np.linalg.norm(g,axis=0)
    # error = np.linalg.norm((p-g))/np.linalg.norm(g)
    # error = np.mean(error) * 100
    return np.round(error*100,4)


file_name ='01_data/inflow.dat'
d = np.genfromtxt(file_name,skip_header=1)
print(d.shape)
y = d[0:,0]
U = d[0:,1]

u_max = U.max()
ds = d
utau =  0.48
y = ds[0:,0]

u = U ;uv = -ds[0:,2]; uu = ds[0:,3]
vv = ds[0:,4] ;ww = ds[0:,5]; 

# dp = [u/utau, uu/utau**2, vv/utau**2, uv/utau**2]
dp = [u, uu, vv, uv]
# %%
fdir = "02_pred/"
# fdir = "02_Tuning_pred/"
cp= args.cp
nl = args.nl
nn = args.nn
epoch = args.epoch
s_w = args.sw
u_w = args.uw
noise = args.noise
# casename = f"DN_noise{noise}_cp{cp}_nl{nl}_nn{nn}_epoch{epoch}_S{s_w}_U{u_w}"
casename = f"DN_3sigma_noise{noise}_cp{cp}_nl{nl}_nn{nn}_epoch{epoch}_S{s_w}_U{u_w}"
ud = np.load(fdir+casename+'.npz')
up = ud["up"]
gn = ud['gn'] # Noisy reference
print(up.shape)
print(gn.shape)
# u, vv, uv, v,p 
upp = up[:,[0,1,2,3]]
# Scale back to the original value 
upp[:,0] *= u_max
gn[:,0]  *= u_max
upn = up[:,[4,5]]
ups = [upp,upn]


#####################
# Least-Square Method for De-Noising
# Reference: DOI: 10.17485/ijst/2016/v9i33/99598
# Link: https://sciresol.s3.us-east-2.amazonaws.com/IJST/Articles/2016/Issue-33/Article19.pdf
##################### 
def l_mat(n):
    """
    Generate L matrix

    Args: 
        n: (int) the dimension of L 
    Returns: 
        L: (NumpyArray) A matrix of ones and off-diagonal of -1 
    """
    L = np.matrix(np.eye(n-1,n))
    for il in range(0,n-1):
        L[il,il+1] = -1 
    return L 

l2_dn   = np.empty_like(gn)
N       = len(y)
L       = l_mat(N)
if args.noise==5:
    lam     = 1
elif args.noise>5:
    lam     = 1.5 
else:
    lam     = 0.5

rhs     = np.eye(N,N) + lam * np.dot(L.T,L)
# print(np.dot(L.T,L))
for i in range(gn.shape[-1]):
    # rhs = np.diag(gn[:,i])
    # l2_dn[:,i] = gn[:,i]@np.linalg.lstsq(rhs, dp[i],rcond=0)[0]
    l2_dn[:,i] = np.linalg.lstsq(rhs, gn[:,i],rcond=0)[0]


#####################
# Assessment
####################
ec      = l2_error(p = upp[:],g=np.array(dp).T) 
en      = l2_error(p = gn[:],g=np.array(dp).T) 
elstq   = l2_error(p = l2_dn, g= np.array(dp).T)
print(r"${\epsilon}_{PINN}$ ="+ f"{ec}%\n"+r"${\epsilon}_{noise}$ ="+ f"{en}%\n")
print(r"${\epsilon}_{Lstsq}$ ="+ f"{elstq}%\n"+r"${\epsilon}_{noise}$ ="+ f"{en}%\n")
print(r"${\epsilon}_{PINN}$ ="+ f"{ec.mean()}%\n"+r"${\epsilon}_{noise}$ ="+ f"{en.mean()}%\n")
print(r"${\epsilon}_{Lstsq}$ ="+ f"{elstq.mean()}%\n"+r"${\epsilon}_{noise}$ ="+ f"{en.mean()}%\n")


#####################
# Plot Figure 
####################
names = [[r"$U$"+" "+"[m/s]",
        r"$\overline{u^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{v^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{uv}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]" ],
        [r"$V (m/s)$","P (pa)"]]

fig, axs = plt.subplots(1,4,figsize= (24,5),sharey=True)
plt.subplots_adjust(wspace=0.15)
for j in range(4):

        axs[j].plot(dp[j],y,"-ko",lw = 3)
        axs[j].plot(gn[:,j],y,"-go",lw = 3)
        
        axs[j].plot(upp[:,j],y,"-or",lw = 2)
        axs[j].plot(l2_dn[:,j],y,"^-b",lw = 2)
        
        axs[j].set_xlabel(names[0][j],fontdict=font_dict)

# [ax.set_ylabel(r"$y (m)$") for ax in axs]
axs[0].set_ylabel(r"$y$"+" [m]",fontdict=font_dict)
plt.savefig(f"04_fig/"+f"DN_Noise_{args.noise}"+".pdf",dpi=1000,bbox_inches="tight")
plt.savefig(f"04_fig/"+f"DN_Noise_{args.noise}"+".jpg",dpi=1000,bbox_inches="tight")
# %%
