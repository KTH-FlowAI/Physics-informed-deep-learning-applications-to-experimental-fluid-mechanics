#%%
import numpy as np
from matplotlib import pyplot as plt 
from pyDOE import lhs
from scipy.interpolate import interp1d
import argparse
plt.rc("font",family = "serif")
plt.rc("font",size = 20)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 16)

parser = argparse.ArgumentParser(description='PINN training')
parser.add_argument('--cp', default= 2000, type=int, help='Number of grid point')
parser.add_argument('--nl', default= 4 , type=int, help='Number of layer')
parser.add_argument('--nn', default= 40 , type=int, help='Number of neuron')
parser.add_argument('--epoch', default=1000, type=int, help='Training Epoch')
parser.add_argument('--f', default=3, type=int, help='Sampling frequency')
parser.add_argument('--sw', default=10, type=int, help='Weight of supervise learning loss')
parser.add_argument('--uw', default=1, type=int, help='Weight of unsupervise learning loss')
args = parser.parse_args()

def l2_error(p,g):
    error = np.linalg.norm((p-g))/np.linalg.norm(g)
    error = np.mean(error) * 100
    return np.round(error,4)


file_name ='01_data/inflow.dat'
d = np.genfromtxt(file_name,skip_header=1)
print(d.shape)
y = d[0:,0]
U = d[0:,1]
mu = 1.8e-5 #m^2/s
U_inf = U.max() # m/s
rho = 1.225 #kg/m^3
nu = mu /rho
U_normalized = U / U_inf
# Integrating
theta = np.trapz( (1-U_normalized)*U_normalized, y)
print(f"The momentum thickness {theta}")
Re_theta = U_inf * theta / nu
print(f"The Re_theta is {Re_theta}")
# Computing Friction Reynolds number 
dudy = np.gradient(U,y)
tau = mu * dudy 
# Wall shear stress
tauw = tau[0]
u_tau = np.sqrt( tauw / rho)
delta = y.max()
Re_tau = u_tau * delta / nu
print(f"The wall shear stress {u_tau}")
print(f"The Re_tau is {Re_tau}")

ds = d
y = ds[0:,0]
u = U ;uv = -ds[0:,2]; uu = ds[0:,3]
vv = ds[0:,4] ;ww = ds[0:,5]; 
dp = [u,uu, vv, uv]
# %%
from scipy.interpolate import interp1d
fdir    = "02_pred/"
cp= args.cp
nl = args.nl
nn = args.nn
epoch = args.epoch
s_w = args.sw
u_w = args.uw
SampleFreq = args.f
ind = 0
casename = f"SR_cp{cp}_nl{nl}_nn{nn}_epoch{epoch}_{s_w}S_{u_w}U_{SampleFreq}Sample"
# casename = f"SR_Hyper_nl{nl}_nn{nn}_epoch{epoch}_{s_w}S_{u_w}U"
ud = np.load(fdir+casename+'.npz')
up = ud["up"]
print(up.shape)
print(f"Computation time is {ud['comp_time']}")
# u,uu, vv, uv, v,p 
upp = up[:,[0,1,2,3]]

u = up[:,0]
uu = up[:,1]
vv = up[:,2]
uv = up[:,3]
cp = y
cp_inter = np.zeros(shape=cp.shape[0]*SampleFreq)
cp_inter[::SampleFreq] = cp
# cp_inter[1:-1:2] = (cp[0:-1] + cp[1:])/2 
cp_inter[-1] = cp[-1]
plt.plot(cp_inter[::4],"o")
plt.plot(cp,"x")
# plt.savefig("test_interp")
#%%
yall = u.shape[0]

masks_ref = np.zeros(shape=u.shape,dtype=bool)
indx0= np.arange(ind,yall-ind,SampleFreq)

masks_ref[indx0] = 1
print(f"The index will be {indx0}")
# x = x[indx]
yd0     =       y[masks_ref]
ud0     =       u[masks_ref]
uvd0    =       uv[masks_ref]
uud0    =       uu[masks_ref]
vvd0    =       vv[masks_ref]
dp0     =       [ud0,uud0,vvd0,uvd0]

masks_exclude = np.ones(shape=u.shape,dtype=bool)
masks_exclude[indx0] = 0
print(f"The exclude index will be {masks_exclude}")
# x = x[indx]
yd1     =       y[masks_exclude]
ud1     =       u[masks_exclude]
uvd1    =       uv[masks_exclude]
uud1    =       uu[masks_exclude]
vvd1    =       vv[masks_exclude]
dp1     =       [ud1,uud1,vvd1,uvd1]
#%%
# print( np.linalg.norm(upp[:,-1]-dp[-1])/np.linalg.norm(dp[-1]) *100)

#%%
fig, axs = plt.subplots(1,4,figsize= (24,5),sharey=True)

names = [[r"$U$"+" "+"[m/s]",
          r"$\overline{u^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
          r"$\overline{v^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
          r"$\overline{uv}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]" ],
          [r"$V (m/s)$","P (pa)"]]

plt.subplots_adjust(wspace=0.35)

for j in range(4):
        print(f"Plotting {names[0][j]}")
        g_ref = dp[j][masks_ref]
        e_ref = l2_error(p = dp0[j],g=g_ref) 
        g_new = dp[j][masks_exclude]
        e_new = l2_error(p = dp1[j],g= g_new) 
        
        g_ref = dp[j][masks_ref]
        print(f"yd0={yd0.shape}, g_ref = {g_ref.shape}")
        if yd0.shape[0]>=4:
                f = interp1d(yd0,g_ref,"cubic")
        else:
               f = interp1d(yd0,g_ref,'quadratic')
        spine_interp = f(cp)
        g_interp = spine_interp[masks_exclude]
        
        e_all = l2_error(p = upp[:,j],g= dp[j])
        
        e_spine = l2_error(p = spine_interp ,g= dp[j]) 

        
        axs[j].plot(dp0[j],yd0,"ro",markersize = 10)
        axs[j].plot(dp1[j],yd1,"g^",markersize = 10)
        axs[j].plot(g_interp,yd1,"d",c="orange",markersize=10)
        axs[j].plot(dp[j],cp, "-b",lw = 3)
        
        print(  r" ${\epsilon}_{ref}$ ="+ f"{e_ref}%\n"+\
                r" ${\epsilon}_{new}$ ="+ f"{e_new}%\n"+\
                r" ${\epsilon}_{spine}$ ="+ f"{e_spine}% \n"+\
                r" ${\epsilon}_{all}$ ="+ f"{e_all}%")
        axs[j].set_xlabel(names[0][j],fontsize=22)
axs[0].set_ylabel("y [m]",fontsize=22)

plt.savefig("04_fig/"+casename+".pdf",dpi=1000,bbox_inches="tight")
plt.savefig("04_fig/"+casename+".jpg",dpi=1000,bbox_inches="tight")
#%%
# from scipy.interpolate import make_interp_spline
# fig, axs = plt.subplots(1,4,figsize= (24,5),sharey=True)

# for j in range(4):
       
#         print(f"Plotting {names[0][j]}")
#         g_ref = dp[j][::2]
#         f = interp1d(yd0,g_ref,"quadratic")
#         spine_interp = f(cp)
#         g_interp = spine_interp[1::2]
        
#         g_new = dp[j][1::2]

#         e_new = l2_error(p = dp1[j],g= g_new) 
#         e_spine = l2_error(p = g_interp,g= g_new) 
        
#         axs[j].plot(dp1[j],yd1,"g^",markersize = 10)
#         axs[j].plot(g_new,yd1,"d",c="orange",markersize=10)
#         axs[j].plot(dp[j],cp, "-b",lw = 3)
                
#         print(r" ${\epsilon}_{new}$ ="+ f"{e_new}%\n"+r" ${\epsilon}_{spine}$ ="+ f"{e_spine}%")
#         axs[j].set_xlabel(names[0][j],fontsize=22)
# axs[0].set_ylabel("y [m]",fontsize=22)



#%%
# import tensorflow as tf
# from tensorflow import keras
# from keras.models import load_model

# model = load_model("03_model/"+casename+".h5")
# print(model.summary())
# #%%
# from scipy.interpolate import interp1d
# ile_name ='01_data/inflow.dat'
# ds = np.genfromtxt(file_name,skip_header=1)
# y = ds[0:,0]
# u = U ;uv = -ds[0:,2]; uu = ds[0:,3]
# vv = ds[0:,4] ;ww = ds[0:,5]; 

# dp = np.array([u,uu, vv, uv])

# y0 = cp 
# y0more = np.linspace(y0.min(),y0.max(),26)
# f = interp1d(y0,np.array(dp),"quadratic")
# dpe = f(y0more)
# dpe = dpe.T
# #%%
# xmore = np.ones(26)*3
# coordmore = np.array([xmore,y0more]).T
# upmore = model.predict(coordmore)
# print(upmore.shape)
# uppmore = upmore[:,[0,1,2,3]]

#%%
# fig, axs = plt.subplots(1,4,figsize= (24,5),sharey=True)
# # names = [[r"$\overline{U}$"+" "+r"$(m/s)$",r"$\overline{uu}$"+" "+r"$(m/s)^2$",r"$\overline{vv}$"+" "+r"$(m/s)^2$",r"$\overline{uv}$"+" "+r"$(m/s)^2$" ],[r"$V (m/s)$","P (pa)"]]
# plt.subplots_adjust(wspace=0.35)

# for j in range(4):
#         e = l2_error(p = uppmore[:,j],g=dpe[:,j]) 
#         axs[j].plot(uppmore[:,j],y0more[:],"ro",markersize = 10)
#         axs[j].plot(dpe[:,j],y0more,"-bo",lw = 3)
#         print(r"${\epsilon}$ ="+ f"{e}%")
#         axs[j].set_xlabel(names[0][j],fontsize=22)

# # [ax.set_xlabel(r"$y (m)$") for ax in axs[:]]
# axs[0].set_ylabel("y [m]",fontsize=22)
# plt.savefig("04_fig/"+"Interp_"+casename+".pdf",dpi=1000,bbox_inches="tight")
# plt.savefig("04_fig/"+"Interp_"+casename+".jpg",dpi=1000,bbox_inches="tight")
#%%
fig, axs = plt.subplots(1,4,figsize= (24,5),sharey=True)
# names = [[r"$\overline{U}$"+" "+r"$(m/s)$",r"$\overline{uu}$"+" "+r"$(m/s)^2$",r"$\overline{vv}$"+" "+r"$(m/s)^2$",r"$\overline{uv}$"+" "+r"$(m/s)^2$" ],[r"$V (m/s)$","P (pa)"]]
plt.subplots_adjust(wspace=0.35)

for j in range(4):
        print(dp[j].shape)        
        axs[j].plot(dp0[j],yd0,"og",markersize = 12)
        axs[j].plot(dp1[j],yd1,"xr",markersize = 12)
        axs[j].set_xlabel(names[0][j],fontsize=22)

axs[0].set_ylabel("y [m]",fontsize=22)
plt.savefig("04_fig/"+"Downsampe"+".pdf",dpi=1000,bbox_inches="tight")
plt.savefig("04_fig/"+"Downsampe"+".jpg",dpi=1000,bbox_inches="tight")

#%%
fig, axs = plt.subplots(1,4,figsize= (24,5),sharey=True)

plt.subplots_adjust(wspace=0.35)

for j in range(4):
        print(dp[j].shape)        

        axs[j].set_xlabel(names[0][j])
        axs[j].plot(dp[j],cp, "-ok",lw = 3)
        axs[j].set_xlabel(names[0][j],fontsize=22)

axs[0].set_ylabel("y [m]",fontsize=22)
plt.savefig("04_fig/"+"Profile"+".pdf",bbox_inches="tight")
plt.savefig("04_fig/"+"Profile"+".jpg",bbox_inches="tight")

# %%
np.random.seed(24)
x = np.array([3])
lb = np.array([x.min(),y.min()])
ub = np.array([x.max(),y.max()])
ncp = 50
cp = lb + (ub-lb) * lhs(2, ncp)
print(cp.shape)
#%%
plt.plot(cp[:,1],"o")
plt.hlines(y=y.max(),xmin=0, xmax=50,colors="k",lw=3)
plt.hlines(y=y.min(),xmin=0, xmax=50,colors="k",lw=3)
plt.yticks([y.min(),y.max()])
plt.ylabel("y [m]")
plt.xlabel("Number of points")
plt.savefig("04_fig/"+"Random Sample"+".jpg",dpi=1000,bbox_inches="tight")
plt.savefig("04_fig/"+"Random Sample"+".pdf",dpi=1000,bbox_inches="tight")
# %%
