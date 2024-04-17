"""
A script to plot the results from PINNS, NNs and Spline Interp
@yuningw 2024/01/03
"""

#%%
import numpy as np
from matplotlib import pyplot as plt 
from pyDOE import lhs
from scipy.interpolate import interp1d
plt.rc("font",family = "serif")
plt.rc('text',usetex=True)
plt.rc("font",size = 20)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 16)
font_dict = {"fontsize":25,'weight':'bold'}

# %%

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
cp      = 50
nl      = 4
nn      = 40 
epoch   = 1000
ind     = 0
s_w     = 10 
u_w     = 1
SampleFreq =6
casename_pinn = f"SR_cp{cp}_nl{nl}_nn{nn}_epoch{epoch}_{s_w}S_{u_w}U_{SampleFreq}Sample"
casename_mlp  = f"SR_NoPI_cp{cp}_nl{nl}_nn{nn}_epoch{epoch}_{s_w}S_{u_w}U_{SampleFreq}Sample"
ud = np.load(fdir+casename_pinn+'.npz')
up = ud["up"]
print(up.shape)
print(f"Computation time is {ud['comp_time']}")
# u,uu, vv, uv, v,p 
upp_pinn = up[:,[0,1,2,3]]
u = up[:,0]
uu = up[:,1]
vv = up[:,2]
uv = up[:,3]
cp = y
cp_inter = np.zeros(shape=cp.shape[0]*SampleFreq)
cp_inter[::SampleFreq] = cp
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
dp0_pinn=       [ud0,uud0,vvd0,uvd0]
masks_exclude = np.ones(shape=u.shape,dtype=bool)
masks_exclude[indx0] = 0
print(f"The exclude index will be {masks_exclude}")
yd1     =       y[masks_exclude]
ud1     =       u[masks_exclude]
uvd1    =       uv[masks_exclude]
uud1    =       uu[masks_exclude]
vvd1    =       vv[masks_exclude]
dp1_pinn=       [ud1,uud1,vvd1,uvd1]
#%%
# We load the results from MLP here

ud = np.load(fdir+casename_mlp+'.npz')
up = ud["up"]
print(up.shape)
print(f"Computation time is {ud['comp_time']}")
# u,uu, vv, uv, v,p 
upp_mlp = up[:,[0,1,2,3]]
u = up[:,0]
uu = up[:,1]
vv = up[:,2]
uv = up[:,3]
cp = y
cp_inter = np.zeros(shape=cp.shape[0]*SampleFreq)
cp_inter[::SampleFreq] = cp
yall = u.shape[0]
masks_ref = np.zeros(shape=u.shape,dtype=bool)
indx0= np.arange(ind,yall-ind,SampleFreq)
masks_ref[indx0] = 1
print(f"The index will be {indx0}")
yd0     =       y[masks_ref]
ud0     =       u[masks_ref]
uvd0    =       uv[masks_ref]
uud0    =       uu[masks_ref]
vvd0    =       vv[masks_ref]
dp0_mlp =       [ud0,uud0,vvd0,uvd0]
masks_exclude = np.ones(shape=u.shape,dtype=bool)
masks_exclude[indx0] = 0
print(f"The exclude index will be {masks_exclude}")
yd1     =       y[masks_exclude]
ud1     =       u[masks_exclude]
uvd1    =       uv[masks_exclude]
uud1    =       uu[masks_exclude]
vvd1    =       vv[masks_exclude]
dp1_mlp =       [ud1,uud1,vvd1,uvd1]

#%% 
# Sort out all data and markers here 
dp1List = [dp1_mlp, dp1_pinn]
dp0List = [dp0_mlp, dp0_pinn]
uppList = [upp_mlp, upp_pinn]
markers = ["^","^"]
colors  = ["g",'r']
#%%
fig, axs = plt.subplots(1,4,figsize= (24,5),sharey=True)

names = [[r"$U$"+" "+"[m/s]",
        r"$\overline{u^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{v^2}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]",
        r"$\overline{uv}$"+" "+"[m"+ r"$^2$" +"/s"+r"$^2$" +"]" ],
        [r"$V (m/s)$","P (pa)"]]
error_dict = {}
for name in names[0]:
        error_dict[name]=np.empty(shape=(len(dp1List)+1,))

modelname = ['NN',"PINN"]
plt.subplots_adjust(wspace=0.15)
for i in range(len(dp1List)):
        dp0     = dp0List[i]
        dp1     = dp1List[i]
        upp     = uppList[i]
        marker  = markers[i]
        color   = colors[i]
        print(f"\nEvaluating results from:{modelname[i]}")
        for j in range(4):
                print(f"Plotting {names[0][j]}")
                g_ref = dp[j][masks_ref]
                e_ref = l2_error(p = dp0[j],g=g_ref) 
                g_new = dp[j][masks_exclude]
                e_new = l2_error(p = dp1[j],g= g_new) 
                
                print(f"yd0={yd0.shape}, g_ref = {g_ref.shape}")
                if yd0.shape[0]>=4:
                        f = interp1d(yd0,g_ref,"cubic")
                else:
                        f = interp1d(yd0,g_ref,'quadratic')
                spine_interp = f(cp)
                g_interp = spine_interp[masks_exclude]
                e_all = l2_error(p = upp[:,j],g= dp[j])
                e_spine = l2_error(p = spine_interp ,g= dp[j])

                error_dict[names[0][j]][i]  = e_all
                error_dict[names[0][j]][-1] = e_spine

                
                axs[j].plot(dp[j],cp, "-ko",lw = 3)  # Original profile 
                axs[j].plot(dp0[j],yd0,"o",c=color,markersize = 10) # Reference Samples
                axs[j].plot(dp1[j],yd1,marker,c=color,markersize = 10) # Results from ML
                axs[j].plot(g_interp,yd1,"d",c="orange",markersize=10) # Results from Interp

                print(  r" ${\epsilon}_{ref}$ ="+ f"{e_ref}%\n"+\
                        r" ${\epsilon}_{new}$ ="+ f"{e_new}%\n"+\
                        r" ${\epsilon}_{spine}$ ="+ f"{e_spine}% \n"+\
                        r" ${\epsilon}_{all}$ ="+ f"{e_all}%")
                axs[j].set_xlabel(names[0][j],font_dict)
axs[0].set_ylabel(r"$y$" + " [m]",font_dict)
plt.savefig("04_fig/"+f"SR_All_compare_{SampleFreq}"+".pdf",dpi=1000,bbox_inches="tight")
plt.savefig("04_fig/"+f"SR_All_compare_{SampleFreq}"+".jpg",dpi=1000,bbox_inches="tight")

#%% 
# Write down the error in a txt file 
import pandas as pd 

df = pd.DataFrame(error_dict)
df.to_csv(f"sr_error_compare_{SampleFreq}.csv")

quit()


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
