"""
Demonstrate the relationship between the error and spacing between sample normalised by BL thickness 
"""
# Enviroment setup
import numpy as np
from matplotlib import pyplot as plt 
from pyDOE import lhs
from scipy.interpolate import interp1d
import pandas as pd 
plt.rc("font",family = "serif")
plt.rc('text',usetex=True)
plt.rc("font",size = 20)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 16)
plt.rc("ytick",labelsize = 16)
class cc:
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian
    gray    = "#DFE0D9" # ermuyu 
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
# U_inf = U.max() # m/s
U_inf = 9.4 # m/s reference mean flow 
U_e   = 8.38
rho = 1.225 #kg/m^3
nu = mu /rho
u_tau  = 0.48 # Given in the papaer 
d99 = np.trapz(rho*(1 - U/U_e),y)
print(f"BL Thickness is = {d99}")
d99 = y[np.where(U<U.max()*0.99)[0]][-1]
print(f"BL Thickness is = {d99}")


ds = d
y = ds[0:,0]
u = U ;uv = -ds[0:,2]; uu = ds[0:,3]
vv = ds[0:,4] ;ww = ds[0:,5]; 
dp = [u, uu, vv, uv]
SampleFreqs = [2,3,6]

fdir    = "02_pred/"
cp      = 50
nl      = 4
nn      = 40 
epoch   = 1000
ind     = 0
s_w     = 10 
u_w     = 1


varNames    = ['avg']
error_dict  = {}
for n in varNames:
    error_dict[n] = []

error_mat = []

names       =  [r"$\varepsilon_{U}$" + " (%) ",
        r"$\varepsilon_{\overline{u^2}}$" + " (%) ",
        r"$\varepsilon_{\overline{v^2}}$" + " (%) ",
        r"$\varepsilon_{\overline{uv}}$" + " (%) ",
        r"$\overline{\varepsilon}$" + r" [\%] " 
        ]
ydiff       = []

for SampleFreq in SampleFreqs:
    
    
    # Retained sample
    y_sample = y[::SampleFreq]
    diff_y_sample = np.mean(np.diff(y_sample))/d99
    
    print(f"The difference between retained y sample: {diff_y_sample}")
    ydiff.append(diff_y_sample)

    df = pd.read_csv(f"sr_error_compare_{SampleFreq}.csv")
    error_val = df.to_numpy()
    error_val = error_val[:,1:]
    error_val = np.concatenate([error_val,error_val.mean(-1).reshape(3,1)],axis=-1)
    error_mat.append(error_val)


error_mat = np.stack(error_mat)
# print(error_mat.shape)
# error_mat = np.concatenate([error_mat, 
                        # error_mat.mean(-1).reshape(3,3,1) ],axis=-1)
print(error_mat.shape)
print(error_mat[0,1,:])


methods     = ["NNs",'PINNs',"Spline Interpolation"]
colors      = ["g","r",'orange']
markers     = ["s","o","^"]

for jl, varname in enumerate(varNames):
    fig, axs = plt.subplots(1,1,figsize=(6,4))
    for il,name in enumerate(methods):
        marker = markers[il]
        color = colors[il]
        axs.plot(
                ydiff,
                error_mat[:,il, jl],
                "-"+marker,color=color, markersize=10, 
                )
    axs.set_xlabel(r"$\overline{\Delta y}" +  " / " +  "\delta_{99}$",fontsize = 20)
    axs.set_ylabel(names[jl], fontsize = 20)
    # axs.set_xticks(ydiff)``
    plt.legend(methods,fontsize = 16)
    plt.savefig('04_fig/' + f'error_Vs_distance_{varname}.pdf',dpi=300,bbox_inches='tight')