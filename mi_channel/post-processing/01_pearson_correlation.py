"""
Print the cross correlation of the prediction 
"""
import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from scipy.signal import correlate2d
from scipy.stats import pearsonr
import argparse
argparser   = argparse.ArgumentParser()
argparser.add_argument("--t",default=5,type=int, help="The number of points sampled from Time")
argparser.add_argument("--s",default=16,type=int, help="The number of points sampled from Space")
argparser.add_argument("--c",default=0,type=float, help="Level of gaussian noise")
args        = argparser.parse_args()


def l2_norm_error(p,g): 
    """
    Compute the l2 norm error 
    """
    import numpy.linalg as LA
    error = (LA.norm(p-g,axis=(0,1)))/LA.norm(g,axis =(0,1))
    return error.mean() * 100
"""
Visualisation of the mini-channel flow output
"""
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)

ref = np.load('../data/min_channel_sr.npz')

x = ref['x'] 
y = ref['y']
z = ref['z']
u = ref['u'] # dimensions  = (nz, ny, nx, nt)
v = ref['v']
w = ref['w']
t = ref['t']

nz,ny,nx,nt = u.shape
u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)
dt = 0.2
nv = 4
nz, ny, nx = xx.shape

if args.c == 0.0:
    pred1 = np.load(f'../results/res_pinn_t{args.t}_s{args.s}.npz')
else:
    pred1 = np.load(f'../results/res_pinn_t{args.t}_s{args.s}_c{args.c:.1f}.npz')

u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]


uy = np.gradient(u[0], y, axis = 1, edge_order=2)
uz = np.gradient(u[0], z, axis = 0, edge_order=1)

vx = np.gradient(u[1], x, axis = 2, edge_order=2)
vz = np.gradient(u[1], z, axis = 0, edge_order=1)

wx = np.gradient(u[2], x, axis = 2, edge_order=2)
wy = np.gradient(u[2], y, axis = 1, edge_order=2)

omega_z = vx - uy

uy_pred1 = np.gradient(u_pred1[0], y, axis = 1, edge_order=2)
uz_pred1 = np.gradient(u_pred1[0], z, axis = 0, edge_order=1)

vx_pred1 = np.gradient(u_pred1[1], x, axis = 2, edge_order=2)
vz_pred1 = np.gradient(u_pred1[1], z, axis = 0, edge_order=1)

wx_pred1 = np.gradient(u_pred1[2], x, axis = 2, edge_order=2)
wy_pred1 = np.gradient(u_pred1[2], y, axis = 1, edge_order=2)

omega_z_pred1 = vx_pred1 - uy_pred1

uy_pred2 = np.gradient(u_pred2[0], y, axis = 1, edge_order=2)
uz_pred2 = np.gradient(u_pred2[0], z, axis = 0, edge_order=1)

vx_pred2 = np.gradient(u_pred2[1], x, axis = 2, edge_order=2)
vz_pred2 = np.gradient(u_pred2[1], z, axis = 0, edge_order=1)

wx_pred2 = np.gradient(u_pred2[2], x, axis = 2, edge_order=2)
wy_pred2 = np.gradient(u_pred2[2], y, axis = 1, edge_order=2)

omega_z_pred2 = vx_pred2 - uy_pred2

x = xx[0]
y = yy[0]

omega_z = omega_z[0]
omega_z_pred1 = omega_z_pred1[0]
omega_z_pred2 = omega_z_pred2[0]


Names = ["R_U","R_V","R_W"]
for i, name in enumerate(Names):

    r_u_pred1,_ = pearsonr(u_pred1[i].flatten(), u[i].flatten())
    print(f"\nCase t{args.t}-s{args.s}:")
    print(f"{name}:\t{r_u_pred1:.4f}")
    # print(u_pred1[i].shape)
    eu          = l2_norm_error(u_pred1[i],u[i])
    print(f"Error: {eu}")
    r_u_pred2,_ = pearsonr(u_pred2[i].flatten(), u[i].flatten())
    print(f"Case t3-s8:")
    print(f"{name}:\t{r_u_pred2:.4f}")

r_omega, _ = pearsonr(omega_z_pred1.flatten(), omega_z.flatten())
print(f"\nCase t{args.t}-s{args.s}:")
print(f"Omega:\t{r_u_pred1:.4f}")



if args.c !=0 : 
    noise_level = args.c 
    print(f"Examine The noise")
    n_u         = np.empty_like(u)

    for i, name in enumerate(Names):

        n_u[i]     = u[i] + np.random.normal(0,noise_level,np.shape(u[i])) * u[i] / 100 
        sr_n, _ = pearsonr(n_u[i].flatten(), u[i].flatten())
        e_n     = l2_norm_error(n_u[i], u[i])

        print(f"AS noise level = {noise_level}, {name} = {sr_n:.4f}")
        print(f"AS noise level = {noise_level}, Error = {e_n:.4f}") 
    # For omega z:
    
    x = ref['x'] 
    y = ref['y']
    
    nuy = np.gradient(n_u[0], y, axis = 1, edge_order=2)

    nvx = np.gradient(n_u[1], x, axis = 2, edge_order=2)
    omega_z_noise = nvx - nuy
    
    print(omega_z_noise.shape, omega_z.shape)
    omega_n,_ = pearsonr(omega_z_noise[0].flatten(),omega_z.flatten())
    
    print(f"AS noise level = {noise_level}, R_Omega = {omega_n:.4f}")