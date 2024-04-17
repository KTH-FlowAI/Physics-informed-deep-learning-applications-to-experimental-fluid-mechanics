"""
Apply the FFT on a snapshots to check the scale 
@yuningw
"""

import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from matplotlib import gridspec
from scipy.stats import pearsonr
from numpy import fft
from utils.plot import colorplate as cc
from utils import plt_rc_setup


def PSD_1D(data,Nx,Ny,Nz,nt,Lx,Ly,Lz, y,utype=0):
    import numpy as np
    Re_Tau = 202 #Direct from simulation
    Re = 5000 #Direct from simulation
    nu = 1/Re #Kinematic viscosity
    u_tau = Re_Tau*nu
    
    eta = nu / u_tau

    # yp = 30
    # yp = int(Ny/1.3)
    yp = -6
    y_loc = y[yp]
    
    print(f"At y = {y_loc}, y+ = {y_loc/eta}")
    
    # Streamwise velocity at wall
    data = data[utype] 
    data  = data - np.mean(data,-1,keepdims=True)
    # data  = data**2
    data    = data[:,yp,:,:]
    
    dkx = 2*np.pi/(Lx) * eta
    dky = 2*np.pi/(Ly) * eta
    dkz = 2*np.pi/(Lz) * eta

    x_range = np.linspace(0, Nx//2, Nx//2)
    y_range = np.linspace(0, Ny//2, Ny//2)
    z_range = np.linspace(0, Nz//2, Nz//2)

    kx =dkx *  np.append(x_range, -x_range)
    ky =dky *  np.append(y_range, -y_range)
    kz =dkz *  np.append(z_range, -z_range)

    # kkx, kkz = np.meshgrid(kxW,kz)
    kkx_norm = np.sqrt(kx**2)
    kkz_norm = np.sqrt(kz**2)
    # kkz_norm = np.sqrt(kkz**2)W
    Lambda_x = (2*np.pi/kkx_norm)
    Lambda_z = (2*np.pi/kkz_norm)
    
    spectra = np.empty(shape=(Nx,nt))
    for t in range(nt):
        u_hat = np.fft.fftn(data[:,:,t]**2)
        
        u_hat = np.fft.fftshift(u_hat)

        kx   = np.fft.fftfreq(Nx,d=Lx/Nx)
        kz   = np.fft.fftfreq(Nz,d=Lz/Nz)
        kx   = np.fft.fftshift(kx)
        kz   = np.fft.fftshift(kz)
        

        # spectra[:,t] = np.mean(np.abs(u_hat)**2,axis=0) * kx / u_tau**2
        spectra[:,t] = np.mean(np.absolute(u_hat)**2,axis=0) * kx / u_tau**2
    
    spectra = np.mean(spectra,-1)

    return spectra, Lambda_x, int(np.ceil(y_loc/eta))



plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1.5)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 14)              
plt.rc('xtick', labelsize = 18)             
plt.rc('ytick', labelsize = 18)
font_dict = {"fontsize":25,'weight':'bold'}


ref = np.load('../data/min_channel_sr.npz')

x = ref['x'] 
y = ref['y']
z = ref['z']
u = ref['u'] # dimensions  = (nz, ny, nx, nt)
v = ref['v']
w = ref['w']
t = ref['t']

# print(f"Shape of U: {u.shape}")

Lx     =  0.6 * np.pi 
Ly     =  1 
Lz     =  0.01125*np.pi

Re_Tau = 202 #Direct from simulation
Re = 5000 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu
eta = nu / u_tau

u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)

nz, ny, nx = xx.shape
print(f"Nz,Ny,Nx = {nz}, {ny}, {nx}")
nt         = len(t)




pred1 = np.load('../results/res_pinn_t3_s16.npz')
u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]

Ylabels = [  
            r"$E_{u}/(\nu \cdot u_{\tau})$",
            r"$E_{v}/(\nu \cdot u_{\tau})$",
            r"$E_{w}/(\nu \cdot u_{\tau})$",
            ]

Ylabels = [  
            r"$k_x \Phi_{uu} / u^2_{\tau}$",
            r"$k_x \Phi_{vv} / u^2_{\tau}$",
            r"$k_x \Phi_{ww} / u^2_{\tau}$",
            ]

Fname  =  ["u","v",'w']

for utype in range(3):

    sp_u, lambx,yloc= PSD_1D(u,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    y= y,
                    utype= utype,
                    )

    sp_t3s8, lambx,yloc= PSD_1D(u_pred2,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    y= y,
                    utype= utype,
                    )



    sp_t3s16, lambx,yloc= PSD_1D(u_pred1,
                    Nx = nx,
                    Ny = ny,
                    Nz = nz,
                    nt = nt,
                    Lx = Lx,
                    Ly = Ly,
                    Lz = Lz,
                    y= y,
                    utype= utype,
                    )

    fig, axs = plt.subplots(1,1,sharex=True, sharey=True, figsize=(6,4))
    print(lambx[:len(lambx)//2:-1])    
    axs.loglog( 
                lambx,
                sp_u,
                '-.',
                c= cc.black,
                lw = 3,
                label = 'Reference'
    )

    axs.loglog( 
                lambx,
                sp_t3s8,
                "-o",
                c= cc.blue,
                lw = 2,
                markersize = 7.5,
                label = r'PINN--t3--s8'
            )

    axs.loglog( 
                lambx,
                sp_t3s16,
                "-o",
                c= cc.red,
                lw = 2,
                markersize = 7.5,
                label = r'PINN--t3--s16'
                )
    
    axs.set_ylabel(Ylabels[utype],font_dict )
    axs.set_xlabel(r"$\lambda^+_{x}$", font_dict)   
    axs.legend(frameon=False, ncol = 3, loc = (0.0, 1.05), fontsize=13)
    fig.savefig( f"Figs/pre_PSD1d_{Fname[utype]}_{yloc}y+.pdf",bbox_inches='tight',dpi=200)