import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
import cmocean
import cmocean.cm as cmo

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')
plt.rc('axes', labelsize = 16, linewidth = 1)
plt.rc('font', size = 16)
plt.rc('legend', fontsize = 16)              
plt.rc('xtick', labelsize = 16)             
plt.rc('ytick', labelsize = 16)

data = loadmat('../data/cylinder_nektar_wake.mat')
u = data['U_star'][:, 0]
v = data['U_star'][:, 1]
p = data['p_star']

x = data['X_star'][:, 0]
y = data['X_star'][:, 1]
t = data['t']

X = np.concatenate((u, v), axis = 0)
U, s, vh = np.linalg.svd(X, full_matrices=False)

nt = 71
a = vh[1, :nt]
t = t[:nt]

u = u.reshape((-1, 100, 200))
v = v.reshape((-1, 100, 200))
p = p.reshape((-1, 100, 200))

x = x.reshape((-1, 100))
y = y.reshape((-1, 100))

u = u[:, :, :nt]
v = v[:, :, :nt]
p = p[:, :, :nt]

np.random.seed(24)
c = 10.0 
c = 0.0
u_noise = u + np.random.normal(0, c, np.shape(u)) * u / 100
v_noise = v + np.random.normal(0, c, np.shape(v)) * v / 100
u_noise = np.stack([u_noise, v_noise])


data_pinn = np.load(f'../results/cylinder_PINN_{c}.npz')
u_pinn = data_pinn['up']
u_pinn[2] = u_pinn[2] - u_pinn[2].mean() + p.mean()
l=10
n=53
e_pinn = np.abs(u_pinn[2, :, :, n] - p[:, :, n])
import numpy.linalg as LA


fig, ax = plt.subplots(2, 2, figsize=(6, 4.5),sharex=True,sharey=True)
# plt.set_cmap('cmo.tarn_r')
cmap = 'cmo.tarn_r'

ax[0,0].contourf(x, y, u_pinn[2, :, :, n], cmap = cmap, levels = l, vmin = -0.5, vmax = 0.05)
c0 = ax[0,1].contourf(x, y, p[:, :, n],    cmap = cmap, levels = l, vmin = -0.5, vmax = 0.05)

l = 12
c1 = ax[1,0].contourf(x, y, e_pinn,   cmap = cmap, levels = l)

for axx in ax.flatten():
    axx.set_aspect('equal')
    axx.set_xticks([1, 4.5, 8])

cax = fig.add_axes([0.95,0.57,0.01,0.26])
cb0 = plt.colorbar(c0,cax=cax,format = "%.2f",shrink = 0.9, pad = 0.25, aspect = 20)
cb0.ax.locator_params(nbins = 5)

cax2 = fig.add_axes([0.15,-0.02,0.3,0.01])
cb2 = plt.colorbar(c1,cax=cax2,format = "%.2f", 
                        orientation='horizontal',
                        shrink = 0.9, 
                        pad = 0.25, aspect = 20)
cb2.set_ticks([0.2*e_pinn.max(), 0.5*e_pinn.max(), 0.8*e_pinn.max()])

cb2.ax.locator_params(nbins=3)

for axx in ax[1,:].flatten():
    axx.set_xlabel('$x$')
    
for axx in ax.flatten()[1:3]:
    axx.set_yticklabels([])

ax[0,0].set_ylabel('$y$')
ax[1,0].set_ylabel('$y$')
ax[0,1].set_title('Reference')
ax[0,0].set_title('PINNs')
ax[1,0].set_title('$\\varepsilon = | p - \\tilde{p} |$', fontsize = 16)


plt.savefig('clean_pressure.pdf', bbox_inches = 'tight', dpi = 500)
plt.savefig('clean_pressure.jpg', bbox_inches = 'tight', dpi = 500)


