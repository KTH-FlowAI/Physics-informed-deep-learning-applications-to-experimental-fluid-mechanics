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

eta = 0.0

u = u + np.random.normal(0.0, eta, np.shape(u)) * u / 100
v = v + np.random.normal(0.0, eta, np.shape(v)) * v / 100

x = x.reshape((-1, 100))
y = y.reshape((-1, 100))

uy = np.gradient(u, y[:, 0], axis = 0, edge_order=2)
vx = np.gradient(v, x[0], axis = 1, edge_order=2)
w = vx - uy

u = u[:, :, :nt]
v = v[:, :, :nt]
p = p[:, :, :nt]

u = u[::10, ::10, ::35]
v = v[::10, ::10, ::35]
p = p[::10, ::10, ::35]

w = w[::10, ::10, ::35]

x = x[::10, ::10]
y = y[::10, ::10]

#%%

fig, ax = plt.subplots(1, 1, figsize=(12, 3))
ax0 = ax.inset_axes([0.75,0.31,0.5,0.5])
ax1 = ax.inset_axes([0.25,-0.21,0.5,0.5])
ax2 = ax.inset_axes([-0.25,0.4,0.5,0.5])

axs = [ax0, ax1, ax2]

plt.set_cmap('cmo.tarn')
ax.plot(t, a, ls = '-', c = 'tab:blue', label = '$a_{1}$', lw = 2)
ax.plot(t[::35], a[::35], 's', c = 'r', markersize = 12)

vmin = u.min().round(2)
vmax = u.max().round(2)

c0 = ax0.pcolormesh(x, y, u[:, :, -1], shading='auto', vmin = vmin, vmax = vmax)
c1 = ax1.pcolormesh(x, y, u[:, :, 1], shading='auto', vmin = vmin, vmax = vmax)
c2 = ax2.pcolormesh(x, y, u[:, :, 0], shading='auto', vmin = vmin, vmax = vmax)

ax0.set_title('$t = 7.0$', loc = 'right')
ax1.set_title('$t = 3.5$', loc = 'right')
ax2.set_title('$t = 0.0$', loc = 'right')

for axx in axs:
    axx.set_aspect('equal')
    axx.set_xticks([])
    axx.set_yticks([])
    axx.set_xlabel('$x$')
    axx.set_ylabel('$y$')

ax.axis('off')

cb0 = fig.colorbar(c0, ax = ax, format = '%.2f', orientation = 'vertical', shrink = 0.6, pad = 0.16)
cb0.ax.locator_params(nbins = 3)

plt.savefig('cylinder_prob.png', bbox_inches = 'tight', dpi = 300)

