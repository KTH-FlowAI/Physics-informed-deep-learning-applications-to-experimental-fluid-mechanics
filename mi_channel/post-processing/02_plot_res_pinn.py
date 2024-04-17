import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
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

u = np.stack([u, v, w])

yy, zz, xx = np.meshgrid(y, z, x)
dt = 0.2
nv = 4
nz, ny, nx = xx.shape

pred1 = np.load('../results/res_pinn_t3_s16.npz')
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
#%%
x = xx[0]
y = yy[0]

omega_z = omega_z[0]
omega_z_pred1 = omega_z_pred1[0]
omega_z_pred2 = omega_z_pred2[0]

u = u[:, 0]
u_pred1 = u_pred1[:, 0]
u_pred2 = u_pred2[:, 0]
#%%
n = 5 #(t = 1.0) 
# n = 15 #(t = 3.0)

fig, ax = plt.subplots(3, 4, figsize=(9, 6), sharex = True, sharey = True)
plt.set_cmap('cmo.tarn')

mi0 = u[0, :, :, n].min()
mx0 = u[0, :, :, n].max()

mi1 = u[1, :, :, n].min()
mx1 = u[1, :, :, n].max()

mi2 = u[2, :, :, n].min()
mx2 = u[2, :, :, n].max()

mi3 = omega_z[:, :, n].min()
mx3 = omega_z[:, :, n].max()

vmin2 = u[:, :, n].min()
vmax2 = u[:, :, n].max()

l = 12

c0 = ax[0, 0].contourf(x, y, u[0, :, :, n], levels = l, vmin = mi0, vmax = mx0)
c1 = ax[0, 1].contourf(x, y, u[1, :, :, n], levels = l, vmin = mi1, vmax = mx1)
c2 = ax[0, 2].contourf(x, y, u[2, :, :, n], levels = l, vmin = mi2, vmax = mx2)
c3 = ax[0, 3].contourf(x, y, omega_z[:, :, n], levels = l, vmin = mi3, vmax = mx3)


ax[1, 0].contourf(x, y, u_pred1[0, :, :, n], levels = l, vmin = mi0, vmax = mx0)
ax[1, 1].contourf(x, y, u_pred1[1, :, :, n], levels = l, vmin = mi1, vmax = mx1)
ax[1, 2].contourf(x, y, u_pred1[2, :, :, n], levels = l, vmin = mi2, vmax = mx2)
ax[1, 3].contourf(x, y, omega_z_pred1[:, :, n], levels = l, vmin = mi3, vmax = mx3)

ax[2, 0].contourf(x, y, u_pred2[0, :, :, n], levels = l, vmin = mi0, vmax = mx0)
ax[2, 1].contourf(x, y, u_pred2[1, :, :, n], levels = l, vmin = mi1, vmax = mx1)
ax[2, 2].contourf(x, y, u_pred2[2, :, :, n], levels = l, vmin = mi2, vmax = mx2)
ax[2, 3].contourf(x, y, omega_z_pred2[:, :, n], levels = l, vmin = mi3, vmax = mx3)
    

for axx in ax.flatten():
    axx.set_aspect('equal')

ax[0, 0].set_ylabel('Reference \n $y$')
ax[1, 0].set_ylabel('PINN \n t3--s16 \n $y$')
ax[2, 0].set_ylabel('PINN \n t3--s8 \n $y$')

for axx in ax[-1]:
    axx.set_xlabel('$x$')
    
tit = ['$u$', '$v$', '$w$', '$\\omega_z$']
i = 0
for axx in ax[0]:
    axx.set_title(tit[i])
    i += 1
    
cb0 = fig.colorbar(c0, ax = ax[:, 0], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb0.ax.locator_params(nbins = 3)

cb1 = fig.colorbar(c1, ax = ax[:, 1], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb1.ax.locator_params(nbins = 3)

cb2 = fig.colorbar(c2, ax = ax[:, 2], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb2.ax.locator_params(nbins = 3)

cb3 = fig.colorbar(c3, ax = ax[:, 3], format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb3.ax.locator_params(nbins = 3)


plt.savefig('channel_res_pinn.png', bbox_inches = 'tight', dpi = 300)

#%%
fig, ax = plt.subplots(1, 3, figsize=(9, 2), sharex = True, sharey = True)
plt.set_cmap('cmo.tarn')
sx = 2
sy = 4
ul = u[1, ::sy, ::sx]
xl = x[0, ::sx]
yl = y[::sy, 0]


c0 = ax[0].pcolormesh(xl, yl, ul[:, :, 0])

c1 = ax[1].pcolormesh(xl, yl, ul[:, :, 10])

c2 = ax[2].pcolormesh(xl, yl, ul[:, :, 20])

ax[0].set_yticks([yl.min(), yl.max()])
ax[0].set_yticklabels([0, 1])

for axx in ax:
    axx.set_xlabel('$x$')
    axx.set_aspect('equal')

ax[0].set_ylabel('$y$')

ax[0].set_title('$t = 0.0$')
ax[1].set_title('$t = 2.0$')
ax[2].set_title('$t = 4.0$')

cb0 = fig.colorbar(c0, ax = ax[0], format = '%.2f', orientation = 'vertical', shrink = 0.6, pad = 0.1)
cb0.ax.locator_params(nbins = 3)

cb1 = fig.colorbar(c1, ax = ax[1], format = '%.2f', orientation = 'vertical', shrink = 0.6, pad = 0.1)
cb1.ax.locator_params(nbins = 3)

cb2 = fig.colorbar(c2, ax = ax[2], format = '%.2f', orientation = 'vertical', shrink = 0.6, pad = 0.1)
cb2.ax.locator_params(nbins = 3)

plt.savefig('channel_training_data.png', bbox_inches = 'tight', dpi = 300)
