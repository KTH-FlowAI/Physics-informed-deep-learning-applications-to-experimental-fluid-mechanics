import numpy as np
from matplotlib import pyplot as plt 
import cmocean
import cmocean.cm as cmo
from matplotlib import gridspec

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
nz, ny, nx = xx.shape

pred1 = np.load('../results/res_pinn_t3_s16.npz')
u_pred1 = pred1['up'][:3]

pred2 = np.load('../results/res_pinn_t3_s8.npz')
u_pred2 = pred2['up'][:3]
#%%
x = xx[0]
y = yy[0]

u = u[:, 0]
u_pred1 = u_pred1[:, 0]
u_pred2 = u_pred2[:, 0]

locx = int(nx/2)
locy = int(ny/1.3)

fig = plt.figure(figsize = (9, 6))
gs = gridspec.GridSpec(3, 4, figure = fig, wspace = 0.8, hspace = 0.4)
ax0 = fig.add_subplot(gs[:, :1], aspect = 'auto')
ax1 = fig.add_subplot(gs[0, 1:], aspect = 'auto')
ax2 = fig.add_subplot(gs[1, 1:], aspect = 'auto', sharex=ax1)
ax3 = fig.add_subplot(gs[2, 1:], aspect = 'auto', sharex=ax1)
plt.set_cmap('cmo.tarn')

ax = [ax0, ax1, ax2, ax3]
l = 8

c0 = ax0.contourf(x, y, u[0, :, :, 0], levels = l)
ax0.scatter(x[locy, locx], y[locy, locx], marker = 's', s = 100, c = 'tab:red')
ax0.set_aspect('equal')

ax0.set_ylabel('$y$')
ax0.set_xlabel('$x$')

ax0.set_yticks([y.min(), 1])
ax0.set_xticks([-0.5, 0.5])

ax0.set_yticklabels([0, 1])

cb0 = fig.colorbar(c0, ax = ax0, format = '%.2f', orientation = 'horizontal', shrink = 0.9, pad = 0.15, aspect = 20)
cb0.ax.locator_params(nbins = 3)

ax1.plot(t, u[0, locy, locx], label = 'Reference')
ax2.plot(t, u[1, locy, locx])
ax3.plot(t, u[2, locy, locx])


ax1.plot(t, u_pred2[0, locy, locx], ls = '--', label = 'PINN--t3--s8')
ax2.plot(t, u_pred2[1, locy, locx], ls = '--')
ax3.plot(t, u_pred2[2, locy, locx], ls = '--')

ax1.plot(t, u_pred1[0, locy, locx], ls = '-.', label = 'PINN--t3--s16')
ax2.plot(t, u_pred1[1, locy, locx], ls = '-.')
ax3.plot(t, u_pred1[2, locy, locx], ls = '-.')

ax1.legend(frameon=False, ncol = 3, loc = (0.0, 1.03), fontsize=13)

ax1.set_xlim(0, 4)
ax1.set_xticks([0.0, 2.0, 4.0])
ax3.set_xlabel('$t$')

ax1.grid(visible=True, axis='x', c = 'pink', ls = '--', lw = 2)
ax2.grid(visible=True, axis='x', c = 'pink', ls = '--', lw = 2)
ax3.grid(visible=True, axis='x', c = 'pink', ls = '--', lw = 2)


ax1.set_ylabel('$u$')
ax2.set_ylabel('$v$')
ax3.set_ylabel('$w$')


plt.savefig('channel_signals.png', bbox_inches = 'tight', dpi = 300)
# 