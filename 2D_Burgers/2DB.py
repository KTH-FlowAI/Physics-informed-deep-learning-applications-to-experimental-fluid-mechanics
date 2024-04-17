import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from pyDOE import lhs


import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, activations
from ScipyOP import optimizer as SciOP

from PINN_2DB import PINNs

from time import time

data = loadmat('./burgers_128x128.mat')
#%%
u = data['uv'][:200, 0]
v = data['uv'][:200, 1]
nt, ny, nx = u.shape

dt = 0.0001 * 20
t = np.arange(0, nt) * dt

Nu  = len(u.flatten())
Nt  = len(t.flatten())

print(f"Total Data: NU = {Nu}, Nt = {Nt}")
t = t[::10]

u = u[::10].transpose((1, 2, 0))
v = v[::10].transpose((1, 2, 0))


Nu  = len(u.flatten())
Nt  = len(t.flatten())


print(f"Training Data: NU = {Nu}, Nt = {Nt}")
x = np.linspace(0, 1, nx, endpoint=True)
y = np.linspace(0, 1, ny, endpoint=True)

print(u.shape, v.shape, )

xx, yy = np.meshgrid(x, y)
#%%
np.random.seed(24)

lb = np.array([x.min(), y.min(), t.min()])
ub = np.array([x.max(), y.max(), t.max()])

cp = lb + (ub-lb) * lhs(3, 20000)

#%%
ns = len(xx.flatten())

ic = np.array([xx.flatten(), yy.flatten(), np.zeros((ns,)) + t[0],
                u[:, :, 0].flatten(), v[:, :, 0].flatten()]).T

pr = 0.1
ind_ic = np.random.choice([False, True], len(ic), p=[1 - pr, pr])
ic = ic[ind_ic]
#%%
ind_bc = np.zeros(xx.shape, dtype = bool)
ind_bc[[0, -1], :] = True; ind_bc[:, [0, -1]] = True

X, Y, T = np.meshgrid(x, y, t)

x_bc = X[ind_bc].flatten()
y_bc = Y[ind_bc].flatten()
t_bc = T[ind_bc].flatten()

u_bc = u[ind_bc].flatten()
v_bc = v[ind_bc].flatten()

bc = np.array([x_bc, y_bc, t_bc, u_bc, v_bc]).T

pr = 0.1
indx_bc = np.random.choice([False, True], len(bc), p=[1 - pr, pr])
bc = bc[indx_bc]
#%%
act = activations.tanh
inp = layers.Input(shape = (3,))
hl = inp
for i in range(8):
    hl = layers.Dense(20, activation = act)(hl)
out = layers.Dense(2)(hl)

model = models.Model(inp, out)
print(model.summary())

lr = 1e-3
opt = optimizers.Adam(lr)
sopt = SciOP(model)

st_time = time()

pinn = PINNs(model, opt, sopt, 1000)
hist = pinn.fit(ic, bc, cp)

en_time = time()
comp_time = en_time - st_time
#%%
cp = np.array([X.flatten(), Y.flatten(), T.flatten()]).T
up = pinn.predict(cp).T
up = up.reshape((2, ny, nx, -1), order = 'C')
#%%
fig, ax = plt.subplots(2, 2)
n = -1
ax[0, 0].contourf(xx, yy, up[0, :, :, n])
ax[0, 1].contourf(xx, yy, u[:, :, n])

ax[1, 0].contourf(xx, yy, up[1, :, :, n])
ax[1, 1].contourf(xx, yy, v[:, :, n])

#%%
np.savez_compressed('2dB_PINN', up = up, comp_time = comp_time)
model.save('2dB_PINN.h5')
