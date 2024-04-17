import numpy as np
from scipy.io import loadmat
from pyDOE import lhs
from tensorflow.keras import models, layers, optimizers, activations
from PINN_cylinder import PINN
from time import time

def get_data(c, n_cp=2000):
    data = loadmat('../data/cylinder_nektar_wake.mat')
    u = data['U_star'][:, 0]
    v = data['U_star'][:, 1]
    p = data['p_star']
    x = data['X_star'][:, 0]
    y = data['X_star'][:, 1]
    t = data['t']
    
    u = u.reshape((-1, 100, 200))
    v = v.reshape((-1, 100, 200))
    p = p.reshape((-1, 100, 200))
    xx = x.reshape((-1, 100))
    yy = y.reshape((-1, 100))
    
    nt = 71
    t = t[:nt]
    x = xx[0, :]
    y = yy[:, 0]
    u = u[:, :, :nt]
    v = v[:, :, :nt]
    p = p[:, :, :nt]
    grid = (x, y, t)
    
    # collocation points for unsupervised learning
    np.random.seed(24)
    lb = np.array([x.min(), y.min(), t.min()])
    ub = np.array([x.max(), y.max(), t.max()])
    cp = lb + (ub-lb) * lhs(3, n_cp)
    
    # low resolution and noisy data for supervised learning
    u_lr = u[::10, ::10, ::35]
    v_lr = v[::10, ::10, ::35]
    x_lr = x[::10]
    y_lr = y[::10]
    t_lr = t[::35]
    
    # add gaussian noise
    u_lr = u_lr + np.random.normal(0, c, np.shape(u_lr)) * u_lr / 100
    v_lr = v_lr + np.random.normal(0, c, np.shape(v_lr)) * v_lr / 100
    
    X, Y, T = np.meshgrid(x_lr, y_lr, t_lr)
    x_sp = X.flatten()
    y_sp = Y.flatten()
    t_sp = T.flatten()
    u_sp = u_lr.flatten()
    v_sp = v_lr.flatten()
    sp = np.array([x_sp, y_sp, t_sp, u_sp, v_sp]).T
    return sp, cp, grid

c = 10.0 # std of gaussian noise
sp, cp, grid = get_data(c)
#%% model training 
nv = 3 #(u, v, p)
act = activations.tanh
inp = layers.Input(shape = (3,))
hl = inp
for i in range(4):
    hl = layers.Dense(20, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())

lr = 1e-3
opt = optimizers.Adam(lr)
n_training_with_adam = 1000

st_time = time()

pinn = PINN(model, opt, n_training_with_adam)
hist = pinn.fit(sp, cp)

en_time = time()
comp_time = en_time - st_time
#%%
x, y, t = grid
ny = y.shape[0]
nx = x.shape[0]

X, Y, T = np.meshgrid(x, y, t)
cp = np.array([X.flatten(), Y.flatten(), T.flatten()]).T
up = pinn.predict(cp).T
up = up.reshape((3, ny, nx, -1), order = 'C')
#%%
np.savez_compressed(f'../results/res_cylinder_Gn{c}', up = up, hist = hist, comp_time=comp_time)
model.save(f'../results/pinn_cylinder_Gn{c}.h5')