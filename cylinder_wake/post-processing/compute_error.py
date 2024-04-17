import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import loadmat
from scipy.stats import pearsonr
c = 10.0

filename_pinn = f'../results/res_cylinder_Gn{c}.npz'
#%%
data = loadmat('../data/cylinder_nektar_wake.mat')
u = data['U_star'][:, 0]
v = data['U_star'][:, 1]
p = data['p_star']

u = u.reshape((-1, 100, 200))
v = v.reshape((-1, 100, 200))
p = p.reshape((-1, 100, 200))

x = data['X_star'][:, 0]
y = data['X_star'][:, 1]
t = data['t']

x = x.reshape((-1, 100))
y = y.reshape((-1, 100))


nt = 71
u = u[:, :, :nt]
v = v[:, :, :nt]
p = p[:, :, :nt]

u_noise = u + np.random.normal(0, c, np.shape(u)) * u / 100
v_noise = v + np.random.normal(0, c, np.shape(v)) * v / 100
u_noise = np.stack([u_noise, v_noise])

u = np.stack((u, v, p), axis = 0)

print(f"Total Number of the sample points: {len(u.flatten())}")
print(f"At Noise level = {c}")

##################
# Examine the l2-norm error 
##################
def error(u, up):
    return np.linalg.norm((u - up), axis = (1, 2))/np.linalg.norm(u, axis = (1, 2)) * 100

u_pinn = np.load('../results/' + filename_pinn)['up']

u_pinn[2] = u_pinn[2] - u_pinn[2].mean() + p.mean()
e_pinn = error(u, u_pinn).mean(1)

print("Error: U, V, P ")
print(e_pinn)

Names = ["U", "V", "P"]
for i in range(3):
    r_u, _ = pearsonr(u_pinn[i].flatten(), u[i].flatten())
    print(f"cross-correlation of {Names[i]}: \t {r_u :.3f}")

##################
# Examine the cross-correlation 
##################

uy = np.gradient(u[0], y[:, 0], axis = 0, edge_order=2)
vx = np.gradient(u[1], x[0], axis = 1, edge_order=2)
w = vx - uy
# w = np.where(np.abs(w) < 0.12, 0, w)

uy_noise = np.gradient(u_noise[0], y[:, 0], axis = 0, edge_order=2)
vx_noise = np.gradient(u_noise[1],  x[0], axis = 1, edge_order=2)
w_noise = vx_noise - uy_noise
# w_noise = np.where(np.abs(w_noise) < 0.12, 0, w_noise)

uy_pinn = np.gradient(u_pinn[0], y[:, 0], axis = 0, edge_order=2)
vx_pinn = np.gradient(u_pinn[1],  x[0], axis = 1, edge_order=2)
w_pinn = vx_pinn - uy_pinn


r_p,_     = pearsonr(w_pinn.flatten(), w.flatten())
r_n,_     = pearsonr(w_noise.flatten(), w.flatten())
print(f"For Vorticity:")
print(f"The cross-correlation:\nPINNS\t{r_p:.3f}\nNoise\t{r_n:.3f}")

