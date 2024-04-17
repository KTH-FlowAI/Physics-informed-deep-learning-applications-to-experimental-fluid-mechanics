import scipy.io
from Solver import FDM_Burgers


ic = scipy.io.loadmat('ICs/IC_Burgers.mat')
U = ic['IC'][0, 0]
V = ic['IC'][0, 1]

M, N = U.shape
n_simu_steps = 30000
dt = 0.0001 # maximum 0.003   
dx = 1.0 / M
R = 200.0

solver = FDM_Burgers(M, N, n_simu_steps, dt, dx, R)
save_step = 20
UV = solver.solve(U, V, save_step) 
solver.plot(UV, -1)

# save data
data_save_dir = './'
scipy.io.savemat(data_save_dir + f'burgers_{M}x{N}.mat', {'uv': UV})

