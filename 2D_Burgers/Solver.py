import numpy as np
from matplotlib import pyplot as plt

class FDM_Burgers():
    def __init__(self, M, N, n_simu_steps, dt, dx, R):
        self.M = M
        self.N = N
        self.n_simu_steps = n_simu_steps
        self.dt = dt
        self.dx = dx
        self.R = R
        
    def apply_laplacian(self, mat, dx = 1.0):
        # dx is inversely proportional to N
        """This function applies a discretized Laplacian
        in periodic boundary conditions to a matrix
        
        For more information see 
        https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Implementation_via_operator_discretization
        """
    
        # the cell appears 4 times in the formula to compute
        # the total difference
        neigh_mat = -5*mat.copy()
    
        # Each direct neighbor on the lattice is counted in
        # the discrete difference formula
        neighbors = [ 
                        ( 4/3,  (-1, 0) ),
                        ( 4/3,  ( 0,-1) ),
                        ( 4/3,  ( 0, 1) ),
                        ( 4/3,  ( 1, 0) ),
                        (-1/12,  (-2, 0)),
                        (-1/12,  (0, -2)),
                        (-1/12,  (0, 2)),
                        (-1/12,  (2, 0)),
                    ]
    
        # shift matrix according to demanded neighbors
        # and add to this cell with corresponding weight
        for weight, neigh in neighbors:
            neigh_mat += weight * np.roll(mat, neigh, (0,1))
    
        return neigh_mat/dx**2
    
    def apply_dx(self, mat, dx = 1.0):
        ''' central diff for dx'''
    
        # np.roll, axis=0 -> row 
        # the total difference
        neigh_mat = -0*mat.copy()
    
        # Each direct neighbor on the lattice is counted in
        # the discrete difference formula
        neighbors = [ 
                        ( 1.0/12,  (2, 0) ),
                        ( -8.0/12,  (1, 0) ),
                        ( 8.0/12,  (-1, 0) ),
                        ( -1.0/12,  (-2, 0) )
                    ]
    
        # shift matrix according to demanded neighbors
        # and add to this cell with corresponding weight
        for weight, neigh in neighbors:
            neigh_mat += weight * np.roll(mat, neigh, (0,1))
    
        return neigh_mat/dx
    
    
    def apply_dy(self, mat, dy = 1.0):
        ''' central diff for dx'''
    
        # the total difference
        neigh_mat = -0*mat.copy()
    
        # Each direct neighbor on the lattice is counted in
        # the discrete difference formula
        neighbors = [ 
                        ( 1.0/12,  (0, 2) ),
                        ( -8.0/12,  (0, 1) ),
                        ( 8.0/12,  (0, -1) ),
                        ( -1.0/12,  (0, -2) )
                    ]
    
        # shift matrix according to demanded neighbors
        # and add to this cell with corresponding weight
        for weight, neigh in neighbors:
            neigh_mat += weight * np.roll(mat, neigh, (0,1))
    
        return neigh_mat/dy
    
    
    def get_temporal_diff(self, U, V, R, dx):
        # u and v in (h, w)
        
        laplace_u = self.apply_laplacian(U, dx)
        laplace_v = self.apply_laplacian(V, dx)
    
        u_x = self.apply_dx(U, dx)
        v_x = self.apply_dx(V, dx)
    
        u_y = self.apply_dy(U, dx)
        v_y = self.apply_dy(V, dx)
    
        # governing equation
        u_t = (1.0/R) * laplace_u - U * u_x - V * u_y
        v_t = (1.0/R) * laplace_v - U * v_x - V * v_y
        
        return u_t, v_t
    
    def update_rk4(self, U0, V0, R=100, dt=0.05, dx=1.0):
        """Update with Runge-kutta-4 method
           See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        """
        ############# Stage 1 ##############
        # compute the diffusion part of the update
    
        u_t, v_t = self.get_temporal_diff(U0, V0, R, dx)
    
        K1_u = u_t
        K1_v = v_t
    
        ############# Stage 2 ##############
        U1 = U0 + K1_u * dt/2.0
        V1 = V0 + K1_v * dt/2.0
    
        u_t, v_t = self.get_temporal_diff(U1, V1, R, dx)
    
        K2_u = u_t
        K2_v = v_t
    
        ############# Stage 3 ##############
        U2 = U0 + K2_u * dt/2.0
        V2 = V0 + K2_v * dt/2.0
    
        u_t, v_t = self.get_temporal_diff(U2, V2, R, dx)
    
        K3_u = u_t
        K3_v = v_t
    
        ############# Stage 4 ##############
        U3 = U0 + K3_u * dt
        V3 = V0 + K3_v * dt
    
        u_t, v_t = self.get_temporal_diff(U3, V3, R, dx)
    
        K4_u = u_t
        K4_v = v_t
    
        # Final solution
        U = U0 + dt*(K1_u+2*K2_u+2*K3_u+K4_u)/6.0
        V = V0 + dt*(K1_v+2*K2_v+2*K3_v+K4_v)/6.0
    
        return U, V
    
    def solve(self, U, V, d_save):    
        U_record = U.copy()[None,...]
        V_record = V.copy()[None,...]
        
        for step in range(self.n_simu_steps):
            
            U, V = self.update_rk4(U, V, self.R, self.dt, self.dx) #[h, w]

            if (step+1) % d_save == 0:
                print(step, '\n')
                U_record = np.concatenate((U_record, U[None,...]), axis=0) # [t,h,w]
                V_record = np.concatenate((V_record, V[None,...]), axis=0)
                           
        UV = np.concatenate((U_record[None,...], V_record[None,...]), axis=0) # (c,t,h,w)
        UV = np.transpose(UV, [1, 0, 2, 3]) # (t,c,h,w)
        
        return UV

    def plot(sef, UV, i):
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3))
        ax[0].imshow(UV[i, 0])
        ax[1].imshow(UV[i, 1])
        
        ax[0].set_title('U')
        ax[1].set_title('V')
        
        ax[0].set_xticks([])
        ax[0].set_yticks([])