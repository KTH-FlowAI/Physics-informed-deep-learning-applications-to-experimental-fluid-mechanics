import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import scipy.optimize as sopt


class PINN(models.Model):
    def __init__(self, model, optimizer, epochs, **kwargs):
        super(PINN, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.sopt = lbfgs_optimizer(self.trainable_variables)
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
        self.nu = 1/5000
              
    @tf.function
    def net_f(self, cp):
        x = cp[:, 0]
        y = cp[:, 1]
        z = cp[:, 2]
        t = cp[:, 3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)
            tape.watch(t)
            X = tf.stack([x, y, z, t], axis = -1)
            pred = self.model(X)
            u = pred[:, 0]
            v = pred[:, 1]
            w = pred[:, 2]
            p = pred[:, 3]
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_z = tape.gradient(u, z)
            
            v_x = tape.gradient(v, x)
            v_y = tape.gradient(v, y)
            v_z = tape.gradient(v, z)
            
            w_x = tape.gradient(w, x)
            w_y = tape.gradient(w, y)
            w_z = tape.gradient(w, z)
            
            
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        u_zz = tape.gradient(u_z, z)
        
        v_xx = tape.gradient(v_x, x)
        v_yy = tape.gradient(v_y, y)
        v_zz = tape.gradient(v_z, z)
        
        w_xx = tape.gradient(w_x, x)
        w_yy = tape.gradient(w_y, y)
        w_zz = tape.gradient(w_z, z)
        
        p_x = tape.gradient(p, x)
        p_y = tape.gradient(p, y)
        p_z = tape.gradient(p, z)
        
        u_t = tape.gradient(u, t)
        v_t = tape.gradient(v, t)
        w_t = tape.gradient(w, t)
        
        f1 = - u_t - u * u_x - v * u_y - w * u_z - p_x + self.nu * (u_xx + u_yy + u_zz)
        f2 = - v_t - u * v_x - v * v_y - w * v_z - p_y + self.nu * (v_xx + v_yy + v_zz)
        f3 = - w_t - u * w_x - v * w_y - w * w_z - p_z + self.nu * (w_xx + w_yy + w_zz)
        
        f4 = u_x + v_y + w_z
        
        f = tf.stack([f1, f2, f3, f4], axis = -1)
        return f
    
    
    @tf.function
    def train_step(self, sp, cp):
        with tf.GradientTape() as tape:
            u_p_sp = self.model(sp[:, :4])            
            f = self.net_f(cp)
            
            loss_u = tf.reduce_mean(tf.square(sp[:, 4:] - u_p_sp[:, :-1]))
            
            loss_f = tf.reduce_mean(tf.square(f))
            loss = loss_u + loss_f
            
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)
        return loss, grads, tf.stack([l1, l2, l3])
    
    def fit(self, sp, cp):
        sp = tf.convert_to_tensor(sp, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
        
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(sp, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(hist.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, hist = self.train_step(sp, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(hist.numpy())
            
            
        self.sopt.minimize(func)
            
        return np.array(self.hist)
    
    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        u_p = self.model(cp)
        return u_p.numpy()
    
    
class lbfgs_optimizer():
    def __init__(self, trainable_vars, method = 'L-BFGS-B'):
        super(lbfgs_optimizer, self).__init__()
        self.trainable_variables = trainable_vars
        self.method = method
        
        self.shapes = tf.shape_n(self.trainable_variables)
        self.n_tensors = len(self.shapes)

        count = 0
        idx = [] # stitch indices
        part = [] # partition indices
    
        for i, shape in enumerate(self.shapes):
            n = np.product(shape)
            idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
            part.extend([i]*n)
            count += n
    
        self.part = tf.constant(part)
        self.idx = idx
    
    def assign_params(self, params_1d):
        params_1d = tf.cast(params_1d, dtype = tf.float32)
        params = tf.dynamic_partition(params_1d, self.part, self.n_tensors)
        for i, (shape, param) in enumerate(zip(self.shapes, params)):
            self.trainable_variables[i].assign(tf.reshape(param, shape))       
    
    def minimize(self, func):
        init_params = tf.dynamic_stitch(self.idx, self.trainable_variables)
        results = sopt.minimize(fun = func, 
                            x0 = init_params, 
                            method = self.method,
                            jac = True, options = {'iprint' : 0,
                                                   'maxiter': 50000,
                                                   'maxfun' : 50000,
                                                   'maxcor' : 50,
                                                   'maxls': 50,
                                                   'gtol': 1.0 * np.finfo(float).eps,
                                                   'ftol' : 1.0 * np.finfo(float).eps})