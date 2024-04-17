import numpy as np
import tensorflow as tf
from tensorflow.keras import models

class PINNs(models.Model):
    def __init__(self, model, optimizer, sopt, epochs, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.sopt = sopt
        self.epochs = epochs
        self.hist = []
        self.epoch = 0
              
    @tf.function
    def net_f(self, cp):
        x = cp[:, 0]
        y = cp[:, 1]
        t = cp[:, 2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            tape.watch(t)
            X = tf.stack([x, y, t], axis = -1)
            pred = self.model(X)
            u = pred[:, 0]
            v = pred[:, 1]
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            v_y = tape.gradient(v, y)
            v_x = tape.gradient(v, x)
            
            
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        v_yy = tape.gradient(v_y, y)
        v_xx = tape.gradient(v_x, x)

        u_t = tape.gradient(u, t)
        v_t = tape.gradient(v, t)
        
        f1 = - u_t - u * u_x - v * u_y + (1/200) * (u_xx + u_yy)
        f2 = - v_t - u * v_x - v * v_y + (1/200) * (v_xx + v_yy)
        
        f = tf.stack([f1, f2], axis = -1)
        return f
    
    
    @tf.function
    def train_step(self, ic, bc, cp):
        with tf.GradientTape() as tape:
            u_p_bc = self.model(bc[:, :3])
            u_p_ic = self.model(ic[:, :3])
            
            f = self.net_f(cp)
            
            loss_bc = tf.reduce_mean(tf.square(bc[:, 3:] - u_p_bc))
            loss_ic = tf.reduce_mean(tf.square(ic[:, 3:] - u_p_ic))
            
            loss_f = tf.reduce_mean(tf.square(f))
            
            loss_u = loss_bc + loss_ic
            loss = loss_u + loss_f
            
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        self.hist.append(l1)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)
        return loss, grads
    
    def fit(self, ic, bc, cp):
        ic = tf.convert_to_tensor(ic, tf.float32)
        bc = tf.convert_to_tensor(bc, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)
        
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads = self.train_step(ic, bc, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads = self.train_step(ic, bc, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            
            
        self.sopt.minimize(func)
            
        return self.hist
    
    def predict(self, cp):
        cp = tf.convert_to_tensor(cp, tf.float32)
        u_p = self.model(cp)
        return u_p.numpy()