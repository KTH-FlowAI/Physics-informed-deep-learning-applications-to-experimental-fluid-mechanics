import numpy as np
import tensorflow as tf
from tensorflow.keras import models
"""
PINNs for wind turbine PIV data
"""
class PINNs(models.Model):
    def __init__(self, model, optimizer, sopt, epochs, s_w, u_w, **kwargs):
        super(PINNs, self).__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.sopt = sopt
        self.epochs = epochs
        
        self.s_w = s_w
        self.u_w = u_w
        
        self.hist = []
        self.epoch = 0


    @tf.function
    def net_f(self, cp):
        x = cp[:, 0]
        y = cp[:, 1]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            X = tf.stack([x,y],axis=-1)
            # X = self.scalex(X)
            pred = self.model(X)
            # pred = self.scale_r(pred)
            u = pred[:, 0]
            uu = pred[:, 1]
            vv = pred[:, 2]
            uv = pred[:, 3]
            # Predicting as if there is no reference
            v = pred[:, 4]
            p = pred[:,5]


            u_x = tape.gradient(u, x)
            v_x = tape.gradient(v, x)
            p_x = tape.gradient(p, x)

            u_y = tape.gradient(u, y)
            v_y = tape.gradient(v, y)
            p_y = tape.gradient(p, y)


        uu_x = tape.gradient(uu,y)    
        uv_x = tape.gradient(uv,x)    
    
        uv_y = tape.gradient(uv,y)    
        vv_y = tape.gradient(vv,y)    
    
        # continuity 
        f0 = u_x + v_y
        # Momentum X
        f1 = u * u_x +  v * u_y + p_x + uu_x  + uv_y 
        # Momentum Y
        f2 = u * v_x +  v * v_y + p_y + uv_x  + vv_y 
        
        f = tf.stack([f0, f1, f2], axis = -1)
        return f
    
    
    @tf.function
    def train_step(self, ic, cp):
        with tf.GradientTape() as tape:
      
            u_p_ic = self.model(ic[:, :2])
            
            f = self.net_f(cp)
            
            loss_ic = tf.reduce_mean(tf.square(ic[:, 2:] - u_p_ic[:, :-2]))
            
            loss_f = tf.reduce_mean(tf.square(f))
            
            loss_u =loss_ic
            loss = self.s_w*loss_u + self.u_w * loss_f
            
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        l1 = tf.reduce_mean(loss)
        l2 = tf.reduce_mean(loss_u)
        l3 = tf.reduce_mean(loss_f)
        
        tf.print('loss:', l1, 'loss_u:', l2, 'loss_f:', l3)

        ll = tf.stack([l1,l2,l3],axis=-1)
        return loss, grads,ll
    
    def fit_scale(self, y):
        ymax = tf.reduce_max(tf.abs(y), axis = 0)
        self.ymax = tf.concat([ymax, [1.0]], 0)
        return y / ymax
    
    @tf.function
    def scale(self, y):
        return y / self.ymax
    
    @tf.function
    def scale_r(self, ys):
        return ys * self.ymax
    
    def fit_scalex(self, x):
        xmax = tf.reduce_max(tf.abs(x), axis = 0)
        xmin = tf.reduce_min(x, axis = 0)
        self.xmax = xmax
        self.xmin = xmin
        xs = ((x - xmin) / (xmax - xmin))
        return xs
    @tf.function
    def scalex(self, x):
        xs = ((x - self.xmin) / (self.xmax - self.xmin)) 
        return xs
    
    @tf.function
    def scalex_r(self, xs):
        x = (xs) * (self.xmax - self.xmin) + self.xmin
        return x



    def fit(self, ic,cp):
        ic = tf.convert_to_tensor(ic, tf.float32)
        cp = tf.convert_to_tensor(cp, tf.float32)

    
    
        def func(params_1d):
            self.sopt.assign_params(params_1d)
            tf.print('epoch:', self.epoch)
            loss, grads,ll = self.train_step(ic, cp)
            grads = tf.dynamic_stitch(self.sopt.idx, grads)
            self.epoch += 1
            self.hist.append(ll.numpy())
            return loss.numpy().astype(np.float64), grads.numpy().astype(np.float64)
        
        for epoch in range(self.epochs):
            tf.print('epoch:', self.epoch)
            loss, grads, ll = self.train_step(ic, cp)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            self.epoch += 1
            self.hist.append(ll.numpy())
    
            
            
        self.sopt.minimize(func)
            
        return self.hist
    
    def predict(self, cp):

        cp = tf.convert_to_tensor(cp, tf.float32)

        u_p = self.model(cp)
 
        return u_p.numpy()