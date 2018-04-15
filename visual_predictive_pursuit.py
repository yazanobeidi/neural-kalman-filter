import nengo
import numpy as np

__author__ = 'yazan'
# Based on "Kalman Filtering Naturally Accounts for Visually Guided and 
#           Predictive Smooth Pursuit Dynamics" by Orban de Xivry et. al.

model = nengo.Network()

with model:
    
    # set standard deviations for random distributions
    psi_sens_add, psi_sens_mult = 10/5, 1.5/5
    int_pred_add, int_pred_mult = psi_sens_add/2, psi_sens_mult/2
    Q, R, D, OM = 1, psi_sens_add, psi_sens_mult, 0.3
    Q2, R2, D2, OM2 = 1, int_pred_add, int_pred_mult, 0.3
    B_int = 1
    def di_add():
        return np.random.normal(scale=psi_sens_add)
    def di_mult():
        return np.random.normal(scale=psi_sens_mult)
    def beta_pred():
        return np.random.normal(scale=R2)
    def psi_pred():
        return np.random.normal(scale=D2)
    def epsilon_pred():
        return np.random.normal(scale=OM2)
    def nt_sens():
        return np.random.normal(scale=OM)
    def di_add_pred():
        return np.random.normal(scale=int_pred_add)
    def di_mult_pred():
        return np.random.normal(scale=int_pred_mult)
    G_int = 0.5 # also used 0.6
    tau_motor = 0.100 # seconds
    G_e = 1/tau_motor
    T_1 = 0.170 # seconds
    T_2 = 0.013 # seconds

    RS_det = nengo.Ensemble(n_neurons=500, dimensions=1)
    RS_noisy = nengo.Ensemble(n_neurons=500, dimensions=1, radius=2)
    K_sens_var = nengo.Ensemble(n_neurons=500, dimensions=2)
    K_sens = nengo.Ensemble(n_neurons=500, dimensions=1)
    # first dim holds conf_sens, second holds [1-k]
    conf_sens = nengo.Ensemble(n_neurons=500, dimensions=2)
    RS_sens_hat = nengo.Ensemble(n_neurons=500, dimensions=1, radius=5)
    # first dim holds RS_noisy-RS_sens_hat, second holds k
    RS_diff = nengo.Ensemble(n_neurons=500, dimensions=2)
    eye_vel = nengo.Ensemble(n_neurons=500, dimensions=1)
    target_vel = nengo.Node([0])
    
    TV_pred_hat = nengo.Ensemble(n_neurons=500, dimensions=1, radius=5)
    TV_diff = nengo.Ensemble(n_neurons=500, dimensions=2)
    TV_obs = nengo.Ensemble(n_neurons=500, dimensions=1)
    K_pred = nengo.Ensemble(n_neurons=500, dimensions=1)
    K_pred_var = nengo.Ensemble(n_neurons=500, dimensions=2)
    conf_pred = nengo.Ensemble(n_neurons=500, dimensions=1)
    u_pred = nengo.Ensemble(n_neurons=500, dimensions=1)
    
    RS_sens_weighting = nengo.Ensemble(n_neurons=500, dimensions=3)
    RS_pred_weighting = nengo.Ensemble(n_neurons=500, dimensions=3)
    RS_sum = nengo.Ensemble(n_neurons=500, dimensions=1)
    Gv = nengo.Ensemble(n_neurons=500, dimensions=1)
    Hv = nengo.Ensemble(n_neurons=500, dimensions=1)
    Av = nengo.Ensemble(n_neurons=500, dimensions=1)
    
    leaky_integrator = nengo.Ensemble(n_neurons=500, dimensions=1)
    
    pre_motor = nengo.Ensemble(n_neurons=500, dimensions=1)
    
    eye_pos = nengo.Ensemble(n_neurons=500, dimensions=1, radius=10)

    # Equation 1
    nengo.Connection(eye_vel, RS_det, transform=-1)
    nengo.Connection(target_vel, RS_det)
    nengo.Connection(RS_det, RS_noisy, 
                     function=lambda x: x+x*di_mult()+di_add())
    # Equation 5
    nengo.Connection(conf_sens[0], K_sens_var[0])
    nengo.Connection(RS_sens_hat, K_sens_var[1])
    nengo.Connection(K_sens_var, K_sens, 
                     function=lambda x: x[0]*(1/(x[0]+(R**2)+\
                                            (D**2)*(x[0]+x[1]*x[1])*(D**2))))
    # Equation 6
    nengo.Connection(K_sens, conf_sens[1], synapse=0.01, transform=0.01,
                     function=lambda x: 1-x)
    nengo.Connection(conf_sens, conf_sens[0], synapse=0.01, 
                     function=lambda x: Q**2+OM**2+x[0]*x[1])
    # Equation 4
    nengo.Connection(RS_noisy, RS_diff[0])
    nengo.Connection(RS_sens_hat, RS_diff[0], transform=-1)
    nengo.Connection(K_sens, RS_diff[1])
    nengo.Connection(RS_diff, RS_sens_hat, synapse=0.01, 
                     function=lambda x: 0.01*x[0]*x[1]+nt_sens())
    nengo.Connection(RS_sens_hat, RS_sens_hat, synapse=0.01)
    # Equation 10
    # later to change target_vel to
    # be the sum of RS_sens_hat and pred_eye_vel
    # see Eq. 8 explanation
    nengo.Connection(target_vel, TV_obs, 
                     function=lambda x: x+x*beta_pred()+psi_pred())
    # Equation 11
    nengo.Connection(TV_pred_hat, TV_diff[0], transform=-1)
    nengo.Connection(TV_obs, TV_diff[0])
    nengo.Connection(K_pred, TV_diff[1])
    nengo.Connection(TV_diff, TV_pred_hat, synapse=0.01, 
                     function=lambda x: 0.01*x[0]*x[1]+epsilon_pred())
    nengo.Connection(TV_pred_hat, TV_pred_hat, synapse=0.01)
    # Equation 12
    nengo.Connection(conf_pred, K_pred_var[0])
    nengo.Connection(TV_pred_hat, K_pred_var[1])
    nengo.Connection(K_pred_var, K_pred, 
                     function=lambda x: x[0]*(1/(x[0]+(R2**2)+\
                                            (D2**2)*(x[0]+x[1]*x[1])*(D2**2))))
    # Equation 13
    nengo.Connection(K_pred, conf_pred, function=lambda x: Q2+OM2+(1-x))
    # Equation 15
    nengo.Connection(conf_sens[0], RS_sens_weighting[0])
    nengo.Connection(conf_pred, RS_sens_weighting[1])
    nengo.Connection(RS_sens_hat, RS_sens_weighting[2]) 
    nengo.Connection(RS_sens_weighting, RS_sum, 
                    function=lambda x: (x[1]/(x[0]+x[1]))*x[2])
    nengo.Connection(conf_sens[0], RS_pred_weighting[0])
    nengo.Connection(conf_pred, RS_pred_weighting[1])
    nengo.Connection(RS_sens_hat, RS_pred_weighting[2]) 
    nengo.Connection(RS_pred_weighting, RS_sum, 
                    function=lambda x: (x[0]/(x[0]+x[1]))*x[2])
    # Equation 16
    # since the following does not work (improper transfer function)
    #nengo.Connection(RS_sum, Gv, 
    #                 synapse=nengo.LinearFilter([7, 0], [0, 1]))
    nengo.Connection(RS_sum, Gv, synapse=None)
    nengo.Connection(RS_sum, Gv, transform=-1, synapse=0.2)
    # Equation 17
    nengo.Connection(Gv, Hv, 
                     synapse=nengo.LinearFilter([35**2], [1,2*0.8*35, 35**2]))
    # Equation 18
    nengo.Connection(Hv, Av, transform=0.7)
    
    # Leaky Integrator, Fig 3
    nengo.Connection(leaky_integrator, leaky_integrator, synapse=0.01, 
                     function=lambda x: x*G_int*G_e)
    nengo.Connection(Av, leaky_integrator, function=lambda x: 0.01*x*G_int,
                     synapse=nengo.LinearFilter([tau_motor, 0], 
                                                [tau_motor, 1, 0]))
                                                
    # Premotor system, Fig 1
    nengo.Connection(leaky_integrator, pre_motor, transform=T_1)
    nengo.Connection(leaky_integrator, pre_motor, 
                     synapse=nengo.LinearFilter([1], [1, 0]))
                     
    # Eye Plant
    nengo.Connection(pre_motor, eye_pos,
                     synapse=nengo.LinearFilter([1], [T_1*T_2, T_1+T_2, 1]))

    # Eye Velocity
    # since the following does not work (improper transfer function)
    #nengo.Connection(eye_pos, eye_vel,
    #                 synapse=nengo.LinearFilter([1, 0], [0, 1]))
    nengo.Connection(eye_pos, eye_vel, synapse=None)
    nengo.Connection(eye_pos, eye_vel, transform=-1, synapse=0.2)
    