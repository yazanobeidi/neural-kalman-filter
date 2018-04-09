import nengo
import numpy as np

__author__ = 'yazan'
# Based on "Kalman Filtering Naturally Accounts for Visually Guided and 
#           Predictive Smooth Pursuit Dynamics" by Orban de Xivry et. al.

model = nengo.Network()

with model:
    
    Q, R, D, OM = 0.001, 0.001, 0.001, 0.001
    Q2, R2, D2, OM2 = 0.001, 0.001, 0.001, 0.001
    psi_sens_add, psi_sens_mult = 0.001, 0.001
    int_pred_add, int_pred_mult = 0.001, 0.001
    B_int = 1
    di_add = np.random.normal(psi_sens_add)
    di_mult = np.random.normal(psi_sens_mult)
    beta_pred = np.random.normal(R2)
    psi_pred = np.random.normal(D2)
    epsilon_pred = np.random.normal(OM2)
    nt_sens = np.random.normal(OM)
    di_add_pred = np.random.normal(int_pred_add)
    di_mult_pred = np.random.normal(int_pred_mult)


    RS_det = nengo.Ensemble(n_neurons=100, dimensions=1)
    RS_noisy = nengo.Ensemble(n_neurons=100, dimensions=1)
    K_sens_var = nengo.Ensemble(n_neurons=100, dimensions=2)
    K_sens = nengo.Ensemble(n_neurons=100, dimensions=1)
    # first dim holds conf_sens, second holds [1-k]
    conf_sens = nengo.Ensemble(n_neurons=100, dimensions=2)
    RS_sens_hat = nengo.Ensemble(n_neurons=100, dimensions=1)
    # first dim holds RS_noisy-RS_sens_hat, second holds k
    RS_diff = nengo.Ensemble(n_neurons=100, dimensions=2)
    eye_vel = nengo.Ensemble(n_neurons=100, dimensions=1)
    target_vel = nengo.Node([0])
    
    TV_pred_hat = nengo.Ensemble(n_neurons=100, dimensions=1)
    TV_diff = nengo.Ensemble(n_neurons=100, dimensions=2)
    TV_obs = nengo.Ensemble(n_neurons=100, dimensions=1)
    K_pred = nengo.Ensemble(n_neurons=100, dimensions=1)
    K_pred_var = nengo.Ensemble(n_neurons=100, dimensions=2)
    conf_pred = nengo.Ensemble(n_neurons=100, dimensions=1)
    u_pred = nengo.Ensemble(n_neurons=100, dimensions=1)
    
    RS_sens_weighting = nengo.Ensemble(n_neurons=100, dimensions=3)
    RS_pred_weighting = nengo.Ensemble(n_neurons=100, dimensions=3)
    RS_sum = nengo.Ensemble(n_neurons=100, dimensions=1)

    # Equation 1
    nengo.Connection(eye_vel, RS_det, transform=-1)
    nengo.Connection(target_vel, RS_det)
    nengo.Connection(RS_det, RS_noisy, function=lambda x: x+x*di_mult+di_add)
    # Equation 5
    nengo.Connection(conf_sens[0], K_sens_var[0])
    nengo.Connection(RS_sens_hat, K_sens_var[1])
    nengo.Connection(K_sens_var, K_sens, 
                     function=lambda x: x[0]*(1/(x[0]+R+D*(x[0]+x[1]*x[1])*D)))
    # Equation 6
    nengo.Connection(K_sens, conf_sens[1], synapse=0.01, 
                     function=lambda x: 1-x)
    nengo.Connection(conf_sens, conf_sens[0], synapse=0.01, 
                     function=lambda x: 0.01*(Q+OM+x[0]*x[1]))
    # Equation 4
    nengo.Connection(RS_noisy, RS_diff[0])
    nengo.Connection(RS_sens_hat, RS_diff[0], transform=-1)
    nengo.Connection(K_sens, RS_diff[1])
    nengo.Connection(RS_diff, RS_sens_hat, synapse=0.01, 
                     function=lambda x: x[0]*x[1]+nt_sens)
    nengo.Connection(RS_sens_hat, RS_sens_hat, synapse=0.01)
    # Equation 10
    nengo.Connection(target_vel, TV_obs, 
                     function=lambda x: x+x*beta_pred+psi_pred)
    # Equation 11
    nengo.Connection(TV_pred_hat, TV_diff[0], transform=-1)
    nengo.Connection(TV_obs, TV_diff[0])
    nengo.Connection(K_pred, TV_diff[1])
    nengo.Connection(TV_diff, TV_pred_hat, synapse=0.01, 
                     function=lambda x: x[0]*x[1]+epsilon_pred)
    nengo.Connection(TV_pred_hat, TV_pred_hat, synapse=0.01)
    # Equation 12
    nengo.Connection(conf_pred, K_pred_var[0])
    nengo.Connection(TV_pred_hat, K_pred_var[1])
    nengo.Connection(K_pred_var, K_pred, 
                     function=lambda x: x[0]*(1/(x[0]+R2+D2*(x[0]+x[1]*x[1])*D2)))
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

