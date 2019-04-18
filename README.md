# neural-kalman-filter
Neural Engineering Framework kalman filter for visually guided and predictive pursuit

It turns out the human visual system uses a Kalman filter to smoothly track objects. Here we simulate the neural dynamics in a biologically plausible manner in Python. Then we compare against actual human data, and the latest state-of-the-art numerical MATLAB simulations (non-bioloigcally plausible).

Nengo based model of predictive smooth pursuit dynamics using two Kalman filters to:  
 1) process information of retinal input  
 2) maintain a dynamic representation of target motion

## Paper Abstract

In this paper a biologically plausible implementation of a unified sensory and predictive smooth eye pursuit model is proposed using Nengo. The relevant dynamics are expressed in terms of the Neural Engineering Framework as a network of transformations between neuronal populations. The resulting neural network is compared with the MATLAB implementation of the same model, as well as with experimental animal data. The Nengo implementation performs well against constant target velocity inputs; in the cases of sinusoidally varying target velocity, or an accelerating target, the model performs poorly, resulting in high frequency oscillations and sudden rapid divergences. Further research is recommended to establish the nature of these unwanted dynamic effects, and to explore whether changes may be made to the proposed model to greater exploit NEF dynamics such as intrinsic neuronal noise.

## What is Nengo?

[Nengo is a Python package where you can simulate neural architectures using biologically plausible linear algebra based models.](https://www.nengo.ai/)

## What is a Kalman Filter?

["Linear quadratic estimation" combines noisy observations with a model to eliminate uncertainties - See Wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)

[See my other repository where I implement the Kalman filter in Python and MATLAB!](https://github.com/yazanobeidi/kalman-filter)


[1] Orban de Xivry, et. al., "Kalman Filtering Naturally Accounts for Visually Guided and
Predictive Smooth Pursuit Dynamics" in J. Neuroscience, October 30, 2014. [Online]