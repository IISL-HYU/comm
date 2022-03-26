# 
# Rayleigh channel fading Jake's Model 
# Written by Sam (sungweon) Hong 
# 
# @ Hanyang university, Information and Intelligence Systems laboratory
# contact me : hongsw94@hanyang.ac.kr

# Import relevant Packages 

import numpy as np 
import random 
from numpy import real, imag 
from math import pi 

## for visualizing 
import matplotlib.pyplot as plt 


def channel_by_time(
    velocity=10,
    frequency=1e9,
    time_steps=np.linspace(0, 1, 1000),
    alpha=pi/6,
    theta_o=0,
    num_rays=12,):
    num_osc = ((num_rays//2 - 1) // 2 ) + 1 
    beta = np.array([(pi * (n+1))/num_osc for n in range(num_osc)])
    theta_n = np.array([beta[i] + (2*pi*i)/num_osc for i in range(num_osc)])
    max_dopp_shift, dopp_shift = doppler_shift(velocity=velocity, 
                                               frequency=frequency, 
                                               num_rays=num_rays,)

    model = (1/np.sqrt(2)) * np.e**(alpha*1j) * np.cos(max_dopp_shift * time_steps + theta_o)
    tmodel = sum([np.e ** (beta[i] * 1j) * np.cos(dopp_shift[i] * time_steps + theta_n[i]) for i in range(num_osc)])

    model = model + tmodel
    return model

def doppler_shift(
    velocity=3,
    frequency=1e9, 
    num_rays=12,
    randray=False, 
    verbose=0,):
    c = 3e8
    alphas = []

    max_dopp_shift = 2 * pi * frequency * velocity / c
    
    if randray:
        alphas = np.array(2 * pi * np.random.uniform(0, 1, num_rays))
    else: 
        alphas = np.array([(2 * pi * n)/num_rays for n in range(num_rays)])
    
    shifts = max_dopp_shift * np.cos(alphas)

    return max_dopp_shift, shifts


if __name__ == "__main__":
    np.random.seed(123454)
    v = 30; f = 1e9; n_rays = 12; 
    num_samples = 1000; num_channels = 5; timespan = .2
    t_channels = [] 

    for _ in range(num_channels):
        seed = np.random.randint(1e10)
        channel = channel_by_time(velocity=v,
                                  frequency=f,
                                  num_rays=n_rays,
                                  time_steps=np.linspace(seed, seed+timespan, num_samples))
        t_channels.append(channel)
    
    plt.figure(figsize=(8, 8))
    
    plt.suptitle(f"{num_channels} different correlated rayleigh channel", fontsize=15)    

    for n in range(num_channels):
        plt.subplot(num_channels, 1, n+1)
        plt.plot(np.linspace(0, timespan, num_samples), 10 * np.log10(real(t_channels[n])**2))
        plt.ylabel("signal level (dB)")
        plt.grid()
    plt.xlabel("time (t)")
    plt.show() 
    
