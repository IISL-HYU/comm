
# computing the SNR improvement in 
# Rayleigh fading channel with selection diversity 
import numpy as np
import matplotlib.pyplot as plt 

# number of bits of symbol
N = 10**4

# Transmitter 
ip = np.random.rand(N) > 0.5 
s = 2 * ip - 1 
