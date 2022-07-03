# Written by Sungweon Hong (Sam) 
# hongsw94@hanyang.ac.kr 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

from modulator import Modulator

class PSK(Modulator):
    def __init__(self, M=2):
        super.__init__(M)

    def modulate(self, bits: np.ndarray):
        pass

    def demodulate(self, symbols: np.ndarray):
        pass 

    
    
    
    