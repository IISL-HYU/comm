###########################################
# Multi-User uplink MIMO with 1-bit ADC   #
# written by Sungweon Hong (Sam Hong)     #
# Hanyang university, Seoul, Korea        #
###########################################


# Import packages
import numpy as np 
import matplotlib.pyplot as plt 
import random
import os 
from estimator import *
from detector import *


class OneBitReceiver():
    
    def __init__(self, 
                 hparam_config=dict()):
        # initialize hyperparameter 
        self.K = hparam_config.get("K", 2) 
        self.N = hparam_config.get("N", 16)
        self.M = hparam_config.get("M", 4) 
        self.T = hparam_config.get("T", 10)

        # snr(dB) scale
        snr_min = hparam_config.get("snr_min", -5)
        snr_max = hparam_config.get("snr_max", 30)
        snr_gap = hparam_config.get("snr_gap", 5)
        self.snr_list = np.arange(snr_min, snr_max+1, snr_gap)

        # 4-qam (QPSK) modulation 
        self.symbols = self.modem()

        # Make symbol space for K-users for in case of ML detection
        self.symbol_space = self.make_symbol_space()

        # select estimator
        self.estimator = zf 
        self.detector = symbol_by_symbol

    def run(self, trials=1, verbose=0):
        for t in range(trials):
            sers = []
            print("_" * 100)
            print("trial: ", t) 
            for snr in self.snr_list:
                # pass through channel 
                H, x, z, r = self.pass_channel(snr)

                # Received signal (Quantized) 
                y = np.sign(r.real) + 1j * np.sign(r.imag) 

                x_hat = self.estimate_and_detect(y, H, snr)

                ser = 1 - np.sum(np.isclose(x, x_hat)) / (self.T * self.K)
                sers.append(ser)
                if verbose == 2:
                    print("snr: ", snr, f"ser: {ser:.2e}")

            if verbose == 1:
                print("length of x:", len(x))
                print("Shape of H, x, z, r, y", H.shape, x.shape, z.shape, r.shape, y.shape)
            print("sers wrt snrs: ", sers)


    def estimate_and_detect(self, y, H, snr):
        x_tilde = self.estimate(y, H, snr)
        x_hat = self.detect(x_tilde)
        return x_hat 

    def estimate(self, y, H, snr):
        return np.matmul(self.estimator(H, snr), y)

    def detect(self, x_tilde):
        return self.detector(x_tilde, self.symbols)

    def pass_channel(self, snr):
        
        x = np.random.randint(0, self.M, size=self.K*self.T).reshape(self.T, self.K, 1)
        x = np.take(self.symbols, x)
        # Channel 
        H = (1 / np.sqrt(2)) * (np.random.randn(self.T, self.N, self.K) + 1j * np.random.randn(self.T, self.N, self.K))
        z = (1 / np.sqrt(2)) * (np.random.randn(self.T, self.N, 1) + 1j * np.random.randn(self.T, self.N, 1))
        # Received signal (real)
        r = np.matmul(H, x) + (10**(-snr/20)) * z 
        return H, x, z, r

    def modem(self, ):
        if self.M == 2: 
            constellation_points = np.array([-1, 1])
        if self.M == 4:
            # without gray coding
            constellation_points = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
            constellation_points /= np.sqrt(2) 
        if self.M == 16: 
            # without gray coding"
            constellation_points = np.array([-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j, -1+3j, 1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j, 3+3j])
            constellation_points /= np.sqrt(10) 
        return constellation_points 

    def make_symbol_space(self,):
        # make symbol space for ML detection
        index_space = np.indices([self.M for _ in range(self.K)]).reshape(self.K, -1).T 
        symbol_space = np.take(self.symbols, index_space)
        return symbol_space 


if __name__ == "__main__":
    hparam_config = dict() 
    hparam_config["K"] = 2
    hparam_config["N"] = 16
    hparam_config["M"] = 4 
    hparam_config["T"] = int(1e5)
    
    receiver = OneBitReceiver(hparam_config=hparam_config)
    print(len(receiver.symbol_space))
    print(receiver.symbols)

    receiver.run(verbose=2) 