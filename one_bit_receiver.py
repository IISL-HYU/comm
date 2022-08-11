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
from modem import * 


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
        self.symbols = modem(self.M)

        # Make symbol space for K-users for in case of ML detection
        self.symbol_space = make_symbol_space(M=self.M, K=self.K, constellation_points=self.symbols)

        # select estimator
        self.estimator = hparam_config.get("estimator", zf)
        self.detector = hparam_config.get("detector", symbol_by_symbol)

    def run(self, trials=1, verbose=0):
        sers_avg = np.zeros_like(self.snr_list, dtype=np.float64)
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
            sers_avg += np.array(sers)
            if verbose == 1:
                print("length of x:", len(x))
                print("Shape of H, x, z, r, y", H.shape, x.shape, z.shape, r.shape, y.shape)
        return sers_avg / trials

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

    def estimate_and_detect(self, y, H, snr):
        x_tilde = self.estimate(y, H, snr)
        x_hat = self.detect(x_tilde)
        return x_hat 

    def estimate(self, y, H, snr):
        return np.matmul(self.estimator(H, snr), y)

    def detect(self, x_tilde):
        return self.detector(x_tilde, self.symbols)



if __name__ == "__main__":
    hparam_config = dict() 
    hparam_config["K"] = 2
    hparam_config["N"] = 16
    hparam_config["M"] = 4 
    hparam_config["T"] = int(1e5)
    hparam_config["estimator"] = zf
    hparam_config["detector"] = symbol_by_symbol
    zf_receiver = OneBitReceiver(hparam_config=hparam_config)
    
    hparam_config["estimator"] = mmse
    mmse_receiver = OneBitReceiver(hparam_config=hparam_config)

    sers_avg_zf = zf_receiver.run(trials=1, verbose=2) 
    sers_avg_mmse = zf_receiver.run(trials=1, verbose=2) 

    plt.figure(figsize=(8, 8))
    plt.semilogy(zf_receiver.snr_list, sers_avg_zf, '-ro', label='ZF', markersize=12, fillstyle='none')
    plt.semilogy(mmse_receiver.snr_list, sers_avg_mmse, '-r*', label='MMSE', markersize=12)
    plt.grid()
    plt.legend()
    plt.xlabel("SNR")
    plt.ylabel("SER")
    plt.title("symbol error rate")
    plt.yticks([1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    plt.show()