
import numpy as np 
import matplotlib.pyplot as plt 
import os 

from one_bit_receiver import OneBitReceiver
from util.detector import * 
from util.estimator import * 
from util.modem import * 


class BussgangReceiver(OneBitReceiver):
    def __init__(self, hparam_config=dict()):
        super().__init__(hparam_config=hparam_config)
    
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

                A, covn = self.decompose(H, snr)

                x_hat = self.estimate_and_detect(y, A, covn)

                ser = 1 - np.sum(np.isclose(x, x_hat)) / (self.T * self.K)
                sers.append(ser)
                if verbose == 2:
                    print("snr: ", snr, f"ser: {ser:.2e}")
            sers_avg += np.array(sers)
            if verbose == 1:
                print("length of x:", len(x))
                print("Shape of H, x, z, r, y", H.shape, x.shape, z.shape, r.shape, y.shape)
        return sers_avg / trials
    
    def cov_r(self, H, snr):  
        """
        Method that Calculates the covariance matrix of the received signal 
        """   
        if len(H.shape) > 2:
            return np.matmul(H, (H.conjugate()).transpose(0, 2, 1)) + \
                (10 ** (-snr/20)) * (np.eye(self.N)) * np.ones(shape=(self.T, self.N, self.N))
        else: 
            return np.matmul(H, (H.conjugate()).T) + \
                (10**(-snr/20))*(np.eye(self.N))

    def decompose(self, H, snr):
        sigma_r = self.cov_r(H, snr)
        diag_sigma = sigma_r * ((np.eye(self.N) * np.ones(shape=(self.T, self.N, self.N))))
        inv_diag_sigma = np.linalg.inv(diag_sigma)
        sqrt_diag_sigma = np.sqrt(inv_diag_sigma)

        # 1. Calculate the effective channel
        self.V = sqrt_diag_sigma * np.sqrt(2 / np.pi) 
        effective_channel = np.matmul(self.V, H)
        
        # 2. Calculate the effective noise 
        lhs = np.matmul(sqrt_diag_sigma, sigma_r) 
        rhs1 = np.matmul(lhs, sqrt_diag_sigma) * 0.9999 # for preventing nan 
        lhs = np.arcsin(rhs1.real) + 1j * np.arcsin(rhs1.imag) 

        rhs = rhs1 - (10 ** (-snr/20)) * inv_diag_sigma
        effective_noise_covariance = np.sqrt(2 / np.pi) * (lhs-rhs) 

        return effective_channel, effective_noise_covariance


        

if __name__ == "__main__":
    sers_avg_zf = np.load('./results/sers_avg_bmmse.npy')

    save_dir = './results/'
    hparam_config = dict() 
    hparam_config["K"] = 2
    hparam_config["N"] = 16
    hparam_config["M"] = 4 
    hparam_config["T"] = int(1e5)
    hparam_config["estimator"] = bmmse
    hparam_config["detector"] = symbol_by_symbol
    bzf_receiver = BussgangReceiver(hparam_config=hparam_config)
    
    sers_avg_zf += bzf_receiver.run(trials=10, verbose=2) 
    sers_avg_zf /= 2
    
    np.save(save_dir + 'sers_avg_bmmse.npy', sers_avg_zf)

    plt.figure(figsize=(9, 8))
    plt.semilogy(bzf_receiver.snr_list, sers_avg_zf, '-ko', label='BZF', markersize=12, fillstyle='none')

    plt.grid()
    plt.legend()
    plt.xlabel("SNR")
    plt.ylabel("SER")
    plt.title("symbol error rate")
    plt.yticks([1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    plt.show()
    