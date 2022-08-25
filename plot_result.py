import numpy as np 
import matplotlib.pyplot as plt 
import os 

from core.core import * 
from util.detector import * 
from util.estimator import * 
from util.modem import * 
from util.theoretic import * 


snr_lst = np.arange(-10, 31, 5)
_, results = load_all_results()


plt.figure(figsize=(9, 8))
# plt.semilogy(snr_lst[1:], results[2], '-s', )
plt.semilogy(snr_lst[:-1], results[3], '-rs', markersize=10, label='One-Stage ML')
plt.semilogy(snr_lst[:-1], results[4], '-bs', markersize=10, label='Original ML')
plt.xlim([-10, 20]); plt.ylim([1e-6, 1e0])
plt.xlabel("SNR(dB)", fontsize=15); plt.ylabel("SER", fontsize=15); plt.title("Symbol-Error-Rate for nML vs ML", fontsize=20)
plt.grid(); plt.legend(fontsize=15)
plt.show()