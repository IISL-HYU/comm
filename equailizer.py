
# equalizer # 

import numpy as np
import matplotlib.pyplot as plt 
import random 
import os 


N = 10**5

ip = np.random.randint(0, 2, N) 
s = 2 * ip - 1 # BPSK modulation 

snr_db = np.arange(0, 20) 
simBER_zf = []
simBER_mmse = []
for snr in snr_db: 
    n = 1/np.sqrt(2) * (np.random.normal(0, 1, N) + 1j*np.random.normal(0, 1, N))
    h = 1/np.sqrt(2) * (np.random.normal(0, 1, N) + 1j*np.random.normal(0, 1, N))

    # channel and noise addition 
    y = h * s + 10**(-snr/20)*n 

    # zero forcing Equalization 
    yHat_zf = y / h 
    yHat_mmse = y / (h + .5 * 10 ** (-snr/20))

    ipHat_zf = [int(np.real(x)>0) for x in yHat_zf]
    ipHat_mmse = [int(np.real(x)>0) for x in yHat_mmse]

    simBER_zf.append(np.sum(ip!=ipHat_zf))
    simBER_mmse.append(np.sum(ip!=ipHat_mmse))

simBER_zf = np.array(simBER_zf) / N 
simBER_mmse = np.array(simBER_mmse) / N 

snr = 10 ** (snr_db/10)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.title("ZFE to MMSE _sungweon_Hong")
plt.plot(simBER_zf, 's-', label='ZFE')
plt.plot(simBER_mmse, '.-', label='MMSE Equalizer')
plt.legend()
plt.xlabel("SNR(dB)")
plt.ylabel("BER")
plt.xticks(range(0, 22, 2))
plt.grid()
plt.subplot(2, 1, 2)
plt.semilogy(simBER_zf, 's-', label='ZFE')
plt.semilogy(simBER_mmse, '.-', label="MMSE Equalizer")
plt.xticks(range(0, 22, 2)) 
plt.xlabel("SNR(dB)")
plt.ylabel("BER(dB)")
plt.legend()
plt.grid()
plt.show() 