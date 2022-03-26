import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfc

# for reproducibility 
np.random.seed(1234)

mc_len = 10000
SNR_db = np.linspace(0, 25, 11)
SNR = 10 ** (0.1 * SNR_db)

# AWGN Channel 
p_awgn = .5 * erfc(np.sqrt(SNR/2))
p_awgn
p_his = []

for i in range(mc_len):

    constell_points = np.sqrt(0.5)*np.random.normal(size=2).reshape(-1, 1) + 1j * np.sqrt(.5)*np.random.normal(size=2).reshape(-1, 1)
    h1, h2 = np.linalg.norm(constell_points, axis=1)

    snr_fading = SNR * (h1**2)
    snr_antenna = SNR * max(h1**2, h2**2)
    snr_mrc = SNR * (h1**2 + h2**2)
    snrs = np.array([snr_fading, snr_antenna, snr_mrc])
    p_chs = .5 * erfc(np.sqrt(snrs/2))
    p_his.append(p_chs)

p_chs = np.average(p_his, axis=0)
print(p_chs[1])

lw = 1

plt.figure(figsize=(9, 5))
# plt.semilogy(SNR_db, p_awgn, 'g', label="AWGN")
plt.semilogy(SNR_db, p_chs[0], 'b', label="1-Rx Fading", lw=lw)
plt.semilogy(SNR_db, p_chs[1], 'k', label="2-Rx Antenna selection", lw=lw)
plt.semilogy(SNR_db, p_chs[2], 'r', label="2-Rx MRC", lw=lw)
plt.xlabel("SNR_db")
plt.ylabel("Error Probability")
plt.title("Error Probabilty")
plt.legend()
plt.grid()
plt.show()
