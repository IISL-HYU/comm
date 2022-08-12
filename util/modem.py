
# simple modulator for BPSK, QPSK, 16-QAM 
import numpy as np

def modem(M):
    if M == 2: 
        constellation_points = np.array([-1, 1])
    if M == 4:
        # without gray coding
        constellation_points = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
        constellation_points /= np.sqrt(2) 
    if M == 16: 
        # without gray coding"
        constellation_points = np.array([-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j, -1+3j, 1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j, 3+3j])
        constellation_points /= np.sqrt(10) 
    return constellation_points 