from core.core import *
import numpy as np 

def make_symbol_space(M, K, constellation_points):
    # make symbol space for ML detection
    index_space = np.indices([M for _ in range(K)]).reshape(K, -1).T 
    symbol_space = np.take(constellation_points, index_space)
    return symbol_space 

def symbol_by_symbol(x_tilde, constellation_points):
    compare = np.abs(x_tilde - constellation_points)
    det_by_idx = np.argmin(compare, axis=-1)
    x_hat = np.take(constellation_points, det_by_idx).reshape(x_tilde.shape)
    return x_hat

def ml(y_re, H_re, symbol_space, snr_dB):
    # complex to real and then reshape for broadcast
    snr = 10**(snr_dB/20)
    T = H_re.shape[0]
    M = symbol_space.shape[-1]
    K = H_re.shape[-1]
    x_shape = tuple(T, K, 1)
    symbol_space_re = np.hstack((symbol_space.real, symbol_space.imag)) * np.ones(shape=(T, M**K, 2*K))
    likelihood = phi(np.matmul(symbol_space_re, H_re.transpose(0, 2, 1)) * y_re.transpose(0, 2, 1) * np.sqrt(2 * snr))
    log_likelihood = np.sum(np.log(likelihood), axis=2)
    x_hat_idx = np.argmax(log_likelihood, axis=1)
    x_hat_ML = np.take(symbol_space, x_hat_idx, axis=0)
    x_hat_ML = x_hat_ML.reshape(x_shape)
    return x_hat_ML 