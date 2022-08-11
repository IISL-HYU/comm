
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