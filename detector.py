
import numpy as np 

def symbol_by_symbol(x_tilde, constellation_points):
    compare = np.abs(x_tilde - constellation_points)
    det_by_idx = np.argmin(compare, axis=-1)
    x_hat = np.take(constellation_points, det_by_idx).reshape(x_tilde.shape)
    return x_hat 