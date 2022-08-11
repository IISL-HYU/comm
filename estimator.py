

import numpy as np 

def zf(H, snr):
    return np.linalg.pinv(H)

def mmse(H, snr):
    T = H.shape[0]
    K = H.shape[-1]
    estimator = np.matmul((H.conjugate()).transpose(0, 2, 1), H) + (10**(-snr/20)) * ((np.eye(K) + 1j * np.eye(K)) * np.ones(shape=(T, K, K)))
    estimator = np.linalg.inv(estimator)
    estimator = np.matmul(estimator, (H.conjugate()).transpose(0, 2, 1))
    return estimator