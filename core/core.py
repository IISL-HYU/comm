import numpy as np
from scipy.special import erf



def qfunc(x: np.ndarray, type='bpsk'):
    type_available = ('bpsk', 'qpsk', '16-qam')
    
    if type=='16-qam':
        result = np.array([1/2 * np.math.erfc(i/np.sqrt(10)) for i in x])
    else: 
        result = np.array([1/2 * np.math.erfc(i/np.sqrt(2)) for i in x])
    return result

def is_pos_def(mat):
    """
    Function that tells if a matrix is positive (semi)definite
    """
    return np.all(np.linalg.eigvals(mat) >= 0)

def phi(z: np.ndarray):
    """
    CDF of normal distribution
    """
    return (1/2) * (1 + erf(z / np.sqrt(2)))

    
# def compute_ser(qfuncs: np.ndarray, type='bpsk'):
#     type_available = ('bpsk', 'qpsk', '16-qam')

#     if type == 'bpsk':
#         result = qfuncs
#     elif type == 'qpsk':
#         result = 1-(1-qfuncs)**2
#     else:
#         result = 1-(1-(3/2)*qfuncs)**2
#     result = np.average(result)
#     return result