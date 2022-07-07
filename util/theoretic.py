import numpy as np 
from core.core import *

def compute_ser(qfuncs: np.ndarray, type='bpsk'):
    type_available = ('bpsk', 'qpsk', '16-qam')

    if type == 'bpsk':
        result = qfuncs
    elif type == 'qpsk':
        result = 1-(1-qfuncs)**2
    else:
        result = 1-(1-(3/2)*qfuncs)**2
    result = np.average(result)
    return result