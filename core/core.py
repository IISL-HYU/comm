import numpy as np
import os 
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

def loadf(fname: str, load_dir='./results/'):
    if load_dir == None:    
        load_dir = "./results/"
    fname = fname
    fdir = os.path.join(load_dir, fname)
    print("\t", fdir)
    return np.load(fdir)
    
def get_all_results(load_dir='./results/'):
    fnames = None 
    for dir_path, dir_names, fname in os.walk(load_dir):
        fnames = fname
    return fnames 

def load_all_results(load_dir='./results/'):
    """
    loads all results and returns its name and value 
    """
    results_key = get_all_results()
    print("results from directory: ")
    results_value = [loadf(i) for i in results_key]
    return results_key, results_value