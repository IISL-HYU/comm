import numpy as np 

class Modulator():
    def __init__(self, M):
        self.M = M
        self.k = int(np.log2(M)) 

    def map(self, bits: np.ndarray):
        pass 

    def demap(self, symbols: np.ndarray):
        pass 

    def dec2bin(self, x: np.ndarray):
        """Decimal array to Binary arrays(string format)"""
        bins = []
        for dec in x:
            bcw = str(bin(dec)[2:]).zfill(self.k)
            bins.append(bcw)
        return np.array(bins)

    def bin2dec(self, bins: np.ndarray):
        """Binary arrays(string format) to decimal arrays"""
        decs = np.zeros_like(bins, dtype=np.int32)
        for i, b in enumerate(bins):
            decs[i] = int(b, 2)
        return decs