# Modulation Class (should be inherited)
import numpy as np 

class Modulator():
    def __init__(self, M, normalize=True):
        self.M = M
        self.k = int(np.log2(M)) 
        self.normalize = normalize
        self.mapping = dict()
        self.demapping = dict()

    def modulate(self, bits: np.ndarray):
        """
        This method should be overriden
        """
        pass 

    def demodulate(self, symbols: np.ndarray):
        """
        This method should be overriden
        """
        pass 

    def dec2bin(self, x: np.ndarray):
            """Decimal array to Binary arrays(string format)"""
            bins = []
            for dec in x:
                bcw = list(str(bin(dec)[2:]).zfill(self.k))
                bins.append(bcw)
            return np.array(bins, dtype=np.int32)

    def bin2dec(self, bins: np.ndarray):
        """Binary arrays(string format) to decimal arrays"""
        decs = np.zeros(shape=(len(bins)), dtype=np.int32)
        for i, b in enumerate(bins):
            b = ''.join(str(bit) for bit in b)
            decs[i] = int(b, 2)
        return decs

    def get_map(self):
        return self.mapping
    
    def get_demap(self):
        return self.demapping

    def __call__(self):
        return self.mapping