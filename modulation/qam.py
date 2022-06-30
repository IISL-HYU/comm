import numpy as np 
from modulator import Modulator

class QAM(Modulator):
    def __init__(self, M=16):
        super().__init__(M)
        self.axis_points = self._make_axis_point(M)
        
        # integer 0 ~ M-1
        self.ip = np.arange(M)

        # decimal to binary (array of string)
        self.ipBin = self.dec2bin(self.ip)
        self.symbols = self._make_symbols()

        # Mapping dictionary for M-QAM modulation
        self.mapping = dict(zip(self.ipBin, self.symbols))

        # Demapping dictionary for M-QAM demodulation 
        self.demapping = {v: k for k, v in self.mapping.items()}

 
    def _make_symbols(self):
        # dividing binary into "b0 b1" and "b2 b3"
        ipBin_real = np.array([b[: self.k//2] for b in self.ipBin])
        ipBin_imag = np.array([b[self.k//2 :] for b in self.ipBin])

        # map "b0b1" and "b2b3" to axis_points 
        ipDec_real = self.bin2dec(ipBin_real)
        ipDec_imag = self.bin2dec(ipBin_imag)

        real_points = np.array([self.axis_points[i] for i in ipDec_real])
        imag_points = np.array([self.axis_points[i] for i in ipDec_imag])

        symbols= real_points + 1j * imag_points 
        return symbols
    
    def _make_axis_point(self, M):
        con_min = -(2 * np.sqrt(M) / 2 - 1)
        con_max = (2 * np.sqrt(M) / 2 - 1)
        p_num = int(np.sqrt(M))
        return np.linspace(con_min, con_max, p_num)

    def __call__(self):
        return self.mapping


if __name__ == "__main__":
    qam = QAM(M=16)
    print(qam())