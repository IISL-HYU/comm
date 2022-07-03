import numpy as np 
import matplotlib.pyplot as plt
from modulator import Modulator

class QAM(Modulator):
    def __init__(self, M=16):
        # inherit Modulator class 
        super().__init__(M)
        
        self.axis_points = self._make_axis_point(M)
        
        # integer 0 ~ M-1
        self.ip = np.arange(M)

        # decimal to binary (array of string)
        self.ipBin = self.dec2bin(self.ip)
        self.symbols = self._make_symbols()

        # Mapping dictionary for M-QAM modulation
        self.mapping = dict()
        for b, s in zip(self.ipBin, self.symbols):
            self.mapping[tuple(b)] = s
        self.demapping = {v: k for k, v in self.mapping.items()}
        # symbol power normalization (E[|x|^2] = 1)
        if self.normalize:
            self.avg_power = self._compute_avg_power()
            self.mapping = {k: v/self.avg_power for k, v in self.mapping.items()}
        else: 
            self.avg_power = 1

        # Demapping dictionary for M-QAM demodulation 

    def modulate(self, bits: np.ndarray):
        idx = np.arange(len(bits) // self.k, dtype=np.int32)
        symstream = np.zeros(shape=(len(idx),), dtype=np.complex64)

        for i in idx:
            sym = tuple(bits[self.k * i : self.k * (i+1)])
            symstream[i] = self.mapping[sym]
        return symstream
    
    def demodulate(self, symbols: np.ndarray):
            
        symbols *= self.avg_power

        rounds = np.fromiter(self.demapping.keys(), dtype=np.complex64)
        rounds = np.unique(rounds) 

        compare = np.subtract.outer(symbols, rounds)
        s_hat = rounds[np.argmin(abs(compare), axis=1)]

        messages_recovered = self.demap(s_hat)
        return messages_recovered

    def demap(self, symbols: np.ndarray):
        """Returns an 1-dimensional array of bitstream"""
        bitstream_recovered = []
        demap = self.get_demap()
        for sym in symbols:
            bitstream_recovered.append(list(demap[sym]))
        return np.array(bitstream_recovered).reshape(-1, 1).squeeze()

    def plot_constellation_points(self, ):
        plt.figure(figsize=(8, 8))
        for b3 in [0, 1]:
            for b2 in [0, 1]:
                for b1 in [0, 1]:
                    for b0 in [0, 1]:
                        B = (b3, b2, b1, b0) 
                        Q = self.mapping[B]
                        plt.plot(Q.real, Q.imag, 'bo', zorder=10)
                        plt.text(Q.real, Q.imag+0.1, "".join(str(x) for x in B), ha='center')
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.grid(True, linestyle="--")
        plt.title(f"Constellation Mapping for {self.M}-QAM")

 
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

    def _compute_avg_power(self,):
        avgpower = [abs(i)**2 for i in self.mapping.values()]
        avgpower = np.sqrt(np.average(avgpower))
        return avgpower


if __name__ == "__main__":
    qam = QAM(M=16)
    mapping_table = qam.get_map()
    demapping_table = qam.get_demap() 

    print(mapping_table) 

    print(demapping_table)