import numpy as np
import matplotlib.pyplot as plt

time_slot = list(range(100))
coeff = []

for t in time_slot:
    constell_points = np.sqrt(0.5)*np.random.normal(size=2).reshape(-1, 1) + 1j * np.sqrt(.5)*np.random.normal(size=2).reshape(-1, 1)
    coeff.append(np.linalg.norm(constell_points, axis=1).reshape(-1, 1))

coeff = np.array(coeff)


plt.figure(figsize=(12, 5))
plt.plot(time_slot, coeff[:, 0], '-b', label="channel_1", lw=0.5)
plt.plot(time_slot, coeff[:, 1], 'r', label="channel_2", lw=0.5)
plt.grid()
plt.legend()
plt.show()