import numpy as np
import matplotlib.pyplot as plt

isochrone_files = ["isoc_z0001.dat", "isoc_z0004.dat", "isoc_z001.dat",
                   "isoc_z004.dat", "isoc_z008.dat", "isoc_z019.dat", "isoc_z030.dat"]

time = 9

for isochrone_file in isochrone_files:
    log_a_isoc, log_b_isoc, log_v_isoc = np.loadtxt(
    isochrone_file, usecols=(0, 8, 9), unpack=True)
    log_bv_isoc = log_b_isoc-log_v_isoc
    indices = np.where(np.abs(log_a_isoc-time) < 0.001)
    plt.plot(log_bv_isoc[indices], log_v_isoc[indices], label=isochrone_file)


plt.gca().invert_yaxis()
plt.legend()
plt.xlabel("B-V")
plt.ylabel("V")
plt.show()

