import scipy.interpolate as ip
import numpy as np
from plaw import plaw
import matplotlib.pyplot as plt


def get_errors(magnitudes):
    sigma = 0.02 * np.power(10, 0.2 * (magnitudes - 22))
    new_magnitudes = np.random.normal(scale = sigma, loc=magnitudes, size = magnitudes.shape)
    return sigma, new_magnitudes


def cmdfunc(logage: float, isofile: str, nstars: int, dmodulus: float):
    loga, m, mb, mv = np.loadtxt(isofile, usecols=(0, 1, 8, 9), unpack=True)
    age_indices = np.where(loga == logage)

    mvfunc = ip.interp1d(m[age_indices], mv[age_indices])
    mbfunc = ip.interp1d(m[age_indices], mb[age_indices])
    masses = plaw(nstars, np.min(m[age_indices]), np.max(m[age_indices]), -2.35)
    imv = mvfunc(masses) + dmodulus
    imb = mbfunc(masses) + dmodulus

    s_imv, imv = get_errors(imv)
    s_imb, imb = get_errors(imb)

    indices = np.where(np.sqrt(s_imv**2 + s_imb**2) < 0.8)


    return imb[indices] - imv[indices], imv[indices]
    return imb -imv, imv


bv, v = cmdfunc(9.85, "isoc_z0004.dat", int(6e4), 20.24)
plt.scatter(bv, v, s=0.1, color="black")

bv, v = cmdfunc(8.9, "isoc_z0004.dat", int(1e4), 20.24)
plt.scatter(bv, v, s=0.1, color="black")

bv, v = cmdfunc(9.50, "isoc_z0004.dat", int(2e4), 20.24)
plt.scatter(bv, v, s=0.1, color="black")


#plt.xlim(-0.5, 1.5)
#plt.ylim(26, 17)
plt.xlabel("B-V")
plt.ylabel("V")
plt.gca().invert_yaxis()
plt.show()
