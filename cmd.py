import numpy as np
import matplotlib.pyplot as plt

logv_m45, logbv_m45 = np.loadtxt("m45.dat", usecols=(2, 3), unpack=True)
logy_47tuc, logby_47tuc, s_log_y_47tuc, s_log_by_47tuc = np.loadtxt(
    "47tuc.dat", usecols=(3, 4, 6, 7), unpack=True)
logv_47tuc = logy_47tuc
logbv_47tuc = 0.055 + 1.707*logby_47tuc
tuc_indices = np.where(s_log_by_47tuc < 0.03)


def closest_distance_square(point, points):
    dist_2 = np.sum((points-point)**2, axis=1)
    return np.power(np.min(dist_2), 1)


def sum_square_closest_distance(points, isochrone):
    sum = 0
    for point in points[::1]:
        sum += closest_distance_square(point, isochrone)
    return sum


def find_closest_isochrone(data, min_distance=5, max_distance=6):
    isochrone_files = ["isoc_z0001.dat", "isoc_z0004.dat", "isoc_z001.dat",
                       "isoc_z004.dat", "isoc_z008.dat", "isoc_z019.dat", "isoc_z030.dat"]
    best_error = float("inf")
    best_isochrone = None
    best_time = None
    best_metalicity = None
    best_x = None
    for isochrone_file in isochrone_files:
        log_a_isoc, log_b_isoc, log_v_isoc = np.loadtxt(
            isochrone_file, usecols=(0, 8, 9), unpack=True)
        log_bv_isoc = log_b_isoc-log_v_isoc
        times = np.arange(np.min(log_a_isoc), np.max(log_a_isoc), 0.05)
        print(isochrone_file)
        for time in times:
            for x in np.linspace(min_distance, max_distance, 10):
                indices = np.where(np.abs(log_a_isoc-time) < 0.001)
                b_v = log_bv_isoc[indices]
                v = log_v_isoc[indices]+x
                if len(v) == 0:
                    continue
                isochrone = np.array([v, b_v]).transpose()
                error = sum_square_closest_distance(data, isochrone)
                if error < best_error:
                    best_error = error
                    best_isochrone = isochrone
                    best_time = time
                    best_metalicity = isochrone_file
                    best_x = x
                    print(time, isochrone_file, error, x)
    print(f"Distance: {10**((best_x/5)+1)}")

    return best_isochrone, best_time, best_metalicity, best_x

data_m45 = np.array([logv_m45, logbv_m45]).transpose()
isochrone_m45, time_m45, metallicity_m45, distance_modulus_m45 = find_closest_isochrone(
    data_m45)
isochrone_m45 = isochrone_m45.transpose()
print(f"Time: {time_m45}, Metallicity: {metallicity_m45}, Distance: {distance_modulus_m45}")
# plt.plot(isochrone_m45[1], isochrone_m45[0]-distance_modulus_m45, label="Best Fit for M45")
# plt.scatter(logbv_m45, logv_m45-distance_modulus_m45, label="M45")


data_47tuc = np.array([logv_47tuc, logbv_47tuc]).transpose()
isochrone_47tuc, time_47tuc, metallicity_47tuc, distance_modulus_47tuc = find_closest_isochrone(
    data_47tuc, 12, 14)
isochrone_47tuc = isochrone_47tuc.transpose()
print(f"Time: {time_47tuc}, Metallicity: {metallicity_47tuc}, Distance: {distance_modulus_47tuc}")
plt.plot(isochrone_47tuc[1], isochrone_47tuc[0]-distance_modulus_47tuc, label="Best Fit for 47 TUC", color="orange")
plt.scatter(logbv_47tuc[tuc_indices],
            logv_47tuc[tuc_indices]-distance_modulus_47tuc, label="47TUC")


plt.legend()
plt.gca().invert_yaxis()
plt.show()
