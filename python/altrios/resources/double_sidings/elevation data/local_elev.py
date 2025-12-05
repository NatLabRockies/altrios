import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===============================================================
# User settings
# corridor_name = "Clovis-Flagstaff"; origin_elev_meters = 1300
# corridor_name = "Barstow-LongBeach"; origin_elev_meters = 645
# corridor_name = "Amarillo-FortWorth"; origin_elev_meters = 1110
# ===============================================================

corridor_name = "Amarillo-FortWorth"
origin_elev_meters = 1110

step = 10
dummy_link_length_tt = 10000

track_file = f"/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/resources/double_sidings/elevation data/tt_elevation_{corridor_name}_grade_data.xlsx"
usgs_file  = f"/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/resources/double_sidings/elevation data/usgs_elevation_{corridor_name}_grade_data.xlsx"

# ===============================================================
# Utility functions
# ===============================================================
def clean_first_point(x, elev):
    """Remove abnormal first elevation value."""
    if elev[0] == 0 or abs(elev[1] - elev[0]) > 20:
        return x[1:], elev[1:]
    return x, elev


def remove_dummy(dist, elev, dummy_length=10000):
    """Remove long flat dummy link segments at both start and end."""
    # start
    start_e = elev[0]
    idx = 0
    while idx < len(elev)-1 and elev[idx] == start_e:
        idx += 1
    if dist[idx] - dist[0] >= dummy_length:
        dist, elev = dist[idx:], elev[idx:]

    # end
    end_e = elev[-1]
    idx = len(elev)-1
    while idx > 0 and elev[idx] == end_e:
        idx -= 1
    if dist[-1] - dist[idx] >= dummy_length:
        dist, elev = dist[:idx+1], elev[:idx+1]

    return dist, elev


def compute_up_down(elev_array):
    delta = np.diff(elev_array)
    uphill = np.where(delta > 0, delta, 0)
    downhill = np.where(delta < 0, -delta, 0)
    return np.cumsum(uphill), np.cumsum(downhill)


def elevation_to_sign(elev_array, thr=0.03):
    delta = np.diff(elev_array)
    grade = (delta / step) * 100   # convert to %
    return np.where(grade > thr, 1, np.where(grade < -thr, -1, 0))


def find_dominant_peak(elev, prominence=20):
    """Detect dominant local peak (most prominent)."""
    peaks, props = find_peaks(elev, prominence=prominence)
    if len(peaks) == 0:
        return np.argmax(elev)  # fallback: global max
    return peaks[np.argmax(props["prominences"])]


# Load data
track_df = pd.read_excel(track_file)
usgs_df  = pd.read_excel(usgs_file)

track_dist = track_df["total_dist_meters"].values
track_elev = track_df["elevation_meters"].values

usgs_dist = usgs_df["total_dist_meters"].values
usgs_elev = usgs_df["elevation_meters"].values

# clean
track_dist, track_elev = clean_first_point(track_dist, track_elev)
usgs_dist,  usgs_elev  = clean_first_point(usgs_dist,  usgs_elev)
track_dist, track_elev = remove_dummy(track_dist, track_elev)


# Resample to common X grid
max_common = min(track_dist.max(), usgs_dist.max())
common_x = np.arange(0, max_common, step)

track_elev_resamp = np.interp(common_x, track_dist, track_elev)
usgs_elev_resamp  = np.interp(common_x, usgs_dist, usgs_elev)

# Normalize
track_norm = track_elev_resamp - track_elev_resamp[0]
usgs_norm  = usgs_elev_resamp  - usgs_elev_resamp[0]


# ===============================================================
# Peak-based alignment (robust)
# ===============================================================
track_peak = find_dominant_peak(track_norm)
usgs_peak  = find_dominant_peak(usgs_norm)

lag = track_peak - usgs_peak
shift_m = lag * step
print(f"\nHorizontal shift (dominant peak align) = {shift_m:.1f} m (lag={lag} points)\n")

# shift USGS horizontally
if lag > 0:
    usgs_shifted = np.concatenate([np.full(lag, usgs_norm[0]), usgs_norm[:-lag]])
elif lag < 0:
    usgs_shifted = np.concatenate([usgs_norm[-lag:], np.full(-lag, usgs_norm[-1])])
else:
    usgs_shifted = usgs_norm.copy()

# Vertical alignment
v_offset = track_norm[track_peak] - usgs_shifted[track_peak]
print(f"Vertical peak offset = {v_offset:.2f} m\n")

usgs_aligned = usgs_shifted + v_offset


# Apply absolute origin elevation
track_final = track_norm + origin_elev_meters
usgs_final  = usgs_aligned + origin_elev_meters


# ===============================================================
# Plot final aligned elevations
# ===============================================================
plt.figure(figsize=(12,5))
plt.plot(common_x/1000, track_final, label="Track Chart", lw=1.5)
plt.plot(common_x/1000, usgs_final,  label="USGS DEM (aligned)", lw=1.3)
plt.xlabel("Distance (km)")
plt.ylabel("Elevation (m)")
plt.title(f"Elevation Profile Comparison {corridor_name} (Dominant-Peak Aligned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ===============================================================
# Summary statistics
# ===============================================================

track_up, track_down = compute_up_down(track_final)
usgs_up,  usgs_down  = compute_up_down(usgs_final)

track_sign = elevation_to_sign(track_final)
usgs_sign  = elevation_to_sign(usgs_final)
mse = np.mean((usgs_final - track_final)**2)

mismatch = np.mean(track_sign != usgs_sign)

print("============= SUMMARY =============")
print(f"Elevation MSE (USGS vs Track Chart) = {mse:.2f}")
print(f"Total Uphill (Track Chart): {track_up[-1]:.1f} m")
print(f"Total Uphill (USGS DEM):    {usgs_up[-1]:.1f} m")
print(f"Total Downhill (Track):     {track_down[-1]:.1f} m")
print(f"Total Downhill (USGS):      {usgs_down[-1]:.1f} m")
print(f"Grade Sign Mismatch:        {mismatch*100:.2f} %")
print("===================================\n")
