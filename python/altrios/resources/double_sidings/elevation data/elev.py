import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


corridor_name = "Clovis-Flagstaff"
origin_elev_meters = 2130

# corridor_name = "Amarillo-FortWorth"
# origin_elev_meters = 1110

# corridor_name = "Barstow-LongBeach"
# origin_elev_meters = 645


dummy_link_length_tt = 10000
step = 10
track_file = f"/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/resources/double_sidings/elevation data/tt_elevation_{corridor_name}_grade_data.xlsx"
usgs_file  = f"/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/resources/double_sidings/elevation data/usgs_elevation_{corridor_name}_grade_data.xlsx"


def clean_first_point(x, elev):
    """Remove invalid first elevation point."""
    if elev[0] == 0:
        return x[1:], elev[1:]
    if abs(elev[1] - elev[0]) > 20:
        return x[1:], elev[1:]
    return x, elev


def remove_dummy(dist, elev, dummy_length=10000):
    """
    Remove dummy link at BOTH start and end.
    Dummy link = long flat (constant elevation) segment.
    """

    # Remove dummy START
    start_elev = elev[0]
    start_idx = 0

    # find first index where elevation changes
    while start_idx < len(elev)-1 and elev[start_idx] == start_elev:
        start_idx += 1

    # if flat region >= dummy_length → remove it
    if dist[start_idx] - dist[0] >= dummy_length:
        dist = dist[start_idx:]
        elev = elev[start_idx:]

    # Remove dummy END
    end_elev = elev[-1]
    end_idx = len(elev) - 1

    # find last index where elevation changes
    while end_idx > 0 and elev[end_idx] == end_elev:
        end_idx -= 1

    # if flat region >= dummy_length → remove it
    if dist[-1] - dist[end_idx] >= dummy_length:
        dist = dist[:end_idx+1]
        elev = elev[:end_idx+1]

    return dist, elev


def compute_up_down(elev_array):
    delta = np.diff(elev_array)
    uphill = np.where(delta > 0, delta, 0)
    downhill = np.where(delta < 0, -delta, 0)
    return np.cumsum(uphill), np.cumsum(downhill)


def elevation_to_sign(elev_array, thr=0.03):
    delta = np.diff(elev_array)
    grade = (delta / step) * 100  # convert to %
    return np.where(grade > thr, 1,
           np.where(grade < -thr, -1, 0))


track_df = pd.read_excel(track_file)
usgs_df  = pd.read_excel(usgs_file)

track_dist = track_df["total_dist_meters"].values
track_elev = track_df["elevation_meters"].values

usgs_dist = usgs_df["total_dist_meters"].values
usgs_elev = usgs_df["elevation_meters"].values


# Clean abnormal first points
track_dist, track_elev = clean_first_point(track_dist, track_elev)
# Clean Track Chart dummy start
usgs_dist,  usgs_elev  = clean_first_point(usgs_dist,  usgs_elev)
track_dist, track_elev = remove_dummy(track_dist, track_elev)


# Resample to common distance axis
max_common_dist = min(track_dist.max(), usgs_dist.max())
common_x = np.arange(0, max_common_dist, step)

track_elev_resamp = np.interp(common_x, track_dist, track_elev)
usgs_elev_resamp  = np.interp(common_x, usgs_dist, usgs_elev)


# Normalize elevation (vertical alignment)
track_elev_norm = track_elev_resamp - track_elev_resamp[0]
usgs_elev_norm  = usgs_elev_resamp - usgs_elev_resamp[0]

# Peak-based horizontal alignment
track_peak_idx = np.argmax(track_elev_norm)
usgs_peak_idx  = np.argmax(usgs_elev_norm)

lag = track_peak_idx - usgs_peak_idx
shift_m = lag * step
print(f"\n Peak horizontal shift applied: {shift_m:.2f} meters  (lag={lag})")

if lag > 0:
    usgs_shifted = np.concatenate([np.full(lag, usgs_elev_norm[0]), usgs_elev_norm[:-lag]])
elif lag < 0:
    usgs_shifted = np.concatenate([usgs_elev_norm[-lag:], np.full(-lag, usgs_elev_norm[-1])])
else:
    usgs_shifted = usgs_elev_norm.copy()


# Vertical alignment based on first bump peak
track_peak_val = track_elev_norm[track_peak_idx]
usgs_peak_val  = usgs_shifted[track_peak_idx]

vertical_offset = track_peak_val - usgs_peak_val
print(f" Vertical peak offset applied: {vertical_offset:.2f} m")

usgs_aligned = usgs_shifted + vertical_offset
usgs_elev_norm = usgs_aligned

track_elev_norm += origin_elev_meters
usgs_elev_norm  += origin_elev_meters


# Plot aligned Elevation Profile
plt.figure(figsize=(12, 5))
plt.plot(common_x/1000, track_elev_norm, label="Track Chart", linewidth=1.5)
plt.plot(common_x/1000, usgs_elev_norm, label="USGS DEM (filtered)", linewidth=1.2)
plt.xlabel("Distance (km)")
plt.ylabel("Elevation (m)")
plt.title(f"Elevation Profile Comparison {corridor_name} (Peak-Aligned)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute Uphill/Downhill
track_up, track_down = compute_up_down(track_elev_norm)
usgs_up,  usgs_down  = compute_up_down(usgs_elev_norm)
plot_x = common_x[1:] / 1000


# Summary Metrics
track_sign = elevation_to_sign(track_elev_norm)
usgs_sign  = elevation_to_sign(usgs_elev_norm)
mismatch_ratio = np.mean(track_sign != usgs_sign)

print("\n========== Summary ==========")
print(f"Total Uphill (Track Chart): {track_up[-1]:.2f} m")
print(f"Total Uphill (USGS):        {usgs_up[-1]:.2f} m")
print(f"Total Downhill (Track):    {track_down[-1]:.2f} m")
print(f"Total Downhill (USGS):     {usgs_down[-1]:.2f} m")
print(f"Grade Sign Mismatch Ratio: {mismatch_ratio*100:.2f}%")
print("================================\n")