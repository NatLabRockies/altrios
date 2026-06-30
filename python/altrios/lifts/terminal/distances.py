"""Yard geometry and travel-time/speed sampling for trucks and hostlers."""
from scipy.stats import triang
import math
import yaml
import pandas as pd
from pathlib import Path

from altrios.lifts.terminal.utilities import resources_root


def calculate_distances(config_path="input/config.yaml", config=None, actual_railcars=None):
    """Compute yard geometric distances for a given site config.

    Distances are derived from a fixed or adaptive yard layout
    (rows/cols/track-lanes/parking-lanes/block-length) plus a
    ``yard:`` section that fixes per-railcar geometry. If
    ``actual_railcars`` is provided, the railcar-utilisation factor
    ``mu`` is computed from the current track occupancy; otherwise it
    is initialised to ``1.0`` (idle / fully reserved track).

    Parameters
    ----------
    config_path : str, optional
        Path to a YAML file containing ``yard`` / ``layout`` /
        ``simulation`` sections. Used only when ``config`` is ``None``.
    config : Mapping, optional
        Pre-parsed configuration mapping. Takes precedence over
        ``config_path`` when provided.
    actual_railcars : int, optional
        Current railcar count on the longest track. When ``None``,
        the function returns the idle-state distance dictionary.

    Returns
    -------
    dict
        Mapping with keys ``M``, ``N``, ``n_t``, ``n_p``, ``n_r``,
        ``P``, ``yard_length``, ``total_lane_length``,
        ``railcar_length``, ``n_max``, ``n``, ``mu``, and min / avg
        (or mean) / max distances for hostler (``d_h_*``),
        repositioning (``d_r_*``), truck (``d_t_*``), and inter-track
        (``d_tr_*``) travel.

    Raises
    ------
    ValueError
        If no row matches the requested batch size in adaptive mode,
        or if ``yard.yard_type`` is neither ``'parallel'`` nor
        ``'perpendicular'``.
    """

    if (config is None) and (config_path is not None):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    # Otherwise: just the config that's passed in from the terminal object

    yard_cfg = config["yard"]
    layout_cfg = config["layout"]

    # --- layout (adaptive or fixed) ---
    mode = layout_cfg.get("mode", "adaptive").lower()
    if mode == "adaptive":
        df = pd.read_excel(resources_root() / "multiple_layout.xlsx")
        batch_size = config["simulation"]["train_batch_size"]
        row = df.loc[df["train batch (k)"] == batch_size]
        if row.empty:
            raise ValueError(f"No layout found for train batch size {batch_size}")
        row = row.iloc[0]
        M, N, n_t, n_p, n_r = int(row["rows (M)"]), int(row["cols (N)"]), int(row["trainlanes (n_t)"]), int(row["parknglanes (n_p)"]), int(row["blocklen (n_r)"])
    else:
        M, N, n_t, n_p, n_r = int(layout_cfg["M"]), int(layout_cfg["N"]), int(layout_cfg["n_t"]), int(layout_cfg["n_p"]), int(layout_cfg["n_r"])

    # --- yard basic parameters ---
    P = 10
    BL_l = 10 * n_r
    BL_w = 80
    A_yard = M * BL_l + (M + 1) * n_p * P  # yard length
    B_yard = N * BL_w + (N + 1) * n_p * P        # yard width
    total_lane_length = A_yard * (N + 1) + B_yard * (M + 1)

    # --- geometry from yard config ---
    railcar_length = float(yard_cfg["railcar_length"])
    d_f = float(yard_cfg["d_f"])
    d_x = float(yard_cfg["d_x"])

    # railcar and track
    # vary
    n_max = math.ceil(A_yard / railcar_length)
    n = 0 if actual_railcars is None else int(actual_railcars)
    mu = 1 if n == 0 else min(1.0, (n * railcar_length) / A_yard)
    # fixed
    YARD_TYPE = yard_cfg["yard_type"]  # choose 'perpendicular' or 'parallel'
    l_c = 60         # The length of a railcar and joint (ft)

    if YARD_TYPE == 'parallel':
        # d_h: hostler distance
        d_h_min = n_t * P + 1.5 * n_p * P
        d_h_max = n_t * P + A_yard - n_p * P + B_yard - n_p * P
        d_h_avg = (d_h_max + d_h_min) / 2

        # d_r: repositioning distance
        d_r_min = 5 * n_r + 40
        d_r_max = ugly_sigma(M) * (10 * n_r + n_p * P) + ugly_sigma(N) * (80 + n_p * P)
        d_r_avg = (d_r_max + d_r_min) / 2

        # d_t: truck distance
        d_t_min = 0.5 * n_p * P
        d_t_max = B_yard - n_p * P + A_yard - n_p * P
        d_t_avg = (d_t_max + d_t_min) / 2

        # d_tr: inter-track distance
        term = max(0, ((mu - 0.5) * (1 - mu)) / mu)
        d_tr_min = term * n_max * l_c + ((n - 1) / 2) * l_c + d_f + d_x + 0.5 * n_p * P
        d_tr_mean = 0.5 * l_c + d_f + d_x + 0.5 * n_p * P
        d_tr_max = n_max * l_c + d_f + d_x + 0.5 * n_p * P

    elif YARD_TYPE == 'perpendicular':
        d_h_min = n_t * P + 1.5 * n_p * P
        d_h_avg = 10 * n_r * M + 80 * N + (M + N + 1.5) * n_p * P + 2 * n_t * P
        d_h_max = n_t * P + A_yard - n_p * P + B_yard - n_p * P

        d_r_min = 0
        d_r_avg = 5 * n_r + 40 + ugly_sigma(M) * (10 * n_r + n_p * P) + ugly_sigma(N) * (80 + n_p * P)
        d_r_max = 10 * n_r + 80 + A_yard - n_p * P + B_yard - n_p * P

        d_t_min = 1.5 * n_p * P
        d_t_avg = 0.5 * (B_yard + A_yard - 0.5 * n_p * P)
        d_t_max = B_yard + A_yard - 2 * n_p * P

        d_tr_min = 0.5 * l_c + d_f + d_x + 0.5 * n_p * P
        term = max(0, ((mu - 0.5) * (1 - mu)) / mu)
        d_tr_mean = term * n_max * l_c + ((n - 1) / 2) * l_c + d_f + d_x + 0.5 * n_p * P
        d_tr_max = n_max * l_c + d_f + d_x + 0.5 * n_p * P

    else:
        raise ValueError("Invalid YARD_TYPE, choose 'parallel' or 'perpendicular'.")

    return {
        "M": M, "N": N, "n_t": n_t, "n_p": n_p, "n_r": n_r, "P": P,
        "yard_length": A_yard,
        "total_lane_length": total_lane_length,
        "railcar_length": railcar_length,
        "n_max": n_max, "n": n, "mu": mu,
        "d_h_min": d_h_min, "d_h_max": d_h_max,
        "d_r_min": d_r_min, "d_r_max": d_r_max,
        "d_t_min": d_t_min, "d_t_max": d_t_max,
        "d_tr_min": d_tr_min, "d_tr_mean": d_tr_mean, "d_tr_max": d_tr_max
    }

def ugly_sigma(x):
    """Discrete weighted sum used by the yard-distance closed form.

    Computes ``sum(2*i*(x-i) for i in 1..x-1) / x**2``. The name
    reflects the formula's origin as an ad-hoc geometric helper in
    the parallel / perpendicular yard models.

    Parameters
    ----------
    x : int
        Upper bound of the summation range.

    Returns
    -------
    float
        Normalised summation value.
    """
    total_sum = 0
    for i in range(1, x):
        total_sum += 2 * i * (x - i)
    return total_sum / (x ** 2)

def speed_density(avg_density, vehicle_type, N):
    """Sample speed from an empirical density-based exponential decay.

    Free-flow speeds (8 mph for hostlers, 10 mph for trucks) are
    decayed by ``exp(-(a*N + b) * avg_density)`` to capture congestion
    response, where the constants ``a``, ``b`` are vehicle-specific.

    Parameters
    ----------
    avg_density : float
        Vehicle density in vehicles per metre of total lane length.
    vehicle_type : str
        Either ``'hostler'`` or ``'truck'``.
    N : int
        Number of yard columns (used as a density-decay multiplier).

    Returns
    -------
    float
        Speed in the same units as the free-flow coefficient (mph).

    Raises
    ------
    ValueError
        If ``vehicle_type`` is not ``'hostler'`` or ``'truck'``.
    """
    if vehicle_type == 'hostler':
        speed = 8 * math.e ** ((-1.5 * N - 0.5) * avg_density)
    elif vehicle_type == 'truck':
        speed = 10 * math.e ** ((-3.5 * N - 0.5) * avg_density)
    else:
        raise ValueError("Invalid vehicle type. Choose 'hostler' or 'truck'.")
    return speed


def simulate_hostler_track_travel(hostler_id, current_veh_num, config=None, config_path="input/config.yaml", params=None):
    """Sample one hostler's inter-track travel time and distance.

    Combines a triangular distance distribution
    (``d_tr_min`` / ``d_tr_mean`` / ``d_tr_max`` from
    :func:`calculate_distances`) with a density-based speed
    (:func:`speed_density`) to produce a single travel-time draw.

    Parameters
    ----------
    hostler_id : int
        Identifier of the hostler being moved. Currently unused in
        the calculation; preserved for logging / future per-hostler
        behaviour.
    current_veh_num : int
        Active vehicle count contributing to congestion-induced
        slowdown.
    config : Mapping, optional
        Pre-parsed configuration mapping forwarded to
        :func:`calculate_distances`.
    config_path : str, optional
        Path to a YAML config; used only when both ``config`` and
        ``params`` are ``None``.
    params : Mapping, optional
        Pre-computed result of :func:`calculate_distances`. Provided
        as a hot-loop optimisation to avoid re-parsing the layout.

    Returns
    -------
    tuple of (float, float, float, float)
        ``(hostler_travel_time, d_tr_dist, hostler_speed,
        veh_density)`` with travel time in hours, distance in metres,
        speed in mph, and density in vehicles per metre.

    Raises
    ------
    ValueError
        If the configured ``d_tr_*`` bounds are inconsistent
        (``d_tr_max <= d_tr_min`` or ``d_tr_mean`` outside the
        range).
    """
    if params is None:
        params = calculate_distances(config=config, config_path=config_path)
    total_lane_length, N = params["total_lane_length"]/3.28, params["N"]    # 1 meter = 3.28 ft
    d_tr_min, d_tr_mean, d_tr_max = params["d_tr_min"]/3.28, params["d_tr_mean"]/3.28, params["d_tr_max"]/3.28

    # sanity check
    if d_tr_max <= d_tr_min:
        raise ValueError(f"Invalid distance range: d_tr_max={d_tr_max} meters, d_tr_min={d_tr_min} meters")
    if not (d_tr_min <= d_tr_mean <= d_tr_max):
        raise ValueError(f"d_tr_mean ({d_tr_mean}) must be between min ({d_tr_min}) meters and max ({d_tr_max}) meters")

    # normalized c (ensure 0 < c < 1)
    c = (d_tr_mean - d_tr_min) / (d_tr_max - d_tr_min)
    c = min(max(c, 1e-6), 1 - 1e-6)  # avoid exactly 0 or 1

    d_tr_dist = triang(c, loc=d_tr_min, scale=d_tr_max - d_tr_min).rvs()
    veh_density = current_veh_num / total_lane_length
    hostler_speed = speed_density(veh_density, 'hostler', N)
    hostler_travel_time = d_tr_dist / (hostler_speed * 3600)    # 1 hr = 3,600 s

    return hostler_travel_time, d_tr_dist, hostler_speed, veh_density
