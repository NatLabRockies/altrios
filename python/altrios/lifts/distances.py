import numpy as np
from altrios.lifts.single_track_parameters import *
from scipy.stats import triang, uniform
import math
import json

def load_layout_config_from_json(path="sim_config.json"):
    import json
    global k, M, N, n_t, n_p, n_r
    with open(path, "r") as f:
        config = json.load(f)
        layout = config["layout"]
        k = layout["k"]
        M = layout["M"]
        N = layout["N"]
        n_t = layout["n_t"]
        n_p = layout["n_p"]
        n_r = layout["n_r"]

# Yard setting: optimal layout output
YARD_TYPE = 'parallel'  # choose 'perpendicular' or 'parallel'
k = 200 # train batch size
M = 2 # decide the number of rows of parking blocks in the layout
N = 3 # decide the number of columns of parking blocks in the layout
n_t = 2 # decide the numbers of train side aisles per group
n_p = 2 # decide the numbers of parking area aisles per group
n_r = 17  # decide the number of spots within each parking block (10 * n_r = BL_l, the length of each parking block)

# Fixed yard parameters
P = 10  # fixed aisle width
BL_w = 80  # fixed block width

A = M * 10 * n_r + (M+1) * n_p * P  # the vertical width of the yard
B = N * 80 + (N+1) * n_p * P # the horizontal length of the yard

# Total length of yard lanes, used to estimate density (veh/ft)
total_lane_length = A * (N + 1) + B * (M + 1)  # total distances of lanes

def speed_density(count, vehicle_type):
    '''
    Unit of speed: ft/s
    '''
    if vehicle_type == 'hostler':   # V_h = (1.7033 + 0.1445 nr + 0.3020 k) * e(-1.4726 * N - 0.5197) * d
        speed = (1.7 + 0.1 * n_r + 0.003 * k) * math.e ** ((-1.5 * N - 0.5) * (count / total_lane_length))
    elif vehicle_type == 'truck':   # V_t = 10 * e(-3.5 * N - 0.5) * d
        speed = 10 * math.e ** ((-3.5 * N - 0.5) * (count / total_lane_length))
    else:
        raise ValueError("Invalid vehicle type. Choose 'hostler' or 'truck'.")
    return speed


def simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max):
    """
    Simulates the truck travel time based on uniform distribution and vehicle density.

    Parameters:
    - total_lane_length: Total lane length (ft)
    - d_t_min, d_t_max: Range for travel distance (ft)
    """

    # Generate truck travel distance from uniform distribution
    d_t_dist = uniform(loc=d_t_min, scale=(d_t_max - d_t_min)).rvs()

    # Calculate vehicle density
    current_veh_num = train_schedule["truck_number"] - len(terminal.truck_store.items)
    veh_density = current_veh_num / total_lane_length

    # Compute truck speed based on density
    truck_speed = speed_density(veh_density, 'truck')
    print(f"Current truck {truck_id} speed is {truck_speed} (m/s)")

    # Compute truck travel time in hours and convert to seconds
    truck_travel_time = (d_t_dist/3.2) / (2 * truck_speed * 3600)  # (ft -> m) / (m/hr)
    print(f"Truck {truck_id} travel time {truck_travel_time} (hr)")

    return truck_travel_time


def simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max):
    global state
    # Generate hostler travel distance from uniform distribution
    d_h_dist = uniform(loc=d_h_min, scale=(d_h_max - d_h_min)).rvs()

    # Calculate vehicle density
    veh_density = current_veh_num / total_lane_length

    # Compute hostler speed based on density
    hostler_speed = speed_density(veh_density, 'hostler')
    print(f"Current hostler {hostler_id} speed is {hostler_speed} (m/s)")

    # Compute hostler travel time in hours and convert to seconds
    hostler_travel_time = (d_h_dist/3.2) / (2 * hostler_speed * 3600)     # (ft -> m) / (m/hr)
    print(f"hostler {hostler_id} travel time {hostler_travel_time} (hr)")

    return hostler_travel_time

def simulate_reposition_travel(hostler_id, current_veh_num, total_lane_length, d_r_min, d_r_max):
    global state
    # Generate reposition travel distance from uniform distribution
    d_r_dist = uniform(loc=d_r_min, scale=(d_r_max - d_r_min)).rvs()

    # Calculate vehicle density
    veh_density = current_veh_num / total_lane_length

    # Compute reposition speed based on density
    hostler_speed = speed_density(veh_density, 'hostler')
    print(f"Current hostler {hostler_id} speed is {hostler_speed} (m/s)")

    # Compute hostler travel time in hours and convert to seconds
    hostler_reposition_travel_time = (d_r_dist/3.2) / (2 * hostler_speed * 3600)      # (ft -> m) / (m/hr)
    print(f"hostler {hostler_id} travel time {hostler_reposition_travel_time} (hr)")

    return hostler_reposition_travel_time


def triang_distribution(min_val, avg_val, max_val):
    c = (avg_val - min_val) / (max_val - min_val)
    return triang(c, loc=min_val, scale=(max_val - min_val))

def uniform_distribution(min_val, max_val):
    return uniform(loc=min_val, scale=(max_val - min_val))

def uniform_mean(min_val, max_val):
    return (max_val + min_val) / 2

def ugly_sigma(x):
    total_sum = 0
    for i in range(1, x):
        total_sum += 2 * i * (x - i)
    result = total_sum / (x ** 2)
    return result

def A(M, n_r, n_p):
    return M * 10 * n_r + (M+1) * n_p * P

def B(N, n_p):
    return N * 80 + (N+1) * n_p * P

# Distance estimation
if YARD_TYPE == 'parallel':
    # d_h: hostler distance
    d_h_min = n_t * P + 1.5 * n_p * P
    d_h_max = n_t * P + A(M, n_r, n_p) - n_p * P + B(N, n_p) - n_p * P
    d_h_avg = (d_h_max + d_h_min) / 2

    # d_r: repositioning distance
    d_r_min = 5 * n_r + 40
    d_r_max = ugly_sigma(M) * (10 * n_r + n_p * P) + ugly_sigma(N) * (80 + n_p * P)
    d_r_avg = (d_r_max + d_r_min) / 2

    # d_t: truck distance
    d_t_min = 0.5 * n_p * P
    d_t_max = B(N, n_p) - n_p * P + A(M, n_r, n_p) - n_p * P
    d_t_avg = (d_t_max + d_t_min) / 2


elif YARD_TYPE == 'perpendicular':
    # d_h
    d_h_min = n_t * P + 1.5 * n_p * P
    d_h_avg = 10 * n_r * M + 80 * N + (M + N + 1.5) * n_p * P + 2 * n_t * P
    d_h_max = n_t * P + A(M, n_r, n_p) - n_p * P + B(N, n_p) - n_p * P

    # d_r
    d_r_min = 0
    d_r_avg = 5 * n_r + 40 + ugly_sigma(M) * (10 * n_r + n_p*P) + ugly_sigma(N) * (80 + n_p * P)
    d_r_max = 10 * n_r + 80 + A(M, n_r, n_p) - n_p * P + B(N, n_p) - n_p * P

    # d_t
    d_t_min = 1.5 * n_p * P
    d_t_avg = 0.5 * (B(N, n_p) + A(M, n_r, n_p) - 0.5 * n_p * P)
    d_t_max = B(N, n_p) + A(M, n_r, n_p) - 2 * n_p * P