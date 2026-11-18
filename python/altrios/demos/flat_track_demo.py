import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import List
import json
from pprint import pprint

# our packages
import altrios as alt 

sns.set()

SHOW_PLOTS = alt.utils.show_plots()

SAVE_INTERVAL = 1

resources_dir = Path(__file__).parents[1] / "resources/"

MPH_TO_MPS = 0.44704

# %%
# Change in drag coefficient for the periodic case as a function of gap size
def drag_change_pct(gap_size: 
                    float) -> float:
    gap_size_array = [0.508, 0.968, 1.186, 1.407, 
                1.564, 1.627, 1.851] #gap size in meters from digitized plot
    drag_change_array = [-32.56, -24.93, -17.85, 
                         -6.678, 0.009, 1.7, 
                         7.876] # Change in drag coefficient for periodic boundary in %
    
    return np.interp(gap_size, xp=gap_size_array, 
                     fp=drag_change_array)

def create_drag_vec(
        num_rail_vehicles: int, 
        gap_size: float) -> List[float]:
    """
    Returns the drag coefficient vector as a function of number of rail vehicles in a consist
    and vehicle gap size

    Arguments:
    ---------
    num_rail_vehicles: int - Number of rail vehicles in the platoon
    gap_size: float - Gap size between the rail vehicles

    Output:
    ---------
    List of drag coefficients for each rail car. len(List) = num_rail_vehicles

    """

    ## From slide 16 of the Aerodynamic model PPT
    drag_vec_10cars_liang = [1.168, 0.292, 0.228,
                            0.217, 0.238, 0.209,
                            0.244,0.244,0.244, 0.409] 

    ## From slide 16 of the Aerodynamic model PPT
    periodic_drag_coeff_liang = 0.193

    rel_drag_change = drag_change_pct(gap_size)
    # rel_drag_change = -29.30
    drag_coeff_baseline = 0.108
    periodic_drag_coeff_ps = drag_coeff_baseline*(1+rel_drag_change/100)
    drag_ratio = periodic_drag_coeff_ps/periodic_drag_coeff_liang
    drag_vec = drag_vec_10cars_liang[0:num_rail_vehicles]

    ## For num_rail_vehicles 1, 2, and 3: 
    ## scaled the value for Liang's car from values in slide 24 
    if num_rail_vehicles == 1:
        drag_vec = [0.904/drag_ratio] 
    elif num_rail_vehicles == 2:
        drag_vec[0] = 0.504/drag_ratio  
        drag_vec[-1] = 0.904/drag_ratio - drag_vec[0]
    elif num_rail_vehicles == 3:
        drag_vec[0] = 0.504/drag_ratio
        # drag_vec[1] = 0.115/drag_ratio    
        drag_vec[-1] = 0.904/drag_ratio - sum(drag_vec[:-1])
    elif num_rail_vehicles >= 4:
        drag_vec[0] = 0.504/drag_ratio
        drag_vec[-1] = drag_vec_10cars_liang[-1]
        if num_rail_vehicles > 10:
            drag_vec = drag_vec_10cars_liang[:-1] + \
                [0.105]*(num_rail_vehicles-9)
            drag_vec[-1] = drag_vec_10cars_liang[-1]

    drag_vec_rail = [round(drag_ratio*x, 3)
                            for x in drag_vec]
    return drag_vec_rail



# PARAMETER_INFO  
num_rail_vehicles = 10
if num_rail_vehicles > 1:
    configuration = 'Slipstream'
elif num_rail_vehicles == 1:
    configuration ='Single Vehicle'
else:
    raise ValueError('Enter valid number of rail vehicles')


gap_size = 0.610 #[m]
train_speed = 25.0 * MPH_TO_MPS #[m/s]
total_time = 30*60 #[s] 
track_length = train_speed*total_time #[m]
time_vec = np.arange(0,total_time,1)
vehicle_type = 'Manifest_Empty'
rail_vehicle_file = resources_dir / Path("rolling_stock/%s.yaml"%vehicle_type)
assert rail_vehicle_file.exists(), f"Rail vehicle file of type {vehicle_type}.yaml does not exists!"
rail_vehicle = alt.RailVehicle.from_file(resources_dir / rail_vehicle_file)

length = track_length
train_length = rail_vehicle.length_meters

# TRAIN NETWORK
elevs = [
    {"offset": row[0], "elev": row[1]} for row in zip(np.array([0.0,track_length]), 
                                                      np.array([0.0,0.0]))
]
# NOTE: we don't need to modify this until Wang has a good solution to the network ingestion
# problem.  The track segments used for this are pretty straight so we're not really hurting
# anything by doing this.
headings = [
    {"offset": i, "heading": 0.0} for i in [0.0, length]
]

speed_set = {'speed_limits': [{'offset_start': 0.0,
     'offset_end': length,
     'speed': 1e3}],
   'speed_params': [],
   'is_head_end': False}

#%%

sim_link = alt.Link.from_yaml(json.dumps({
    'idx_curr': 1,
    'idx_flip': 0,
    'idx_next': 0,
    'idx_next_alt': 0,
    'idx_prev': 0,
    'idx_prev_alt': 0,
    'length': length, # whatever the trip length is
    'elevs': elevs, 
    'headings': headings,
    'speed_sets': {},
    'speed_set': speed_set,
    'cat_power_limits': [],
    'link_idxs_lockout': []
}))

network = alt.Network([
    alt.Link.default(),
    sim_link,
])

#%%

speed_trace = alt.SpeedTrace(
    time_seconds=time_vec,
    speed_meters_per_second=train_speed*np.ones_like(time_vec)
)

# RUN SIMULATION
train_config = alt.TrainConfig(
    # cars_empty=0,
    rail_vehicles = [rail_vehicle],
    n_cars_by_type = {vehicle_type: num_rail_vehicles},
    # cars_loaded=num_rail_vehicles,
    train_type=None,
    train_length_meters=train_length,
    train_mass_kilograms=None,
    cd_area_vec = np.array(
                    create_drag_vec(num_rail_vehicles, 
                    gap_size)
                    ),
    # drag_coeff_vec = np.array([0.504, 0.113, 0.088, 0.084, 0.092, 0.081, 0.094, 0.094, 0.094,
    #    0.158])*rail_vehicle.drag_area_loaded_square_meters,
)

res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root() / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)
# instantiate electric drivetrain (motors and any gearboxes)
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/electric_drivetrain/struct.ElectricDrivetrain.html
edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0., 1.],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

loco_type = alt.BatteryElectricLoco(res, edrv)


bel: alt.Locomotive = alt.Locomotive(
    loco_type=loco_type,
    loco_params=alt.LocoParams.from_dict(dict(
        pwr_aux_offset_watts=8.55e3,
        pwr_aux_traction_coeff_ratio=540.e-6,
        force_max_newtons=667.2e3,
)))

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel] #* num_rail_vehicles
# instantiate consist
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

init_train_state = alt.InitTrainState(speed_meters_per_second = train_speed)

tsb = alt.TrainSimBuilder(
    train_id="0",
    train_config=train_config,
    loco_con=loco_con,
    init_train_state = init_train_state
)

train_sim: alt.SetSpeedTrainSim = tsb.make_set_speed_train_sim(
    # rail_vehicle=rail_vehicle,
    network=network,
    link_path=[alt.LinkIdx(1)],
    speed_trace=speed_trace,
    save_interval=SAVE_INTERVAL,
)


t0 = time.perf_counter()
train_sim.walk()
t1 = time.perf_counter()
print(f'Time to simulate: {t1 - t0:.5g}')

# %%

loco0: alt.Locomotive = train_sim.loco_con.loco_vec.tolist()[0]

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.pwr_whl_out_watts) / 1e3,
    label="tract pwr",
)
ax[0].set_ylabel('Power [kW]')
ax[0].legend()

ax[1].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.res_aero_newtons) / 1e3,
    label='aero',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.res_rolling_newtons) / 1e3,
    label='rolling',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.res_curve_newtons) / 1e3,
    label='curve',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.res_bearing_newtons) / 1e3,
    label='bearing',
)
ax[1].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.res_grade_newtons) / 1e3,
    label='grade',
)
ax[1].set_ylabel('Force [MN]')
ax[1].legend()

ax[2].plot(
    np.array(train_sim.history.time_seconds),
    np.array(loco0.res.history.soc)
)
ax[2].set_ylabel('SOC')
ax[2].set_ylim([0,1])

ax[-1].plot(
    np.array(train_sim.history.time_seconds),
    train_sim.history.speed_meters_per_second,
    label='achieved'
)
ax[-1].set_xlabel('Time [s]')
ax[-1].set_ylabel('Speed [m/s]')
ax[-1].legend()

fig1, ax1 = plt.subplots(3, 1, sharex=True)
ax1[0].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.offset_in_link_meters) / 1_000,
    label='current link',
)
ax1[0].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.offset_meters) / 1_000,
    label='overall',
)
ax1[0].legend()
ax1[0].set_ylabel('Net Dist. [km]')

ax1[1].plot(
    np.array(train_sim.history.time_seconds),
    train_sim.history.link_idx_front,
    linestyle='',
    marker='.',
)
ax1[1].set_ylabel('Link Idx Front')

ax1[-1].plot(
    np.array(train_sim.history.time_seconds),
    train_sim.history.speed_meters_per_second,
)
ax1[-1].set_xlabel('Time [s]')
ax1[-1].set_ylabel('Speed [m/s]')

plt.tight_layout()


fig2, ax2 = plt.subplots(3, 1, sharex=True)
ax2[0].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.pwr_whl_out_watts) / 1e3,
    label="tract pwr",
)
ax2[0].set_ylabel('Power [kW]')
ax2[0].legend()

ax2[1].plot(
    np.array(train_sim.history.time_seconds),
    np.array(train_sim.history.grade_front) * 100.,
)
ax2[1].set_ylabel('Grade [%] at\nHead End')

ax2[-1].plot(
    np.array(train_sim.history.time_seconds),
    train_sim.history.speed_meters_per_second,
)
ax2[-1].set_xlabel('Time [s]')
ax2[-1].set_ylabel('Speed [m/s]')

plt.tight_layout()


if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()

res_dict = {'Vehicle Type' : vehicle_type,
            'Vehicle Mass [kg]' : round(rail_vehicle.mass_static_base_kilograms*num_rail_vehicles,3),
            'Vehicle Speed [m/s]': round(train_sim.state.speed_meters_per_second,3),
            'Vehicle Frontal Area [m2]:': round(rail_vehicle.cd_area_square_meters,3),
            'Track Distance [km]' : round(train_sim.state.offset_meters/1E3, 2),
            'Tractive Power [kW]' : round(train_sim.state.pwr_whl_out_watts/1E3, 2),
            'Resistance - Aerodynamic [kN]' : round(train_sim.state.res_aero_newtons/1E3, 2),
            'Resistance - Rolling [kN]' : round(train_sim.state.res_rolling_newtons/1E3, 2),
            'Resistance - Curve [kN]' : round(train_sim.state.res_curve_newtons/1E3, 2),
            'Resistance - Grade [kN]' : round(train_sim.state.res_grade_newtons/1E3, 2),
            'Resistance - Bearing [kN]' : round(train_sim.state.res_bearing_newtons/1E3, 2),
            'Configuration': configuration}

pprint(res_dict)