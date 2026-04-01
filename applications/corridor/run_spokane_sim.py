import os
import sys
import time
import logging
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure we can import altrios
try:
    import altrios as alt
except ImportError:
    # If running from source, might need to adjust path or install package
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../python"))
    import altrios as alt

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_simulation(network_file, locations_file, output_html):
    logger.info(f"Loading Network from {network_file}...")
    try:
        network = alt.Network.from_file(network_file)
    except Exception as e:
        logger.error(f"Failed to load network: {e}")
        return

    logger.info(f"Loading Locations from {locations_file}...")
    try:
        location_map = alt.import_locations(locations_file)
    except Exception as e:
        logger.error(f"Failed to load locations: {e}")
        return

    SAVE_INTERVAL = 1

    # 1. Build Train Config
    logger.info("Configuring Train...")
    res_root = alt.resources_root()

    try:
        rail_vehicle_loaded = alt.RailVehicle.from_file(
            res_root / "rolling_stock/Manifest_Loaded.yaml"
        )
        rail_vehicle_empty = alt.RailVehicle.from_file(
            res_root / "rolling_stock/Manifest_Empty.yaml"
        )
        
        # Override Max Speed to 50 mph (approx 22.352 m/s)
        target_speed_mps = 22.352 
        
        d_loaded = rail_vehicle_loaded.to_pydict()
        d_loaded['speed_max_meters_per_second'] = target_speed_mps
        rail_vehicle_loaded = alt.RailVehicle.from_pydict(d_loaded)
        
        d_empty = rail_vehicle_empty.to_pydict()
        d_empty['speed_max_meters_per_second'] = target_speed_mps
        rail_vehicle_empty = alt.RailVehicle.from_pydict(d_empty)
        
        logger.info(f"Overridden Train Max Speed to {target_speed_mps:.2f} m/s (50 mph)")

        train_config = alt.TrainConfig(
            rail_vehicles=[rail_vehicle_loaded, rail_vehicle_empty],
            n_cars_by_type={
                "Manifest_Loaded": 80,
                "Manifest_Empty": 20,
            },
            train_length_meters=None,
            train_mass_kilograms=None,
        )
    except Exception as e:
        logger.warning(f"Could not load rolling stock: {e}. Using defaults/fallbacks.")
        # Try to construct minimal defaults if files missing, or return
        return

    # Consist: 3 Diesel Locomotives
    loco_vec = [alt.Locomotive.default()] * 3
    loco_con = alt.Consist(loco_vec, SAVE_INTERVAL)

    # 2. Setup Simulation Builder
    tsb = alt.TrainSimBuilder(
        train_id="Spokane-Lethbridge-001",
        origin_id="Spokane",
        destination_id="Lethbridge",
        train_config=train_config,
        loco_con=loco_con,
    )

    logger.info("Creating SpeedLimitTrainSim...")
    try:
        train_sim: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
            location_map=location_map,
            save_interval=SAVE_INTERVAL,
        )
    except Exception as e:
        logger.error(f"Failed to make speed limit train sim: {e}")
        return

    # 3. Make Estimated Times
    logger.info("Running make_est_times...")
    try:
        est_time_net, _consist = alt.make_est_times(train_sim, network)
    except Exception as e:
        logger.error(f"make_est_times failed: {e}")
        return

    # 4. Run Dispatch
    logger.info("Running Dispatch...")
    try:
        t0 = time.perf_counter()

        # run_dispatch returns a vector of TimedLinkPaths
        timed_link_paths = alt.run_dispatch(
            network,
            alt.SpeedLimitTrainSimVec([train_sim]),
            [est_time_net],
            True,  # stop_if_fail
            False,  # verbose
        )

        t1 = time.perf_counter()
        logger.info(f"Dispatch took {t1 - t0:.2f}s")

        if not timed_link_paths:
            logger.error("Dispatch returned no paths! Check if feasible.")
            return

        timed_link_path = timed_link_paths[0]

    except Exception as e:
        logger.error(f"Dispatch execution failed: {e}")
        return

    # 5. Extract Results
    # Check if history is populated in train_sim object
    sim_data = train_sim.to_pydict()
    history = sim_data.get("history", {})

    # If dispatch didn't populate history (it might simulate efficiently without saving full history?),
    # run walk_timed_path
    if not history or not history.get("time_seconds"):
        logger.info(
            "History empty after dispatch. Running walk_timed_path to generate detailed history..."
        )
        try:
            train_sim.walk_timed_path(network, timed_link_path)
            sim_data = train_sim.to_pydict()
            history = sim_data.get("history", {})
        except Exception as e:
            logger.error(f"walk_timed_path failed: {e}")
            return

    if not history or not history.get("time_seconds"):
        logger.error("Simulation history is still empty!")
        return

    # Convert to DataFrame
    dist_key = (
        "offset_back_meters" if "offset_back_meters" in history else "distance_meters"
    )

    def get_hist(key, default=0.0):
        val = history.get(key)
        if val is None:
            return [default] * len(history["time_seconds"])
        return val

    # Determine Elevation Key
    elev_key = "elevation_meters"
    if "elev_front_meters" in history:
        elev_key = "elev_front_meters"
    elif "elev_back_meters" in history:
        elev_key = "elev_back_meters"
    else:
        logger.warning(f"No obvious elevation key found. Keys: {list(history.keys())[:10]}...")

    df = pd.DataFrame(
        {
            "time": history["time_seconds"],
            "speed_mps": history["speed_meters_per_second"],
            "speed_limit_mps": get_hist("speed_limit_meters_per_second"),
            "dist_m": get_hist(dist_key),
            "elevation_m": get_hist(elev_key),
            "power_w": get_hist("pwr_whl_out_watts"),
        }
    )

    # Calculate Fuel
    try:
        fuel_j = train_sim.get_energy_fuel_joules(False)
        logger.info(f"Total Fuel Consumed: {fuel_j / 1e9:.4f} GJ")
    except:
        pass

    # 6. Visualize
    create_visualization(df, output_html)


def create_visualization(df, output_html):
    logger.info(f"Creating visualization at {output_html}...")

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Elevation Profile", "Speed Profile", "Power"),
        row_heights=[0.3, 0.4, 0.3],
    )

    # 1. Elevation Profile
    fig.add_trace(
        go.Scatter(
            x=df["dist_m"] / 1000,
            y=df["elevation_m"],
            name="Elevation (m)",
            fill="tozeroy",
        ),
        row=1,
        col=1,
    )

    # 2. Speed Profile
    fig.add_trace(
        go.Scatter(
            x=df["dist_m"] / 1000,
            y=df["speed_mps"] * 2.23694,
            name="Speed (mph)",
            line=dict(color="blue"),
        ),
        row=2,
        col=1,
    )

    if "speed_limit_mps" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["dist_m"] / 1000,
                y=df["speed_limit_mps"] * 2.23694,
                name="Speed Limit (mph)",
                line=dict(color="red", dash="dash"),
            ),
            row=2,
            col=1,
        )

    # 3. Power Profile
    fig.add_trace(
        go.Scatter(
            x=df["dist_m"] / 1000,
            y=df["power_w"] / 1e6,
            name="Power (MW)",
            line=dict(color="green"),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title="Spokane to Lethbridge Simulation Results",
        xaxis3_title="Distance (km)",
        yaxis1_title="Elevation (m)",
        yaxis2_title="Speed (mph)",
        yaxis3_title="Power (MW)",
        height=900,
        showlegend=True,
    )

    fig.write_html(output_html)
    logger.info("Visualization Saved.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Spokane-Lethbridge simulation")
    parser.add_argument(
        "--network",
        default="spokane_lethbridge_with_sidings",
        help="Network file prefix under networks/ (default: spokane_lethbridge_with_sidings)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML file path (default: networks/<network>_simulation_results.html)",
    )
    args = parser.parse_args()

    workspace_root = os.getcwd()
    network_yaml = os.path.join(workspace_root, "networks", f"{args.network}.yaml")
    locations_csv = os.path.join(workspace_root, "networks", f"{args.network}_locations.csv")
    output_html = args.output or os.path.join(
        workspace_root, "networks", f"{args.network}_simulation_results.html"
    )

    if not os.path.exists(network_yaml):
        logger.error(f"Network file not found: {network_yaml}")
        return

    if not os.path.exists(locations_csv):
        logger.error(f"Locations file not found: {locations_csv}")
        return

    run_simulation(network_yaml, locations_csv, output_html)


if __name__ == "__main__":
    main()
