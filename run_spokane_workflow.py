import subprocess
import os
import sys
import logging

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Workflow")


def run_script(script_path):
    """Run a python script and stream output."""
    logger.info(f"Running {script_path}...")

    # Use the same python interpreter as the current process
    cmd = [sys.executable, script_path]

    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            logger.info(f"{script_path} completed successfully.")
        else:
            logger.error(f"{script_path} failed with return code {result.returncode}")
            sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_path}: {e}")
        sys.exit(1)


def main():
    workspace_root = os.getcwd()  # Assumption: running from root

    # 1. Build Network
    logger.info("=== STEP 1: Building Network from OSM XML ===")
    build_script = os.path.join(
        workspace_root,
        "applications",
        "network_builder",
        "build_altrios_with_sidings.py",
    )
    if not os.path.exists(build_script):
        logger.error(f"Script not found: {build_script}")
        return
    run_script(build_script)

    # 2. Visualize Network Geometry
    logger.info("=== STEP 2: Visualizing Network Geometry ===")
    viz_script = os.path.join(
        workspace_root, "applications", "network_builder", "plot_altrios_path.py"
    )
    if os.path.exists(viz_script):
        run_script(viz_script)
    else:
        logger.warning(f"Visualization script not found: {viz_script}")

    # 3. Run Simulation
    logger.info("=== STEP 3: Running Simulation ===")
    sim_script = os.path.join(
        workspace_root, "applications", "corridor", "run_spokane_sim.py"
    )
    if os.path.exists(sim_script):
        run_script(sim_script)
    else:
        logger.error(f"Simulation script not found: {sim_script}")
        return

    logger.info("=== Workflow Completed Successfully ===")
    logger.info("Results available in 'networks/' directory:")
    logger.info("- Network: spokane_lethbridge_with_sidings.yaml")
    logger.info("- Geometry Plot: spokane_lethbridge_with_sidings_plot.html")
    logger.info("- Simulation Results: spokane_simulation_results.html")


if __name__ == "__main__":
    main()
