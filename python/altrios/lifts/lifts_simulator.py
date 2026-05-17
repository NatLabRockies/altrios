"""Legacy public entrypoint for the LIFTS intermodal rail simulator.

The implementation now lives in `terminal_sim.run_terminal_simulation`, which
dispatches by registered terminal mode. This module preserves the historical
`lifts_simulator.run_simulation(...)` import path used by demos and notebooks.
"""
import polars as pl

from altrios.lifts import utilities
from altrios.lifts.classes import loggingLevel
from altrios.lifts.terminal_sim import run_terminal_simulation


def run_simulation(
        train_consist_plan: pl.DataFrame,
        terminal: str,
        out_path=None,
        log_level: loggingLevel = loggingLevel.BASIC) -> pl.DataFrame:
    """Run the intermodal rail terminal simulation (compat shim).

    New code should call `terminal_sim.run_terminal_simulation(mode=..., ...)`
    directly so the mode is explicit.
    """
    return run_terminal_simulation(
        mode="intermodal_rail",
        train_consist_plan=train_consist_plan,
        terminal=terminal,
        out_path=out_path,
        log_level=log_level,
    )


if __name__ == "__main__":
    consist_plan = (pl.read_csv(utilities.package_root() / 'resources' / 'train_consist_plan.csv')
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )
    run_simulation(
        train_consist_plan=consist_plan,
        terminal = "Allouez",
        out_path = utilities.package_root() / 'demos' / 'lifts' / 'demos' / 'starter_demo' / 'results'
    )
