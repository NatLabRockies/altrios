"""Energy-use record buffer and per-event recording helpers shared by the
LIFTS SimPy actor modules. The module-level ``energy_use_records`` list is
appended to by the actor processes (truck/crane/hostler) and consumed/cleared
by ``terminal_sim.run_terminal_simulation``, which converts the buffer into
the returned ``vehicle_log_df``.

The simulation tracks energy use in the fuel's native unit (gallons for
Diesel/Hybrid, kWh for Electric). A coarse CO2-equivalent emissions column
is appended at end-of-sim using the static :data:`CO2_KG_PER_UNIT` table
below. These are not curated, site-specific emission factors; downstream
consumers needing rigorous pollutant masses (CO2, NOx, PM, ...) should
recompute from the energy column with their own factor set.
"""
from altrios.lifts import utilities

# Module-level energy-use record buffer; populated by container_process /
# handle_remaining_oc / crane_unload_process / crane_load_process /
# truck_entry / truck_exit and consumed by run_terminal_simulation, which
# materializes it into a polars DataFrame returned to the caller.
energy_use_records: list = []

# Approximate CO2-equivalent emission factors, applied at end-of-sim to
# convert per-event energy use into a single ``emissions(kgCO2)`` column on
# the returned ``vehicle_log_df``. Diesel/Hybrid use kg CO2 per gallon
# (EPA ~10.21 for ultra-low-sulfur diesel); Electric uses kg CO2 per kWh
# (representative US grid average ~0.40). Override by editing this dict if
# you have a better factor set.
CO2_KG_PER_UNIT: dict = {
    "Diesel": 10.21,
    "Hybrid": 10.21,
    "Electric": 0.40,
}


def _energy_type_for(vehicle_obj) -> str:
    """Normalize hostler/truck/crane .type into the keys used by compute_energy_use."""
    t = getattr(vehicle_obj, "type", "Diesel")
    return str(t).capitalize()


def _record_trip_energy(terminal, vehicle_obj, vehicle_kind, status, train_id,
                        container_id, event_type, travel_time, env_now, track_id=""):
    """Helper to compute + append a trip energy-use record."""
    energy_type = _energy_type_for(vehicle_obj)
    energy_value = utilities.compute_energy_use(
        terminal, status=status, move="trip", vehicle=vehicle_kind,
        energy_type=energy_type, travel_time=travel_time,
    )
    utilities.record_energy_use(
        energy_use_records,
        vehicle_type=vehicle_kind,
        fuel_type=energy_type,
        resource_id=getattr(vehicle_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="yard",
        energy_value=energy_value,
        travel_time=travel_time,
        env_now=env_now,
    )


def _record_load_energy(terminal, crane_obj, status, train_id, container_id,
                        event_type, env_now, track_id):
    """Helper to compute + append a per-lift crane energy-use record."""
    energy_type = _energy_type_for(crane_obj)
    energy_value = utilities.compute_energy_use(
        terminal, status=status, move="load", vehicle="crane",
        energy_type=energy_type, travel_time=0.0,
    )
    utilities.record_energy_use(
        energy_use_records,
        vehicle_type="crane",
        fuel_type=energy_type,
        resource_id=getattr(crane_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="track",
        energy_value=energy_value,
        travel_time=0.0,
        env_now=env_now,
    )


def _record_side_energy(terminal, hostler_obj, train_id, container_id, env_now):
    """Helper to compute + append a side-pick energy-use record."""
    energy_type = _energy_type_for(hostler_obj)
    energy_value = utilities.compute_energy_use(
        terminal, status="loaded", move="side", vehicle="hostler",
        energy_type=energy_type, travel_time=0.0,
    )
    utilities.record_energy_use(
        energy_use_records,
        vehicle_type="hostler",
        fuel_type=energy_type,
        resource_id=getattr(hostler_obj, "id", ""),
        track_id="",
        train_id=train_id,
        container_id=container_id,
        event_type="side_pick",
        zone="parking",
        energy_value=energy_value,
        travel_time=0.0,
        env_now=env_now,
    )


