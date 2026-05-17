"""Emission record buffer and per-event recording helpers shared by the LIFTS
SimPy actor modules. The module-level ``emission_records`` list is appended to
by the actor processes (truck/crane/hostler) and consumed/cleared by
``lifts_simulator.run_simulation``.
"""
from altrios.lifts import utilities

# Module-level emission record buffer; populated by container_process /
# handle_remaining_oc / crane_unload_process / crane_load_process / truck_entry
# / truck_exit and consumed by run_simulation when out_path is supplied.
emission_records: list = []


def _energy_type_for(vehicle_obj) -> str:
    """Normalize hostler/truck/crane .type into the keys used by emission_calculation."""
    t = getattr(vehicle_obj, "type", "Diesel")
    return str(t).capitalize()


def _record_trip_emissions(terminal, vehicle_obj, vehicle_kind, status, train_id,
                           container_id, event_type, travel_time, env_now, track_id=""):
    """Helper to compute + append a trip emission record."""
    energy_type = _energy_type_for(vehicle_obj)
    emission_value = utilities.emission_calculation(
        terminal, status=status, move="trip", vehicle=vehicle_kind,
        energy_type=energy_type, travel_time=travel_time,
    )
    utilities.record_emission(
        emission_records,
        vehicle_type=vehicle_kind,
        resource_id=getattr(vehicle_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="yard",
        emission_value=emission_value,
        travel_time=travel_time,
        env_now=env_now,
    )


def _record_load_emissions(terminal, crane_obj, status, train_id, container_id,
                           event_type, env_now, track_id):
    """Helper to compute + append a per-lift crane emission record."""
    energy_type = _energy_type_for(crane_obj)
    emission_value = utilities.emission_calculation(
        terminal, status=status, move="load", vehicle="crane",
        energy_type=energy_type, travel_time=0.0,
    )
    utilities.record_emission(
        emission_records,
        vehicle_type="crane",
        resource_id=getattr(crane_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="track",
        emission_value=emission_value,
        travel_time=0.0,
        env_now=env_now,
    )


def _record_side_emissions(terminal, hostler_obj, train_id, container_id, env_now):
    """Helper to compute + append a side-pick emission record."""
    energy_type = _energy_type_for(hostler_obj)
    emission_value = utilities.emission_calculation(
        terminal, status="loaded", move="side", vehicle="hostler",
        energy_type=energy_type, travel_time=0.0,
    )
    utilities.record_emission(
        emission_records,
        vehicle_type="hostler",
        resource_id=getattr(hostler_obj, "id", ""),
        track_id="",
        train_id=train_id,
        container_id=container_id,
        event_type="side_pick",
        zone="parking",
        emission_value=emission_value,
        travel_time=0.0,
        env_now=env_now,
    )
