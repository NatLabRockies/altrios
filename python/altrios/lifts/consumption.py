"""Consumption record buffer and per-event recording helpers shared by the
LIFTS SimPy actor modules. The module-level ``consumption_records`` list
is appended to by the actor processes (truck/crane/hostler) and
consumed/cleared by ``terminal_sim.run_terminal_simulation``, which
converts the buffer into the returned ``resource_log_df``.

The simulation tracks energy consumption in the fuel's native unit
(gallons for Diesel/Hybrid, kWh for Electric). A coarse CO2-equivalent
emissions column is appended at end-of-sim using the static
:data:`CO2_KG_PER_UNIT` table below. These are not curated, site-specific
emission factors; downstream consumers needing rigorous pollutant masses
(CO2, NOx, PM, ...) should recompute from the energy column with their
own factor set.

Phase 3A.4 renamed this module from ``energy_use`` to ``consumption`` as
part of the broader ``record_energy_use`` → ``record_consumption``
generalization (see ``WORKFLOW_ENGINE_PLAN.md`` decision #11). Phase
3A.5 then renamed the output dataframe variable
``vehicle_log_df`` → ``resource_log_df``, renamed the column
``energy_consumption(gal_or_kWh)`` → ``consumption_value``, and added
``role`` and ``quantity`` columns so the same dataframe can absorb
non-energy consumption rows in Phase 3D.
"""
from altrios.lifts import utilities

# Module-level consumption record buffer; populated by container_process /
# handle_remaining_oc / crane_unload_process / crane_load_process /
# truck_entry / truck_exit and consumed by run_terminal_simulation, which
# materializes it into a polars DataFrame returned to the caller.
consumption_records: list = []

# Approximate CO2-equivalent emission factors, applied at end-of-sim to
# convert per-event energy consumption into a single ``emissions(kgCO2)``
# column on the returned ``resource_log_df``. Diesel/Hybrid use kg CO2
# per gallon (EPA ~10.21 for ultra-low-sulfur diesel); Electric uses kg
# CO2 per kWh (representative US grid average ~0.40). Override by editing
# this dict if you have a better factor set.
CO2_KG_PER_UNIT: dict = {
    "Diesel": 10.21,
    "Hybrid": 10.21,
    "Electric": 0.40,
}


def _fuel_type_for(vehicle_obj) -> str:
    """Normalize hostler/truck/crane .type into the keys used by compute_consumption."""
    t = getattr(vehicle_obj, "type", "Diesel")
    return str(t).capitalize()


def _record_trip_consumption(output, energy_use_config, vehicle_obj, vehicle_kind, status, train_id,
                             container_id, event_type, travel_time, env_now, track_id=""):
    """Helper to compute + append a trip consumption record.

    ``output`` (``OutputCollector`` or ``None``) receives the row via
    :meth:`OutputCollector.record_consumption` when non-None; the
    module-level :data:`consumption_records` buffer is also populated
    for the legacy parity tests. ``energy_use_config`` is the
    ``config["energy_use"]`` sub-dict.
    """
    fuel_type = _fuel_type_for(vehicle_obj)
    consumption_value = utilities.compute_consumption(
        energy_use_config, status=status, move="trip", vehicle=vehicle_kind,
        energy_type=fuel_type, travel_time=travel_time,
    )
    row = utilities.record_consumption(
        consumption_records,
        vehicle_type=vehicle_kind,
        role="equipment",
        fuel_type=fuel_type,
        resource_id=getattr(vehicle_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="yard",
        consumption_value=consumption_value,
        travel_time=travel_time,
        env_now=env_now,
    )
    if output is not None:
        output.record_consumption(row)


def _record_load_consumption(output, energy_use_config, crane_obj, status, train_id, container_id,
                             event_type, env_now, track_id):
    """Helper to compute + append a per-lift crane consumption record."""
    fuel_type = _fuel_type_for(crane_obj)
    consumption_value = utilities.compute_consumption(
        energy_use_config, status=status, move="load", vehicle="crane",
        energy_type=fuel_type, travel_time=0.0,
    )
    row = utilities.record_consumption(
        consumption_records,
        vehicle_type="crane",
        role="equipment",
        fuel_type=fuel_type,
        resource_id=getattr(crane_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="track",
        consumption_value=consumption_value,
        travel_time=0.0,
        env_now=env_now,
    )
    if output is not None:
        output.record_consumption(row)


def _record_side_consumption(output, energy_use_config, hostler_obj, train_id, container_id, env_now):
    """Helper to compute + append a side-pick consumption record.

    The side-pick is logically performed by a side-loading crane, but no
    such resource type exists in the simulation yet (TODO: add a per-yard
    or per-track side_loading_crane store with its own pool, fuel mix, and
    consumption config block; until then, the assigned hostler stands in
    as the actor). The record's ``resource_type`` is set to
    ``"side_loading_crane"`` so the event is visible as such in
    ``resource_log_df``; ``resource_id`` / ``fuel_type`` are still taken
    from the hostler so the per-event consumption uses a real fuel type.
    """
    fuel_type = _fuel_type_for(hostler_obj)
    consumption_value = utilities.compute_consumption(
        energy_use_config, status="loaded", move="side", vehicle="hostler",
        energy_type=fuel_type, travel_time=0.0,
    )
    row = utilities.record_consumption(
        consumption_records,
        vehicle_type="side_loading_crane",
        role="equipment",
        fuel_type=fuel_type,
        resource_id=getattr(hostler_obj, "id", ""),
        track_id="",
        train_id=train_id,
        container_id=container_id,
        event_type="side_pick",
        zone="parking",
        consumption_value=consumption_value,
        travel_time=0.0,
        env_now=env_now,
    )
    if output is not None:
        output.record_consumption(row)


# ---------------------------------------------------------------------------
# Phase 1F helpers for the new flow modules (yard_flow / vessel_flow /
# drayage_flow). As of Phase 1H these pass equipment-specific vehicle keys
# to ``compute_consumption``, which resolves them against the per-equipment
# rates in ``energy_use.load_consumption`` / ``energy_use.trip_consumption``
# (with the legacy ``crane_*`` / ``hostler_*`` keys retained as fallbacks).
# ---------------------------------------------------------------------------

def _record_stack_lift_consumption(
    output, energy_use_config, crane_obj, pool_name, status, train_id,
    container_id, event_type, env_now, zone="stack",
):
    """Per-lift consumption record for a stack-lift equipment piece.

    ``pool_name`` is the equipment family (``"main_stack_rtg"``,
    ``"top_pick"``, ``"sts_crane"``, ``"rail_track_rtg"``) and is used
    both as the ``resource_type`` label on ``resource_log_df`` and as the
    per-equipment rate key (``<pool_name>_<status>``) in
    ``energy_use.load_consumption``.
    """
    fuel_type = _fuel_type_for(crane_obj)
    consumption_value = utilities.compute_consumption(
        energy_use_config, status=status, move="load", vehicle=pool_name,
        energy_type=fuel_type, travel_time=0.0,
    )
    row = utilities.record_consumption(
        consumption_records,
        vehicle_type=pool_name,
        role="equipment",
        fuel_type=fuel_type,
        resource_id=getattr(crane_obj, "id", ""),
        track_id=str(getattr(crane_obj, "track_id", "")
                     or getattr(crane_obj, "berth_id", "")
                     or ""),
        train_id=str(train_id),
        container_id=container_id,
        event_type=event_type,
        zone=zone,
        consumption_value=consumption_value,
        travel_time=0.0,
        env_now=env_now,
    )
    if output is not None:
        output.record_consumption(row)


def _record_yard_tractor_trip_consumption(
    output, energy_use_config, tractor_obj, status, train_id,
    container_id, event_type, travel_time, env_now,
):
    """Per-trip consumption record for a yard tractor haul. Looks up the
    per-equipment ``yard_tractor_loaded`` / ``yard_tractor_empty`` rates
    (with the legacy ``hostler_*`` keys as fallback)."""
    fuel_type = _fuel_type_for(tractor_obj)
    consumption_value = utilities.compute_consumption(
        energy_use_config, status=status, move="trip", vehicle="yard_tractor",
        energy_type=fuel_type, travel_time=travel_time,
    )
    row = utilities.record_consumption(
        consumption_records,
        vehicle_type="yard_tractor",
        role="equipment",
        fuel_type=fuel_type,
        resource_id=getattr(tractor_obj, "id", ""),
        track_id=str(getattr(tractor_obj, "pool", "")),
        train_id=str(train_id),
        container_id=container_id,
        event_type=event_type,
        zone="yard",
        consumption_value=consumption_value,
        travel_time=travel_time,
        env_now=env_now,
    )
    if output is not None:
        output.record_consumption(row)
