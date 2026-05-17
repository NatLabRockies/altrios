"""Truck-side processes: gate ingress/egress and per-train truck spawning."""
import random

from altrios.lifts import utilities
from altrios.lifts.classes import container, truck
from altrios.lifts.emissions import _record_trip_emissions


def truck_entry(env, terminal, truck_obj, oc, train_schedule):
    """One loaded truck enters the gate, drops its OC at a parking slot."""
    train_id = train_schedule["train_id"]
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(0, terminal.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(travel_time)
    terminal.in_gates.release(ingate_request)

    utilities.record_container_event(terminal, oc.to_string(), 'truck_arrival', env.now)
    utilities.record_container_event(terminal, oc.to_string(), 'truck_dropoff', env.now)
    yield terminal.parking_slots.put(oc)
    _record_trip_emissions(terminal, truck_obj, "truck", "loaded", train_id,
                           oc.to_string(), "truck_entry", travel_time, env.now)


def empty_truck(env, terminal, truck_obj):
    """Empty truck just enters/leaves the gate to balance the IC>OC case."""
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(0, terminal.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(travel_time)
    terminal.in_gates.release(ingate_request)
    _record_trip_emissions(terminal, truck_obj, "truck", "empty",
                           getattr(truck_obj, "train_id", ""), "",
                           "empty_truck_entry", travel_time, env.now)


def truck_arrival(env, terminal, train_schedule):
    """Spawn all trucks that bring OCs (and any extra empty trucks) for a train."""
    train_id = train_schedule['train_id']
    truck_number = train_schedule["truck_number"]
    diesel_pct = terminal.TRUCK_DIESEL_PERCENTAGE
    num_diesel = round(truck_number * diesel_pct)
    num_electric = truck_number - num_diesel

    trucks = (
        [truck(type="Diesel", id=i, train_id=train_id) for i in range(num_diesel)] +
        [truck(type="Electric", id=i + num_diesel, train_id=train_id) for i in range(num_electric)]
    )

    oc_number = train_schedule["oc_number"]
    terminal.total_oc[train_id] = oc_number

    oc_start = terminal.OC_COUNT[train_id]
    for oc_id in range(oc_start, oc_start + oc_number):
        terminal.oc_store.put(container(type='Outbound', id=oc_id, train_id=train_id))
    terminal.OC_COUNT[train_id] = oc_start + oc_number

    truck_entries_needed = oc_number
    empty_truck_needed = max(0, len(trucks) - truck_entries_needed)

    for _ in range(truck_entries_needed):
        if not trucks:
            break
        this_truck = trucks.pop(0)
        oc = yield terminal.oc_store.get()
        env.process(truck_entry(env, terminal, this_truck, oc, train_schedule))
        yield terminal.truck_store.put(this_truck)

    for _ in range(empty_truck_needed):
        if not trucks:
            break
        this_truck = trucks.pop(0)
        env.process(empty_truck(env, terminal, this_truck))
        yield terminal.truck_store.put(this_truck)

    if not terminal.all_trucks_arrived_events[train_id].triggered:
        terminal.all_trucks_arrived_events[train_id].succeed()


def truck_exit(env, terminal, truck_obj, ic, train_schedule):
    train_id = train_schedule['train_id']
    out_gate_request = terminal.out_gates.request()
    yield out_gate_request
    travel_time = terminal.TRUCK_OUTGATE_TIME + random.uniform(0, terminal.TRUCK_OUTGATE_TIME_DEV)
    yield env.timeout(travel_time)
    terminal.out_gates.release(out_gate_request)
    utilities.record_container_event(terminal, ic.to_string(), 'truck_exit', env.now)
    _record_trip_emissions(terminal, truck_obj, "truck", "loaded",
                           train_id, ic.to_string(), "truck_exit",
                           travel_time, env.now)
    yield terminal.truck_store.put(truck_obj)
