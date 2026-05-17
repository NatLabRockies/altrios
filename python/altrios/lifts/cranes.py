"""Crane unload/load SimPy processes."""
import random

import simpy

from altrios.lifts import utilities
from altrios.lifts.containers import container_process
from altrios.lifts.emissions import _record_load_emissions


def crane_unload_process(env, terminal, train_schedule, track_id):
    """Drain all ICs for this train from train_ic_stores onto the chassis FilterStore."""
    train_id = train_schedule['train_id']

    def unload_crane_worker(env):
        crane_obj = yield terminal.cranes_by_track[track_id].get()
        try:
            while True:
                if not any(item.train_id == train_id for item in terminal.train_ic_stores.items):
                    break
                ic = yield terminal.train_ic_stores.get(lambda x: x.train_id == train_id)
                crane_unload_time = (terminal.CONTAINERS_PER_CRANE_MOVE_MEAN +
                                     random.uniform(0, terminal.CRANE_MOVE_DEV_TIME))
                yield env.timeout(crane_unload_time)
                yield terminal.chassis.put(ic)
                utilities.record_container_event(terminal, ic.to_string(), 'crane_unload', env.now)
                _record_load_emissions(terminal, crane_obj, "loaded", train_id,
                                       ic.to_string(), "crane_unload", env.now, track_id)
                env.process(container_process(env, terminal, train_schedule))
        finally:
            yield terminal.cranes_by_track[track_id].put(crane_obj)

    num_cranes = terminal.cranes_on_track[track_id]
    unload_processes = [env.process(unload_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, unload_processes)

    if not terminal.train_ic_unload_events[train_id].triggered:
        terminal.train_ic_unload_events[train_id].succeed()
        print(f"[Event] All ICs for Train-{train_id} on Track-{track_id} unloaded at {env.now:.3f}")


def crane_load_process(env, terminal, track_id, train_schedule):
    """Wait for OCs to be staged on chassis, then load them onto the train."""
    train_id = train_schedule['train_id']
    yield terminal.train_start_load_events[train_id]

    def load_crane_worker(env):
        crane_obj = yield terminal.cranes_by_track[track_id].get()
        try:
            while True:
                if not any(
                    (item.type == 'Outbound' and item.train_id == train_id)
                    for item in terminal.chassis.items
                ):
                    break
                oc = yield terminal.chassis.get(
                    lambda x: x.type == 'Outbound' and x.train_id == train_id
                )
                crane_load_time = (terminal.CONTAINERS_PER_CRANE_MOVE_MEAN +
                                   random.uniform(0, terminal.CRANE_MOVE_DEV_TIME))
                yield env.timeout(crane_load_time)
                yield terminal.train_oc_stores.put(oc)
                utilities.record_container_event(terminal, oc.to_string(), 'crane_load', env.now)
                _record_load_emissions(terminal, crane_obj, "loaded", train_id,
                                       oc.to_string(), "crane_load", env.now, track_id)
        finally:
            yield terminal.cranes_by_track[track_id].put(crane_obj)

    num_cranes = terminal.cranes_on_track[track_id]
    load_processes = [env.process(load_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, load_processes)

    if not terminal.train_end_load_events[train_id].triggered:
        terminal.train_end_load_events[train_id].succeed()
        print(f"[Event] All OCs for Train-{train_id} on Track-{track_id} loaded at {env.now:.3f}")
