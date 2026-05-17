"""Crane unload/load SimPy processes."""
import random

import simpy

from altrios.lifts import utilities
from altrios.lifts.classes import loggingLevel
from altrios.lifts.containers import container_process
from altrios.lifts.emissions import _record_load_emissions


def crane_unload_process(env, terminal, train_schedule, track_id):
    """Drain all ICs for this train from the per-train IC store onto the shared chassis IC store."""
    train_id = train_schedule['train_id']
    state = terminal.state
    ic_store = state.train_ic_store(train_id)

    def unload_crane_worker(env):
        crane_obj = yield state.cranes_by_track[track_id].get()
        try:
            while True:
                if not ic_store.items:
                    break
                ic = yield ic_store.get()
                crane_unload_time = (terminal.CONTAINERS_PER_CRANE_MOVE_MEAN +
                                     random.uniform(0, terminal.CRANE_MOVE_DEV_TIME))
                yield env.timeout(crane_unload_time)
                yield state.chassis_ic_store.put(ic)
                state.chassis_ic_count_by_train[train_id] = (
                    state.chassis_ic_count_by_train.get(train_id, 0) + 1
                )
                utilities.record_container_event(terminal, ic.to_string(), 'crane_unload', env.now)
                _record_load_emissions(terminal, crane_obj, "loaded", train_id,
                                       ic.to_string(), "crane_unload", env.now, track_id)
                env.process(container_process(env, terminal, train_schedule))
        finally:
            yield state.cranes_by_track[track_id].put(crane_obj)

    num_cranes = terminal.cranes_on_track[track_id]
    unload_processes = [env.process(unload_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, unload_processes)

    if not terminal.state.train_ic_unload_events[train_id].triggered:
        terminal.state.train_ic_unload_events[train_id].succeed()
        terminal.log(loggingLevel.BASIC, f"[Event] All ICs for Train-{train_id} on Track-{track_id} unloaded at {env.now:.3f}")


def crane_load_process(env, terminal, track_id, train_schedule):
    """Wait for OCs to be staged on chassis, then load them onto the train."""
    train_id = train_schedule['train_id']
    state = terminal.state
    yield state.train_start_load_events[train_id]

    oc_store = state.chassis_oc_store(train_id)

    def load_crane_worker(env):
        crane_obj = yield state.cranes_by_track[track_id].get()
        try:
            while True:
                if not oc_store.items:
                    break
                oc = yield oc_store.get()
                state.chassis_oc_count_by_train[train_id] = (
                    state.chassis_oc_count_by_train.get(train_id, 0) - 1
                )
                crane_load_time = (terminal.CONTAINERS_PER_CRANE_MOVE_MEAN +
                                   random.uniform(0, terminal.CRANE_MOVE_DEV_TIME))
                yield env.timeout(crane_load_time)
                yield state.train_oc_stores.put(oc)
                utilities.record_container_event(terminal, oc.to_string(), 'crane_load', env.now)
                _record_load_emissions(terminal, crane_obj, "loaded", train_id,
                                       oc.to_string(), "crane_load", env.now, track_id)
        finally:
            yield state.cranes_by_track[track_id].put(crane_obj)

    num_cranes = terminal.cranes_on_track[track_id]
    load_processes = [env.process(load_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, load_processes)

    if not terminal.state.train_end_load_events[train_id].triggered:
        terminal.state.train_end_load_events[train_id].succeed()
        terminal.log(loggingLevel.BASIC, f"[Event] All OCs for Train-{train_id} on Track-{track_id} loaded at {env.now:.3f}")
