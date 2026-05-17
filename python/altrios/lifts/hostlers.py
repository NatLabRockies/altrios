"""Hostler dispatch helpers shared by container handling and OC staging."""


def get_hostler(terminal):
    """Prefer an already-active hostler; fall back to a parked one."""
    parked_available = len(terminal.state.parked_hostlers.items) > 0
    active_available = len(terminal.state.active_hostlers.items) > 0
    if (not active_available) and parked_available:
        assigned_hostler = terminal.state.parked_hostlers.get()
    else:
        assigned_hostler = terminal.state.active_hostlers.get()

    return assigned_hostler


def return_hostler(env, terminal, assigned_hostler, travel_time_to_active,
                   travel_time_to_parking, active_hostlers_needed=None):
    if active_hostlers_needed is None:
        active_hostlers_needed = len(terminal.state.active_hostlers.get_queue) > 0
    if active_hostlers_needed:
        yield env.timeout(travel_time_to_active)
        yield terminal.state.active_hostlers.put(assigned_hostler)
    else:
        yield env.timeout(travel_time_to_parking)
        yield terminal.state.parked_hostlers.put(assigned_hostler)
