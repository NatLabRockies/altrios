import simpy
import random
import polars as pl
from altrios.lifts import utilities
from altrios.lifts.distances import *
from altrios.lifts.dictionary import *
from altrios.lifts.schedule import *
from altrios.lifts.single_track_vehicle_performance import record_vehicle_event

# import sys
#
# if len(sys.argv) < 3:
#     raise ValueError("Not enough arguments. Please provide HOSTLER_NUMBER and CRANE_NUMBER.")
#
# HOSTLER_NUMBER = int(sys.argv[1])
# CRANE_NUMBER = int(sys.argv[2])

def record_event(container_id, event_type, timestamp):
    '''
    The simulation record logs for each container, including timestamps for arrival, loading, unloading, and departure.
    This data is used to calculate average processing times for inbound and outbound containers.
    When the function is called, it will record container_id and corresponding handling time.
    It is saved as an Excel file.
    '''
    global state
    if container_id is None:
        x = 5
    if container_id not in state.container_events:
        state.container_events[container_id] = {}
    state.container_events[container_id][event_type] = timestamp


def handle_truck_arrivals(env, in_gate_resource):
    '''
    Trucks arrive according to the poisson distribution between the timetable schedule.
    If all trucks are prepared, trigger all_trucks_ready_event.
    '''
    global state
    truck_id = 1
    state.TRUCK_ARRIVAL_MEAN = abs(state.TRAIN_ARRIVAL_HR - state.previous_train_departure) / max(state.INBOUND_CONTAINER_NUMBER, state.OUTBOUND_CONTAINER_NUMBER)
    print(f"Current time is {env.now}")
    print(f"Next TRAIN_ARRIVAL_HR:{state.TRAIN_ARRIVAL_HR}")
    print(f"TRUCK_ARRIVAL_MEAN IS {state.TRUCK_ARRIVAL_MEAN}")

    while truck_id <= state.TRUCK_NUMBERS:
        inter_arrival_time = random.expovariate(1 / state.TRUCK_ARRIVAL_MEAN)
        yield env.timeout(inter_arrival_time)
        state.truck_arrival_time.append(env.now)

        env.process(truck_through_gate(env, in_gate_resource, truck_id))
        truck_id += 1

    if truck_id > state.TRUCK_NUMBERS:
        # print(f"truck_id = {truck_id} vs TRUCK_NUM = {TRUCK_NUMBERS}")
        if not state.all_trucks_ready_event.triggered:
            state.all_trucks_ready_event.succeed()
            # print(f"{env.now}: All trucks arrived for the {TRAIN_ID} train.")


def truck_through_gate(env, in_gate_resource, truck_id):
    '''
    Objective: Trucks pass through the gate to enter and exit the terminal. The simulation tracks the time taken for each truck to pass through the gate.
    Steps:
    - Record truck arrival time
    - Check availability of the ingate resource
        - If there is a empty gate, enter the gate and finish procedures
        - If not, join the queuing module to record queuing time
    - After passing through the gate, put the container in the outbound queuehandle_container
    - Trucks drop outbound container before trains arrive, where the # of outbound containers equals to the # of inbound containers using bring_all_outbound_containers
    - Outbound container mapping creation: truck ID --> outbound container ID
    '''
    global state

    with in_gate_resource.request() as request:
        yield request
        wait_time = max(0, state.truck_arrival_time[truck_id - 1] - state.last_leave_time)
        if wait_time <= 0:
            wait_time = 0  # first arriving trucks
            # print(f"Truck {truck_id} enters the gate without waiting")
        else:
            # print(f"Truck {truck_id} enters the gate and queued for {wait_time} hrs")
            state.truck_waiting_time.append(wait_time)

        yield env.timeout(state.TRUCK_INGATE_TIME + random.uniform(0, state.TRUCK_INGATE_TIME_DEV))

        # Case 1: Normal handling when OC >= IC (all trucks have containers)
        if state.OUTBOUND_CONTAINER_NUMBER >= state.INBOUND_CONTAINER_NUMBER:
            env.process(handle_container(env, truck_id))

        # Case 2: OC < IC, extra empty trucks are needed
        else:
            if truck_id <= state.OUTBOUND_CONTAINER_NUMBER:
                env.process(handle_container(env, truck_id))  # Loaded trucks
            else:
                env.process(empty_truck(env, truck_id))  # Empty trucks


def handle_container(env, truck_id):
    '''
    The process of track dropping off OCs before train arrives records time which follows triangle distribution.
    It considers individual differences in container processing, considering (min, avg, max)
    '''
    global state

    container_id = state.outbound_container_id_counter
    if container_id is None:
        x = 5
    state.outbound_container_id_counter += 1
    record_event(container_id, 'truck_arrival', env.now)

    d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    yield env.timeout(d_t_dist / (2 * state.TRUCK_SPEED_LIMIT))

    record_event(container_id, 'truck_drop_off', env.now)
    # print(f"{env.now}: Truck {truck_id} drops outbound container {container_id}.")
    state.last_leave_time = env.now


def empty_truck(env, truck_id):
    '''
    Trucks without OCs enter the gate. These trucks are assigned to balance the IC and OC gap.
    '''
    global state

    d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    yield env.timeout(d_t_dist / (2 * state.TRUCK_SPEED_LIMIT))

    # print(f"{env.now}: Empty truck {truck_id} arrives.")
    state.last_leave_time = env.now


def train_arrival(env, train_timetable, train_processing, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, out_gate_resource):
    '''
    Trains arrive according to the timetable schedule.
    '''
    global state

    for i, train in enumerate(train_timetable):
        state.TRAIN_ARRIVAL_HR = train['arrival_time']
        state.TRAIN_DEPARTURE_HR = train['departure_time']
        state.INBOUND_CONTAINER_NUMBER = train['full_cars']
        state.OUTBOUND_CONTAINER_NUMBER = train['oc_number']
        state.TRUCK_NUMBERS = train['truck_number']
        state.TRAIN_ID = train['train_id']

        print(f"---------- Next Train {state.TRAIN_ID} Is On the  Way ----------")
        print(f"IC {state.INBOUND_CONTAINER_NUMBER}")
        print(f"OC {state.OUTBOUND_CONTAINER_NUMBER}")

        outbound_containers_store.items.clear()
        for oc in range(state.record_oc_label, state.record_oc_label + state.OUTBOUND_CONTAINER_NUMBER):  # from 10001 to 10001 + OC
            # print("oc_number", oc)
            outbound_containers_store.put(oc)
            # yield outbound_containers_store.put(oc)
            # print(f"Current store contents after putting {oc}: {outbound_containers_store.items}")

        # print("outbound_containers_store is:", outbound_containers_store.items)

        # Trucks enter until the precious train departs, if not the first truck
        state.previous_train_departure = train_timetable[i-1]['departure_time'] if i > 0 else 0
        print(f"Schedule {state.TRUCK_NUMBERS} Trucks arriving between previous train departure at {state.previous_train_departure} and current train arrival at {state.TRAIN_ARRIVAL_HR}")
        env.process(handle_truck_arrivals(env, in_gate_resource))

        # Trains arrive according to the timetable, fix negative delay bug
        delay = state.TRAIN_ARRIVAL_HR - env.now
        if delay <= 0:
            yield env.timeout(0)
        else:
            yield env.timeout(delay)

        train_id = state.train_id_counter
        print(f"Train {state.TRAIN_ID} ({train_id} in the dictionary) arrives at {env.now}")

        # for container_id in range(inbound_container_id_counter, inbound_container_id_counter + INBOUND_CONTAINER_NUMBER):
        for container_id in range(int(state.inbound_container_id_counter), int(state.inbound_container_id_counter) + int(state.INBOUND_CONTAINER_NUMBER)):  # fix float error

            record_event(container_id, 'train_arrival', env.now)

        with train_processing.request() as request:
            yield request
            state.oc_chassis_filled_event = env.event()
            yield env.process(process_train(env, train_id, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, train_processing, state.oc_chassis_filled_event, out_gate_resource))
            state.train_id_counter += 1

        state.record_oc_label += state.OUTBOUND_CONTAINER_NUMBER
        # print("record_oc_label", record_oc_label)
        # print("oc_variance in train_process:", oc_variance)


def process_train(env, train_id, cranes, hostlers, chassis, in_gate_resource, outbound_containers_store, truck_store, train_processing, oc_chassis_filled_event, out_gate_resource):
    '''
    This function is used to take charge of all processing from train side to inland side.
    STEP 1: A train arrives and calls this function process_train
    STEP 2: Cranes start moving and drop off IC to chassis
    STEP 3: Hostlers pick up IC and drop off OC to chassis
    STEP 4: Trucks pick up IC and leave gates
    STEP 5: Once all OC are prepared, chassis uploads OC, and train departs
    '''
    global state

    start_time = env.now

    # Cranes unload all IC
    unload_processes = []
    chassis_inbound_ids = []  # To save chassis_id, current_inbound_id to hostler_transfer_IC_single_loop

    # if train_id < TRAIN_NUMBERS:
    for chassis_id in range(1, int(state.INBOUND_CONTAINER_NUMBER) + 1):
        unload_process = env.process(crane_and_chassis(env, train_id, 'unload', cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, out_gate_resource, oc_chassis_filled_event))
        unload_processes.append(unload_process)

    # All IC are processed
    # print("Unload process is:", unload_processes)
    yield simpy.events.AllOf(env, unload_processes)
    results = yield simpy.events.AllOf(env, unload_processes)

    # To pass chassis_id, current_inbound_id to hostler_transfer_IC_single_loop as a list from calling chassis_inbound_ids
    for result in results.values():
        chassis_id, current_inbound_id = result
        chassis_inbound_ids.append((chassis_id, current_inbound_id))
        env.process(hostler_transfer(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id, truck_store, cranes,
                             train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event,
                             out_gate_resource))

    # # Once all OC are dropped by hostlers, crane start working
    # print("Chassis are filled with OC (-1) now. ")
    # print(f"Chassis status after OC processed is: {chassis_status}, where ")
    # print(f"there are {chassis_status.count(0)} chassis is filled with OC (0)")
    # print(f"there are {chassis_status.count(-1)} chassis is filled with empty (-1)")
    # print(f"there are {chassis_status.count(1)} chassis is filled with IC (1)")

    # Cranes move all OC to chassis
    load_processes = []
    for chassis_id in range(1, state.OUTBOUND_CONTAINER_NUMBER + 1):
        load_process = env.process(crane_and_chassis(env, train_id, 'load', cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource, chassis_id=chassis_id))
        load_processes.append(load_process)
    yield simpy.events.AllOf(env, load_processes)

    # Check if all outbound containers are loaded (all chassis is empty 0), the train departs
    if state.chassis_status.count(-1) == state.TRAIN_UNITS:
        # oc_chassis_filled_event.succeed()
        state.TRAIN_ID_FIXED = state.TRAIN_ID
        print(f"Train {state.TRAIN_ID_FIXED} is ready to depart at {env.now}.")
        env.process(train_departure(env, train_id))
        state.time_per_train.append(env.now - start_time)

    end_time = env.now
    state.time_per_train.append(end_time - start_time)
    state.train_series += 1
    state.oc_variance += state.OUTBOUND_CONTAINER_NUMBER


def crane_and_chassis(env, train_id, action, cranes, hostlers, chassis, truck_store, train_processing, outbound_containers_store, in_gate_resource, out_gate_resource, oc_chassis_filled_event, chassis_id=None):
    '''
    This function is used to provide loading and uploading processes of cranes and chassis.
    Unload process happens when a train arrives and ICs on the train is moved to the chassis.
    Load process happens when a train arrives and OCs on the chassis are moved to the train.
    The simplified function could refer to the demo.py.
    '''

    global state

    # # Print before requesting crane resource
    if action == 'unload':
        crane_id = state.crane_id_counter
        state.crane_id_counter = (state.crane_id_counter % state.CRANE_NUMBER) + 1
        # print("inbound_id_counter", inbound_container_id_counter)
        for container_id in range(int(state.inbound_container_id_counter), int(state.inbound_container_id_counter) + int(state.INBOUND_CONTAINER_NUMBER)):  # fix float error
            # print("container_id now:", container_id)
            yield env.timeout(state.CRANE_UNLOAD_CONTAINER_TIME_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME))
            record_event(container_id, 'crane_unload', env.now)
            # print(f"Crane {crane_id} unloads inbound container {inbound_container_id_counter} from train {train_id} at {env.now}")

    # if action == 'load':
    #     for container_id in range(record_oc_label, record_oc_label + OUTBOUND_CONTAINER_NUMBER):
    #         yield env.timeout(CRANE_LOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))
    #         chassis_status[chassis_id - 1] = -1
    #         # print(f"Crane {crane_id} loads outbound container {container_id} to train {TRAIN_ID} at {env.now}")
    #         record_event(container_id, 'crane_load', env.now)

    with cranes.request() as request:
        yield request

        # # Print after acquiring crane resource
        # print(f"[{env.now}] Crane {crane_id_counter} acquired crane resource. Available cranes: {cranes.count}/{cranes.capacity}")

        start_time = env.now
        record_vehicle_event('crane', state.crane_id_counter, 'start', start_time)    # performance record: starting

        if action == 'unload':
            # crane_id = crane_id_counter
            # crane_id_counter = (crane_id_counter % CRANE_NUMBER) + 1

            chassis_id = ((state.inbound_container_id_counter - 1) % state.CHASSIS_NUMBER) + 1

            current_inbound_id = state.inbound_container_id_counter
            state.inbound_container_id_counter += 1
            # yield env.timeout(CRANE_UNLOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))

            # for chassis_id in range(int(inbound_container_id_counter), int(inbound_container_id_counter) + int(INBOUND_CONTAINER_NUMBER)):
            state.chassis_status[chassis_id - 1] = 1

            end_time = env.now
            record_vehicle_event('crane', state.crane_id_counter, 'end', end_time)     # performance record: ending

            # hostler picks up IC
            env.process(hostler_transfer(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource))

            return chassis_id, current_inbound_id

        elif action == 'load':
            if chassis_id not in state.outbound_containers_mapping:
                print(f"Notice: No outbound container mapped to chassis {chassis_id} at {env.now}")
                return

            container_id = state.outbound_containers_mapping[chassis_id]  # Retrieve container ID from mapping
            # print("outbound_containers_mapping in crane and chassis func:", outbound_containers_mapping)
            # print("container_id in crane and chassis func:", container_id)

            if state.CRANE_NUMBER == 1:
                crane_id = 1
            else:
                crane_id = (chassis_id % state.CRANE_NUMBER) + 1

            state.chassis_status[chassis_id - 1] = -1

            # yield env.timeout(CRANE_LOAD_CONTAINER_TIME_MEAN + random.uniform(0, CRANE_MOVE_DEV_TIME))
            # chassis_status[chassis_id - 1] = -1
            # print(f"Crane {crane_id} loads outbound container {container_id} from chassis {chassis_id} to train {TRAIN_ID} at {env.now}")
            # record_event(container_id, 'crane_load', env.now)

            for container_id in range(state.record_oc_label, state.record_oc_label + state.OUTBOUND_CONTAINER_NUMBER):
                yield env.timeout(state.CRANE_LOAD_CONTAINER_TIME_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME))
                # chassis_status[chassis_id - 1] = -1
                # print(f"Crane {crane_id} loads outbound container {container_id} to train {TRAIN_ID} at {env.now}")
                record_event(container_id, 'crane_load', env.now)

        # # At this point, the crane resource should be released
        # print(f"[{env.now}] Crane {crane_id_counter} has released crane resource. Available cranes: {cranes.count}/{cranes.capacity}")


def hostler_transfer(env, hostlers, container_type, chassis, chassis_id, container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource):
    '''
    Once IC is put on the chassis, a hostler from hostler source is assigned to pick up the IC.
    '''
    global state

    with hostlers.request() as request:
        yield request
        if container_id is None:
            x =5
        start_time = env.now
        record_vehicle_event('hostler', state.hostler_id_counter, 'start', start_time)  # performance record

        hostler_id = state.hostler_id_counter
        state.hostler_id_counter = (state.hostler_id_counter % state.HOSTLER_NUMBER) + 1

        with chassis.request() as chassis_request:
            yield chassis_request

            if container_type == "inbound":
                x = 5

            if container_type == 'inbound' and state.chassis_status[chassis_id - 1] == 1:
                d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
                state.HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * state.HOSTLER_SPEED_LIMIT)
                print(f"Hostler pick-up time is:{state.HOSTLER_TRANSPORT_CONTAINER_TIME}")
                yield env.timeout(state.HOSTLER_TRANSPORT_CONTAINER_TIME)
                record_event(container_id, 'hostler_pickup', env.now)
                print(f"Hostler {hostler_id} picks up inbound container {container_id} from chassis {chassis_id} and heads to parking area at {env.now}")

                state.chassis_status[chassis_id - 1] = -1

                # Hostler drop off: different route for picking-up and dropping-off
                d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
                state.HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * state.HOSTLER_SPEED_LIMIT)
                print(f"Hostler drop-off time is:{state.HOSTLER_TRANSPORT_CONTAINER_TIME}")
                yield env.timeout(state.HOSTLER_TRANSPORT_CONTAINER_TIME)
                if container_id is None:
                    x =5
                record_event(container_id, 'hostler_dropoff', env.now)
                print(f"Hostler {hostler_id} drops off inbound container {container_id} from chassis {chassis_id} and moves toward the assigned outbound container at {env.now}")

                end_time = env.now
                record_vehicle_event('hostler', state.hostler_id_counter, 'end', end_time)  # performance record

                # Process functions of notify_truck and handle_outbound_container simultaneously
                env.process(notify_truck(env, truck_store, container_id, out_gate_resource))

                # Assign outbound container and chassis_id for the hostler which drops off an inbound container
                chassis_id, state.outbound_container_id = yield env.process(outbound_container_decision_making(
                    env, hostlers, chassis, container_id, truck_store, cranes, train_processing,
                    outbound_containers_store,
                    in_gate_resource, oc_chassis_filled_event, out_gate_resource))

                # Process outbound containers
                if chassis_id is not None and state.outbound_container_id is not None:
                    env.process(handle_outbound_container(env, hostler_id, chassis_id, state.outbound_container_id, truck_store,
                                                  cranes, train_processing, outbound_containers_store, in_gate_resource))


# When OC are fully processed, but IC are not
def hostler_transfer_IC_single_loop(env, hostlers, container_type, chassis, chassis_id, container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource):
    '''
    This function is designed for a case that when OCs are fully processed, but ICs on the chassis are waiting to be transported.
    '''
    print(f"Starting single hostler transfer IC loop for chassis {chassis_id} at {env.now}")
    global state

    print(f"Requesting hostler for IC at chassis {chassis_id} at {env.now}")

    with hostlers.request() as request:
        print(f"Request available hostlers: {hostlers.count} vs total hostlers {state.HOSTLER_NUMBER}, Hostlers capacity: {hostlers.capacity} at {env.now}")
        yield request

        start_time = env.now
        record_vehicle_event('hostler', state.hostler_id_counter, 'start', start_time)  # performance record

        hostler_id = state.hostler_id_counter
        state.hostler_id_counter = (state.hostler_id_counter % state.HOSTLER_NUMBER) + 1

        with chassis.request() as chassis_request:
            yield chassis_request

            if container_type == 'inbound' and state.chassis_status[chassis_id - 1] == 1:
                state.chassis_status[chassis_id - 1] = -1
                print(f"Single loop chassis status {state.chassis_status}")
                print(f"There are {state.chassis_status.count(1)} IC")
                print(f"There are {state.chassis_status.count(-1)} empty")
                print(f"There are {state.chassis_status.count(0)} OC")
                d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
                state.HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * state.HOSTLER_SPEED_LIMIT)

                yield env.timeout(state.HOSTLER_TRANSPORT_CONTAINER_TIME)
                # hostler picks up the rest of IC from the chassis
                # chassis_status[chassis_id - 1] = -1
                record_event(container_id, 'hostler_pickup', env.now)
                print(f"Hostler {hostler_id} picks up inbound container {container_id} from chassis {chassis_id} to parking area at {env.now}")

                # hostler drops off the IC
                d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
                state.HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * state.HOSTLER_SPEED_LIMIT)
                yield env.timeout(state.HOSTLER_TRANSPORT_CONTAINER_TIME)
                record_event(container_id, 'hostler_dropoff', env.now)
                print(f"Hostler {hostler_id} drops off inbound container {container_id} from chassis {chassis_id} to parking area at {env.now}")

                # Check if all chassis filled
                if state.chassis_status.count(0) == state.OUTBOUND_CONTAINER_NUMBER and state.chassis_status.count(
                        -1) == state.TRAIN_UNITS - state.OUTBOUND_CONTAINER_NUMBER and not oc_chassis_filled_event.triggered:
                    print(f"Chassis is fully filled with OC, and cranes start moving: {state.chassis_status}")
                    print(f"where there are {state.chassis_status.count(0)} chassis filled with OC (0)")
                    print(f"where there are {state.chassis_status.count(-1)} chassis filled with empty (-1)")
                    print(f"where there are {state.chassis_status.count(1)} chassis filled with IC (1)")
                    oc_chassis_filled_event.succeed()
                    return
                else:
                    print(f"Chassis is not fully filled: {state.chassis_status}")
                    print(f"where there are {state.chassis_status.count(0)} chassis filled with OC (0)")
                    print(f"where there are {state.chassis_status.count(-1)} chassis filled with empty (-1)")
                    print(f"where there are {state.chassis_status.count(1)} chassis filled with IC (1)")

                end_time = env.now
                record_vehicle_event('hostler', hostler_id, 'end', end_time)  # performance record

                # trucks pick up IC
                yield env.process(notify_truck(env, truck_store, container_id, out_gate_resource))


def outbound_container_decision_making(env, hostlers, chassis, current_inbound_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource):
    '''
    This function is designed to assign hostlers for a given OC, and which chassis will be dropped off.
    '''
    global state
    # Check if outbound_containers_store has outbound container
    if len(outbound_containers_store.items) > 0:
        outbound_container_id = yield outbound_containers_store.get()
        print(f"Outbound containers remaining: {len(outbound_containers_store.items)}")

        if -1 in state.chassis_status:
            chassis_id = state.chassis_status.index(-1) + 1  # find the first chassis
            # If chassis are not assigned with outbound container
            if chassis_id not in state.outbound_containers_mapping:
                # outbound_container_id += state.record_oc_label
                state.outbound_containers_mapping[chassis_id] = outbound_container_id
                state.chassis_status[chassis_id - 1] = 0  # already assigned outbound container
                print(f"OC mapping created: outbound container {outbound_container_id} assigned to chassis {chassis_id}")
            else:
                print(f"Chassis {chassis_id} is already mapped to an outbound container.")
        else:
            print("No empty chassis available for outbound container assignment.")

    # if outbound_containers_store is null, check if we need operate single loop
    else:
        chassis_id = None
        outbound_container_id = None
        # chassis_status = 1: inbound containers are not loaded
        if state.chassis_status.count(1) != 0:
            print(f"Haven't finished all IC yet at {env.now}. Starting single loop.")
            chassis_id = state.chassis_status.index(1) + 1
            state.chassis_status[chassis_id - 1] = 0  # assigned with IC
            # single loop takes rest inbound container
            yield env.process(hostler_transfer_IC_single_loop(env, hostlers, 'inbound', chassis, chassis_id, current_inbound_id,
                                                truck_store, cranes, train_processing,
                                                outbound_containers_store, in_gate_resource, oc_chassis_filled_event, out_gate_resource))
        else:
            print("All inbound containers have been processed.")

    if outbound_container_id is None:
        x = 5
    return chassis_id, outbound_container_id


def handle_outbound_container(env, hostler_id, chassis_id, outbound_container_id, truck_store, cranes, train_processing, outbound_containers_store, in_gate_resource):
    '''
    This function is designed to record container processing time for the outbound container assignment.
    '''

    global state

    d_h_dist = create_triang_distribution(d_h_min, d_h_avg, d_h_max).rvs()
    state.HOSTLER_TRANSPORT_CONTAINER_TIME = d_h_dist / (2 * state.HOSTLER_SPEED_LIMIT)

    d_r_dist = create_triang_distribution(d_r_min, d_r_avg, d_r_max).rvs()
    state.HOSTLER_FIND_CONTAINER_TIME = d_r_dist / (2 * state.TRUCK_SPEED_LIMIT)
    yield env.timeout(state.HOSTLER_FIND_CONTAINER_TIME)

    record_event(outbound_container_id, 'hostler_pickup', env.now)
    print(f"Hostler {hostler_id} picks up outbound container {outbound_container_id} from parking area to chassis {chassis_id} at {env.now}")

    yield env.timeout(state.HOSTLER_TRANSPORT_CONTAINER_TIME)

    record_event(outbound_container_id, 'hostler_dropoff', env.now)
    print(f"Hostler {hostler_id} drops off outbound container {outbound_container_id} to chassis {chassis_id} at {env.now}")


# truck pick up IC
def notify_truck(env, truck_store, container_id, out_gate_resource):
    '''
    notify trucks when ICs are dropped off at the parking spot.
    '''
    global state
    truck_id = yield truck_store.get()
    yield env.timeout(state.TRUCK_INGATE_TIME)
    print(f"Truck {truck_id} arrives at parking area and prepare to pick up inbound container {container_id} at {env.now}")
    yield env.process(truck_transfer(env, truck_id, container_id, out_gate_resource))


def truck_transfer(env, truck_id, container_id, out_gate_resource):
    '''
    Truck transfer function for IC transfer.
    Record processing time and consider gate queues.
    '''
    global state

    start_time = env.now
    record_vehicle_event('truck', truck_id, 'start', start_time)  # performance record

    # Truck moves to the parking area
    yield env.timeout(state.TRUCK_TO_PARKING)
    record_event(container_id, 'truck_pickup', env.now)
    print(f"Truck {truck_id} picks up inbound container {container_id} at {env.now}")

    # Calculate the transport time for the truck
    d_t_dist = create_triang_distribution(d_t_min, d_t_avg, d_t_max).rvs()
    state.TRUCK_TRANSPORT_CONTAINER_TIME = d_t_dist / (2 * state.TRUCK_SPEED_LIMIT)
    yield env.timeout(state.TRUCK_TRANSPORT_CONTAINER_TIME)

    # Request out_gate_resource resource before the truck exits
    with out_gate_resource.request() as request:
        yield request

        # Simulate the time it takes for the truck to pass through the gate
        yield env.timeout(state.TRUCK_OUTGATE_TIME + random.uniform(0,state.TRUCK_OUTGATE_TIME_DEV))
        record_event(container_id, 'truck_exit', env.now)
        print(f"Truck {truck_id} exits gate with inbound container {container_id} at {env.now}")

    # End performance recording
    end_time = env.now
    record_vehicle_event('truck', truck_id, 'end', end_time)


def train_departure(env, train_id):
    '''
    Train departs according to the timetable. Delay otherwise.
    '''
    global state

    if env.now < state.TRAIN_DEPARTURE_HR:
        yield env.timeout(state.TRAIN_DEPARTURE_HR - env.now)
    yield env.timeout(state.TRAIN_INSPECTION_TIME)
    print(f"Train {state.TRAIN_ID_FIXED} ({train_id} in the dictionary) departs at {env.now}")

    for container_id in range(state.record_oc_label - state.OUTBOUND_CONTAINER_NUMBER, state.record_oc_label):
        record_event(container_id, 'train_depart', env.now)


def run_simulation(
        train_consist_plan: pl.DataFrame,
        terminal: str,
        out_path = None):
    '''
    This function is to run the simulation for
    '''
    global state
    state.terminal = terminal
    state.initialize_from_consist_plan(train_consist_plan)

    print(f"Starting simulation with No.{state.TRAIN_ID} trains, {state.HOSTLER_NUMBER} hostlers, {state.CRANE_NUMBER} cranes, and {state.TRUCK_NUMBERS} trucks.")
    env = simpy.Environment()

    # Resources
    train_processing = simpy.Resource(env, capacity=1)
    cranes = simpy.Resource(env, capacity=state.CRANE_NUMBER)
    chassis = simpy.Resource(env, capacity=state.CHASSIS_NUMBER)
    hostlers = simpy.Resource(env, capacity=state.HOSTLER_NUMBER)
    in_gate_resource = simpy.Resource(env, capacity=state.IN_GATE_NUMBERS)
    out_gate_resource = simpy.Resource(env, capacity=state.OUT_GATE_NUMBERS)
    outbound_containers_store = simpy.Store(env, capacity=100)
    truck_store = simpy.Store(env, capacity=100)

    # Initialize trucks
    truck_store.items.clear()
    # print("TRUCK_NUMBERS:",  TRUCK_NUMBERS)
    for truck_id in range(1, 100 + 1):
        truck_store.put(truck_id)
    # print("TRUCK_STORE:", truck_store.items)

    state.all_trucks_ready_event = env.event()

    # # toy case
    # train_timetable = [
    #     {"train_id": 19, "arrival_time": 187, "departure_time": 200, "empty_cars": 3, "full_cars":7, "oc_number": 2, "truck_number":7 },
    #     {"train_id": 25, "arrival_time": 250, "departure_time": 350, "empty_cars": 4, "full_cars":6, "oc_number": 2, "truck_number":6 },
    #     {"train_id": 49, "arrival_time": 400, "departure_time": 600, "empty_cars": 5, "full_cars":5, "oc_number": 2, "truck_number":5 },
    #     {"train_id": 60, "arrival_time": 650, "departure_time": 750, "empty_cars": 6, "full_cars":4, "oc_number": 2, "truck_number":4 },
    #     {"train_id": 12, "arrival_time": 800, "departure_time": 1000, "empty_cars": 7, "full_cars":3, "oc_number": 4, "truck_number":4 },
    # ]

    # REAL TEST
    train_timetable = build_train_timetable(train_consist_plan, terminal, swap_arrive_depart = True, as_dicts = True)
    TRAIN_NUMBERS = len(train_timetable)

    # env.process(train_arrival(env, train_processing, cranes, in_gate_resource, outbound_containers_store, truck_store, train_timetable))
    env.process(train_arrival(env, train_timetable, train_processing, cranes, hostlers, chassis, in_gate_resource,
                  outbound_containers_store, truck_store, out_gate_resource))

    env.run(until=state.sim_time)

    # Performance Matrix: train processing time
    avg_time_per_train = sum(state.time_per_train) / len(state.time_per_train)
    print(f"Average train processing time: {sum(state.time_per_train) / len(state.time_per_train) if state.time_per_train else 0:.2f}")
    print("Simulation completed. ")
    with open("avg_time_per_train.txt", "w") as f:
        f.write(str(avg_time_per_train))

    # Create DataFrame for container events
    container_data = (
        pl.from_dicts(
            [dict(event, **{'container_id': container_id}) for container_id, event in state.container_events.items()]
        )
        .with_columns(
            pl.when(pl.col("container_id") < 10001).then(pl.lit("inbound")).otherwise(pl.lit("outbound")).alias("container_type")
        )
        .with_columns(
            pl.when(
                pl.col("container_type") == pl.lit("inbound"),
                pl.col("truck_exit").is_not_null(),
                pl.col("train_arrival").is_not_null()
            )
            .then(
                pl.col("truck_exit") - pl.col("train_arrival")
            )
            .when(
                pl.col("container_type") == pl.lit("outbound"),
                pl.col("train_depart").is_not_null(),
                pl.col("truck_drop_off").is_not_null()
            )
            .then(
                pl.col("train_depart") - pl.col("truck_drop_off")
            )
            .otherwise(None)
            .alias("container_processing_time")
        )
        .sort("container_id")
        .select(pl.col("container_id", "container_type"), pl.all().exclude("container_id", "container_type"))
    )
    if out_path is not None:
        container_data.write_excel(out_path / f"simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx")

    save_vehicle_logs()

    print("Done!")
    return container_data


if __name__ == "__main__":
    run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'demos' / 'starter_demo' / 'train_consist_plan.csv'),
        terminal = "Allouez",
        out_path = utilities.package_root() / 'demos' / 'starter_demo' / 'results'
    )