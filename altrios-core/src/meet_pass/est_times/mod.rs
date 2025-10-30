#![allow(unused_imports)]

use super::disp_imports::*;
use crate::consist::Consist;
use crate::track::Network;
use uc::SPEED_DIFF_JOIN;
use uc::TIME_NAN;

pub(crate) mod est_time_structs;
mod update_times;

use est_time_structs::*;
use update_times::*;

/// Estimated time node for dispatching
/// Specifies the expected time of arrival when taking the shortest path with no delays
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct EstTime {
    /// Scheduled time of arrival at the node
    pub time_sched: si::Time,
    /// Time required to get to the next node when passing at speed
    pub time_to_next: si::Time,
    /// Distance to the next node
    pub dist_to_next: si::Length,
    /// Speed at which the train will pass this node assuming no delays
    pub speed: si::Velocity,

    /// Index of link leaving the next node in the network when traveling along the shortest path from this node
    pub idx_next: EstIdx,
    /// Index of alternative link leaving next node (if it exists)
    /// Used if the shortest path is blocked up ahead
    pub idx_next_alt: EstIdx,
    /// Index of link leaving the previous node if the shortest path was taken to reach this node
    pub idx_prev: EstIdx,
    /// Index of link leaving the alternate previous node (if it exists)
    pub idx_prev_alt: EstIdx,

    /// Combination of link index and est type for this node
    /// Fake events have null link index
    pub link_event: LinkEvent,
}

impl EstTime {
    pub fn time_sched_next(&self) -> si::Time {
        self.time_sched + self.time_to_next
    }
}
impl Default for EstTime {
    fn default() -> Self {
        Self {
            time_sched: TIME_NAN,
            time_to_next: si::Time::ZERO,
            dist_to_next: si::Length::ZERO,
            speed: si::Velocity::ZERO,
            idx_next: EST_IDX_NA,
            idx_next_alt: EST_IDX_NA,
            idx_prev: EST_IDX_NA,
            idx_prev_alt: EST_IDX_NA,
            link_event: Default::default(),
        }
    }
}

#[serde_api]
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "pyo3", pyclass(module = "altrios", subclass, eq))]
pub struct EstTimeNet {
    pub val: Vec<EstTime>,
}

#[pyo3_api]
impl EstTimeNet {
    pub fn get_running_time_hours(&self) -> f64 {
        (self.val.last().unwrap().time_sched - self.val.first().unwrap().time_sched)
            .get::<si::hour>()
    }
}

impl Init for EstTimeNet {}
impl SerdeAPI for EstTimeNet {}

impl EstTimeNet {
    pub fn new(val: Vec<EstTime>) -> Self {
        Self { val }
    }
}

#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn check_od_pair_valid(
    origs: Vec<Location>,
    dests: Vec<Location>,
    network: Vec<Link>,
) -> anyhow::Result<()> {
    if let Err(error) = get_link_idx_options(&origs, &dests, &network) {
        Err(error)
    } else {
        Ok(())
    }
}

/// Get link indexes that lead to the destination (CURRENTLY ALLOWS LOOPS THAT
/// ARE TOO SMALL TO FIT THE TRAIN!)
pub fn get_link_idx_options(
    origs: &[Location],
    dests: &[Location],
    links: &[Link],
) -> Result<(IntSet<LinkIdx>, Vec<Location>), anyhow::Error> {
    // Allocate our empty link_idxs processing vector with enough initial space.
    let mut link_idxs_proc = Vec::<LinkIdx>::with_capacity(64.max(dests.len()));
    // Ensure our link_idx_set is initialized with the same capacity. We use
    // the set to ensure we only process each link_idx once.
    let mut link_idx_set =
        IntSet::<LinkIdx>::with_capacity_and_hasher(link_idxs_proc.capacity(), Default::default());
    // Go ahead and put all of the destination link indexes into our processing vector.
    link_idxs_proc.extend(dests.iter().map(|x| x.link_idx));

    // Keep track of the updated length of the origins.
    let mut origs = origs.to_vec();
    let mut origs_len_new = 0;

    // Now, pop each link_idx from the processing vector and process it.
    while let Some(link_idx) = link_idxs_proc.pop() {
        // If the link_idx has not yet been processed, process it.
        if !link_idx_set.contains(&link_idx) {
            link_idx_set.insert(link_idx);
            let origs_start = origs_len_new;
            for i in origs_start..origs.len() {
                if link_idx == origs[i].link_idx {
                    origs.swap(i, origs_len_new);
                    origs_len_new += 1;
                }
            }

            // If no origins were found, add the appropriate previous idxs to
            // the processing vector.
            if origs_start == origs_len_new {
                if let Some(&idx_prev) = links[link_idx.idx()].idx_prev.real() {
                    link_idxs_proc.push(idx_prev);
                    if let Some(&idx_prev_alt) = links[link_idx.idx()].idx_prev_alt.real() {
                        link_idxs_proc.push(idx_prev_alt);
                    }
                }
            }
        }
    }

    if link_idx_set.contains(&LinkIdx::default()) {
        bail!("Link idx options is not allowed to contain fake link idx!");
    }

    // No paths found, so return an error.
    if origs_len_new == 0 {
        bail!(
            "No valid paths found from any origin to any destination!\norigs: {:?}\ndests: {:?}",
            origs,
            dests
        );
    }
    origs.truncate(origs_len_new);

    // Return the set of processed link indices.
    Ok((link_idx_set, origs))
}

/// Convert sequence of train states to sequence of estimated times
/// that will be added to the estimated time network.
fn update_est_times_add(
    est_times_add: &mut Vec<EstTime>,
    movement: &[SimpleState],
    link_pts: &[LinkPoint],
    length: si::Length,
) -> anyhow::Result<()> {
    est_times_add.clear();
    let state_first = movement.first().unwrap();

    // Initialize location indices
    let mut pt_idx_back = 0;
    while link_pts[pt_idx_back].offset <= *state_first.offset.get_fresh(|| format_dbg!())? - length
    {
        pt_idx_back += 1;
    }
    let mut pt_idx_front = pt_idx_back;
    while link_pts[pt_idx_front].offset <= *state_first.offset.get_fresh(|| format_dbg!())? {
        pt_idx_front += 1;
    }

    // Convert movement to estimated times at linkPoints
    let mut offset_next = link_pts[pt_idx_front]
        .offset
        .min(link_pts[pt_idx_back].offset + length);
    for i in 1..movement.len() {
        // Add estimated times while in range
        while offset_next <= *movement[i].offset.get_fresh(|| format_dbg!())? {
            let dist_diff_x2 =
                2.0 * (*movement[i].offset.get_fresh(|| format_dbg!())? - offset_next);
            let speed = (*movement[i].speed.get_fresh(|| format_dbg!())?
                * *movement[i].speed.get_fresh(|| format_dbg!())?
                - (*movement[i].speed.get_fresh(|| format_dbg!())?
                    - *movement[i - 1].speed.get_fresh(|| format_dbg!())?)
                    / (*movement[i].time.get_fresh(|| format_dbg!())?
                        - *movement[i - 1].time.get_fresh(|| format_dbg!())?)
                    * dist_diff_x2)
                .sqrt();
            let time_to_next = *movement[i].time.get_fresh(|| format_dbg!())?
                - dist_diff_x2 / (*movement[i].speed.get_fresh(|| format_dbg!())? + speed);

            // Add either an arrive or a clear event depending on which happened earlier
            let link_event =
                if link_pts[pt_idx_back].offset + length < link_pts[pt_idx_front].offset {
                    pt_idx_back += 1;
                    if pt_idx_back == 1 {
                        offset_next = link_pts[pt_idx_front]
                            .offset
                            .min(link_pts[pt_idx_back].offset + length);
                        continue;
                    }
                    LinkEvent {
                        link_idx: link_pts[pt_idx_back - 1].link_idx,
                        est_type: EstType::Clear,
                    }
                } else {
                    pt_idx_front += 1;
                    LinkEvent {
                        link_idx: link_pts[pt_idx_front - 1].link_idx,
                        est_type: EstType::Arrive,
                    }
                };

            est_times_add.push(EstTime {
                time_to_next,
                dist_to_next: offset_next,
                speed,
                link_event,
                ..Default::default()
            });
            offset_next = link_pts[pt_idx_front]
                .offset
                .min(link_pts[pt_idx_back].offset + length);
        }
    }
    Ok(())
}

/// Insert a new estimated time into the network.
/// Insertion may not occur if the estimated time would be a duplicate.
/// Returns true if insertion occurred.
fn insert_est_time(
    est_times: &mut Vec<EstTime>,
    est_alt: &mut EstTime,
    link_event_map: &mut LinkEventMap,
    est_insert: &EstTime,
) -> bool {
    let mut is_insert = false;
    loop {
        let idx_push = est_times.len().try_into().unwrap();
        let idx_next = est_times[est_alt.idx_prev.idx()].idx_next;

        // If the insert time can be inserted directly, insert it and return true
        if idx_next == EST_IDX_NA {
            let est_prev = &mut est_times[est_alt.idx_prev.idx()];
            est_prev.idx_next = idx_push;
            est_prev.time_to_next = est_insert.time_to_next - est_prev.time_to_next;
            est_prev.dist_to_next = est_insert.dist_to_next - est_prev.dist_to_next;

            link_event_map
                .entry(est_insert.link_event)
                .or_default()
                .insert(est_prev.idx_next);
            let idx_old = est_alt.idx_prev;
            est_alt.idx_prev = est_prev.idx_next;

            est_times.push(*est_insert);
            est_times.last_mut().unwrap().idx_prev = idx_old;
            is_insert = true;
            break;
        }

        // If the insert time is the same as the next estimated time, update stored values, do not insert, and return false
        let est_match = &est_times[idx_next.idx()];
        if est_match.link_event == est_insert.link_event
            && (est_insert.speed - est_match.speed).abs() < SPEED_DIFF_JOIN
        {
            est_alt.idx_prev = idx_next;
            break;
        }

        // If there is no alternate node, insert a fake one
        let est_prev = &mut est_times[est_alt.idx_prev.idx()];
        if est_prev.idx_next_alt == EST_IDX_NA {
            est_prev.idx_next_alt = idx_push;
            est_times.push(*est_alt);
            est_alt.idx_prev = idx_push;
        }
        // Otherwise, update info est_alt
        else {
            est_alt.idx_prev = est_prev.idx_next_alt;
        }
    }

    est_alt.time_to_next = est_insert.time_to_next;
    est_alt.dist_to_next = est_insert.dist_to_next;
    is_insert
}

/// Update join paths and perform the space match, saving the ones that were extended
fn update_join_paths_space(
    est_join_paths_prev: &[EstJoinPath],
    est_join_paths: &mut Vec<EstJoinPath>,
    est_idxs_temp: &mut Vec<EstIdx>,
    est_time_add: &EstTime,
    est_times: &[EstTime],
    is_event_seen: bool,
) {
    assert!(est_join_paths.is_empty());
    assert!(est_idxs_temp.is_empty());

    for est_join_path in est_join_paths_prev {
        let mut est_time_prev = &est_times[est_join_path.est_idx_next.idx()];
        // Do not save the join path if it stops
        if est_time_prev.idx_next == EST_IDX_NA {
            continue;
        }

        // For arrive events, do not change the space match status
        let link_idx_match = if est_time_add.link_event.est_type == EstType::Arrive {
            est_join_path.link_idx_match
        }
        // For clear events, continue processing if a space match happened or is happening
        else if est_join_path.has_space_match()
            || est_join_path.link_idx_match == est_time_add.link_event.link_idx
        {
            track::LINK_IDX_NA
        }
        // For clear events, save the join path and continue if a space match has not happened
        // Note that est_join_path.idx_next does not change because the number of clear events can differ prior to a space match
        else {
            est_join_paths.push(*est_join_path);
            continue;
        };

        // If the join path cannot possibly continue, skip it. TODO: Verify that this is correct
        if !is_event_seen {
            continue;
        }

        // If the join path has already matched in space, find all estimated times that extend the join path along the travel path and save them
        // Note, new space matches are not handled here since they may be advanced over multiple est idxs
        if est_join_path.has_space_match() {
            // Iterate over all alternate nodes
            loop {
                // If the event matches, push a new extended join path
                if est_time_add.link_event == est_times[est_time_prev.idx_next.idx()].link_event {
                    est_join_paths.push(EstJoinPath::new(link_idx_match, est_time_prev.idx_next));
                }
                // Break when there are no more alternate nodes to check
                if est_time_prev.idx_next_alt == EST_IDX_NA {
                    break;
                }
                est_time_prev = &est_times[est_time_prev.idx_next_alt.idx()];
            }
        }
        // Advance to all possible next arrive events and check for space match using clear event
        else {
            loop {
                // Loop until reaching an event match, an arrive event, or the end
                loop {
                    // Add alternate node to the processing stack (if applicable)
                    if est_time_prev.idx_next_alt != EST_IDX_NA {
                        est_idxs_temp.push(est_time_prev.idx_next_alt)
                    }
                    debug_assert!(est_time_prev.idx_next != EST_IDX_NA);
                    let est_time_next = &est_times[est_time_prev.idx_next.idx()];

                    // If the event matches, push the new join path and stop advancing
                    if est_time_add.link_event == est_time_next.link_event {
                        debug_assert!(
                            est_time_add.link_event.est_type == EstType::Arrive
                                || link_idx_match.is_fake()
                        );
                        est_join_paths
                            .push(EstJoinPath::new(link_idx_match, est_time_prev.idx_next));
                        break;
                    }
                    // Break when reaching an arrive event
                    if est_time_next.link_event.est_type == EstType::Arrive
                        || est_time_next.idx_next == EST_IDX_NA
                    {
                        break;
                    }
                    est_time_prev = est_time_next;
                }
                match est_idxs_temp.pop() {
                    None => break,
                    Some(est_idx) => est_time_prev = &est_times[est_idx.idx()],
                };
            }
        }
    }
}

/// Check speed difference for space matched join paths and perform join for the best speed match (if sufficiently good)
/// Returns true if a join occurred.
fn perform_speed_join(
    est_join_paths: &[EstJoinPath],
    est_times: &mut Vec<EstTime>,
    est_time_add: &EstTime,
) -> bool {
    let mut speed_diff_join = SPEED_DIFF_JOIN;
    let mut est_idx_join = EST_IDX_NA;
    for est_join_path in est_join_paths {
        if est_join_path.has_space_match() {
            let speed_diff =
                (est_times[est_join_path.est_idx_next.idx()].speed - est_time_add.speed).abs();
            if speed_diff < speed_diff_join {
                speed_diff_join = speed_diff;
                est_idx_join = est_join_path.est_idx_next;
            }
        }
    }

    if speed_diff_join < SPEED_DIFF_JOIN {
        // TODO: Add assertion from C++

        let est_idx_last = (est_times.len() - 1).try_into().unwrap();
        // Join to specified estimated time
        loop {
            // If the target join node has a free previous index, join and return as successful
            let est_time_join = &mut est_times[est_idx_join.idx()];
            if est_time_join.idx_prev == EST_IDX_NA {
                est_time_join.idx_prev = est_idx_last;

                let est_time_prev = &mut est_times[est_idx_last.idx()];
                est_time_prev.idx_next = est_idx_join;
                est_time_prev.time_to_next = est_time_add.time_to_next - est_time_prev.time_to_next;
                est_time_prev.dist_to_next = est_time_add.dist_to_next - est_time_prev.dist_to_next;
                return true;
            }

            // If the target join node has a free previous alt index, attach a fake node
            if est_time_join.idx_prev_alt == EST_IDX_NA {
                let est_idx_attach = est_times.len().try_into().unwrap();
                est_times[est_idx_join.idx()].idx_prev_alt = est_idx_attach;
                est_times.push(EstTime {
                    idx_next: est_idx_join,
                    ..Default::default()
                });
                est_idx_join = est_idx_attach;
            }
            // Otherwise, traverse the previous alt index
            else {
                est_idx_join = est_time_join.idx_prev_alt;
            }
        }
    }
    false
}

/// For arrive events with an entry in the link event map, add new join paths
fn add_new_join_paths(
    link_event_add: &LinkEvent,
    link_event_map: &LinkEventMap,
    est_join_paths_save: &mut Vec<EstJoinPath>,
) {
    // Only add join paths for arrive events
    if link_event_add.est_type != EstType::Arrive {
        return;
    }
    if let Some(est_idxs_next) = link_event_map.get(link_event_add) {
        let mut est_idxs_new;
        // If there are no join paths, make a new join path for each est idx in the link event map entry
        let est_idxs_push = if est_join_paths_save.is_empty() {
            est_idxs_next
        }
        // If there are remaining join paths, use it to eliminate est idxs from the cloned link event map entry
        else {
            est_idxs_new = est_idxs_next.clone();
            for est_join_path in &*est_join_paths_save {
                est_idxs_new.remove(&est_join_path.est_idx_next);
            }
            &est_idxs_new
        };

        // Push a new join path for each value in est_idxs_push
        for est_idx in est_idxs_push {
            est_join_paths_save.push(EstJoinPath::new(link_event_add.link_idx, *est_idx));
        }
    }
}

/// `make_est_times` function creates an estimated-time network (`EstTimeNet`) and train consist (`Consist`) from a
/// given `SpeedLimitTrainSim` and rail `network`. This function performs the following
/// major steps in a loop, until all trains (in `saved_sims`) are processed:
///
/// 1. **Initialize** a train simulation for each origin node and add it to the `saved_sims` stack.
/// 1. **Pop** a train from `saved_sims`, then:
///     1. Run the simulation (`update_movement`) until a condition in `saved_sim.update_movement()`
///        in `saved_sims.update_movement()` is met.
///     1. Convert the results into `EstTime` nodes.
///         - If the path diverges from the existing network, insert new `EstTime` events and
///           set `has_split = true`.
///         - If `has_split` is set, attempt to **join** back to an existing sequence in the
///           `EstTime` network (the “space and speed match” check). If successful, **break**
///           out of this train's processing loop.
///     1. If the simulation reaches the final destination, add the last node to `est_idxs_end` and **break**.
///     1. Otherwise, add the next link(s) that continues the path toward the destination. If
///        multiple branches exist, clone the train sim for the alternate path and push it to
///        `saved_sims`. (Processing continues until we reach a break.)
///
/// 1. **Post-process** the resulting `EstTimeNet` by fixing references, linking final nodes,
///    and updating times forward and backward.
/// 1. **Return** the completed `EstTimeNet` and the `Consist`.
///
/// # Arguments
///
/// * `speed_limit_train_sim` - `SpeedLimitTrainSim` is an instance of
///    train simulation in which speed is allowed to vary according to train
///    capabilities and speed limit.
/// * `network` - Network comprises an ensemble of links (path between junctions
///    along the rail with heading, grade, and location) this simulation
///    operates on.
/// * `path_for_failed_sim` - if provided, saves failed `speed_limit_train_sim` at Path
///
/// # Returns
///
/// A tuple of:
/// * `EstTimeNet` - The fully built estimated-time network,
/// * `Consist` - The locomotive consist associated with the simulation.
///
/// # Errors
///
/// Returns an error if:
/// * The provided origins or next link options are invalid.
/// * The path is unexpectedly truncated.
/// * The simulation fails internally while updating movements or extending paths.
pub fn make_est_times<N: AsRef<[Link]>>(
    mut speed_limit_train_sim: SpeedLimitTrainSim,
    network: N,
    path_for_failed_sim: Option<PathBuf>,
) -> anyhow::Result<(EstTimeNet, Consist)> {
    speed_limit_train_sim.set_save_interval(None);
    let network = network.as_ref();
    let dests = &speed_limit_train_sim.dests;
    // Step 1a: Gather valid next-link indices for each origin/destination.
    let (link_idx_options, origs) =
        get_link_idx_options(&speed_limit_train_sim.origs, dests, network)
            .with_context(|| format_dbg!())?;
    // We'll store our estimated times and a map of link events here.
    let mut est_times = Vec::with_capacity(network.len() * 10);
    let mut consist_out = None;
    let mut saved_sims: Vec<SavedSim> = vec![];
    let mut link_event_map =
        LinkEventMap::with_capacity_and_hasher(est_times.capacity(), Default::default());
    // The departure time for all initial events
    let time_depart = *speed_limit_train_sim
        .state
        .time
        .get_fresh(|| format_dbg!())?;

    // -----------------------------------------------------------------------
    // Push initial "fake" nodes.
    // These help define the start of our `EstTime` sequence.
    // -----------------------------------------------------------------------
    est_times.push(EstTime {
        idx_next: 1,
        ..Default::default()
    });
    est_times.push(EstTime {
        time_to_next: time_depart,
        idx_prev: 0,
        ..Default::default()
    });

    // -----------------------------------------------------------------------
    // Add origin estimated times:
    // For each origin, we ensure offset=0, is_front_end=false, and a real link_idx.
    // Then create two `EstTime` events: (Arrive + Clear) for the train's tail and head.
    // -----------------------------------------------------------------------
    for orig in origs {
        ensure!(
            orig.offset == si::Length::ZERO,
            "Origin offset must be zero!"
        );
        ensure!(
            !orig.is_front_end,
            "Origin must be relative to the tail end!"
        );
        ensure!(orig.link_idx.is_real(), "Origin link idx must be real!");

        let mut est_alt = EstTime {
            time_to_next: time_depart,
            dist_to_next: orig.offset,
            idx_prev: 1,
            ..Default::default()
        };

        // Arrive event
        insert_est_time(
            &mut est_times,
            &mut est_alt,
            &mut link_event_map,
            &EstTime {
                time_to_next: time_depart,
                dist_to_next: orig.offset,
                speed: si::Velocity::ZERO,
                link_event: LinkEvent {
                    link_idx: orig.link_idx,
                    est_type: EstType::Arrive,
                },
                ..Default::default()
            },
        );

        // Clear event
        insert_est_time(
            &mut est_times,
            &mut est_alt,
            &mut link_event_map,
            &EstTime {
                time_to_next: time_depart,
                dist_to_next: orig.offset
                    + *speed_limit_train_sim
                        .state
                        .length
                        .get_fresh(|| format_dbg!())?,
                speed: si::Velocity::ZERO,
                link_event: LinkEvent {
                    link_idx: orig.link_idx,
                    est_type: EstType::Clear,
                },
                ..Default::default()
            },
        );

        // Save this train simulator state to be processed.
        // NOTE, there may be a way to just clone the state(s) and not the whole thing
        saved_sims.push(SavedSim {
            train_sim: {
                let mut train_sim = Box::new(speed_limit_train_sim.clone());
                train_sim
                    .extend_path_tpc(network, &[orig.link_idx])
                    .with_context(|| format_dbg!())?;
                train_sim
            },
            join_paths: vec![],
            est_alt,
        });
    }

    // -----------------------------------------------------------------------
    // Reset distance to zero for any alternate "fake" nodes.
    // This helps unify distance offsets for subsequent processing.
    // -----------------------------------------------------------------------
    {
        let mut est_idx_fix = 1;
        while est_idx_fix != EST_IDX_NA {
            est_times[est_idx_fix.idx()].dist_to_next = si::Length::ZERO;
            est_idx_fix = est_times[est_idx_fix.idx()].idx_next_alt;
        }
    }

    let mut movement = Vec::<SimpleState>::with_capacity(32);
    let mut est_times_add = Vec::<EstTime>::with_capacity(32);
    let mut est_idxs_store = Vec::<EstIdx>::with_capacity(32);
    let mut est_join_paths_save = Vec::<EstJoinPath>::with_capacity(16);
    let mut est_idxs_end = Vec::<EstIdx>::with_capacity(8);

    // -----------------------------------------------------------------------
    // Main loop: process each saved simulation until `saved_sims` is empty.
    // -----------------------------------------------------------------------
    while let Some(mut sim) = saved_sims.pop() {
        let mut has_split = false;
        ensure!(
            sim.train_sim.link_idx_last().unwrap().is_real(),
            "Last link idx must be real! Link points={:?}",
            sim.train_sim.link_points()
        );

        // -----------------------
        // 'path loop: keep updating this train's movement until:
        //   - we find a join
        //   - the train finishes
        //   - or the path unexpectedly ends.
        // -----------------------

        'path: loop {
            // Step 2a: simulate the train for one cycle of movement.
            sim.update_movement(&mut movement)
                .with_context(|| format_dbg!())?;
            // Convert the new movement states into `EstTime` additions.
            update_est_times_add(
                &mut est_times_add,
                &movement,
                sim.train_sim.link_points(),
                *sim.train_sim.state.length.get_fresh(|| format_dbg!())?,
            )?;

            // Step 2b: for each new EstTime, either insert it or try to join an existing path.
            for est_time_add in &est_times_add {
                // Only check for joins if we've already split from the main path.
                if has_split {
                    // Attempt to join existing paths if there's a "space match".
                    // This checks alignment of Arrive events and ensures speeds match.
                    update_join_paths_space(
                        &sim.join_paths,
                        &mut est_join_paths_save,
                        &mut est_idxs_store,
                        est_time_add,
                        &est_times,
                        link_event_map.contains_key(&est_time_add.link_event),
                    );

                    // If the join succeeds, break to the outer loop because this sim has finished being processed
                    if perform_speed_join(&est_join_paths_save, &mut est_times, est_time_add) {
                        est_join_paths_save.clear();
                        sim.join_paths.clear();
                        break 'path;
                    }

                    // If not joined, record new potential join paths.
                    add_new_join_paths(
                        &est_time_add.link_event,
                        &link_event_map,
                        &mut est_join_paths_save,
                    );

                    std::mem::swap(&mut sim.join_paths, &mut est_join_paths_save);
                    est_join_paths_save.clear();
                }

                // Insert event into the network. If it's a new branch, set `has_split = true`.
                if insert_est_time(
                    &mut est_times,
                    &mut sim.est_alt,
                    &mut link_event_map,
                    est_time_add,
                ) {
                    has_split = true;
                }
            }

            // If the train_sim finished, add destination node to final processing (all links should be clear)
            if sim.train_sim.is_finished() {
                est_idxs_end.push((est_times.len() - 1).try_into().unwrap());
                if consist_out.is_none() {
                    consist_out = Some(sim.train_sim.loco_con);
                }
                break;
            }
            // Otherwise, append the next link options and continue simulating
            else {
                let link_idx_prev = &sim.train_sim.link_idx_last().unwrap();
                let link_idx_next = network[link_idx_prev.idx()].idx_next;
                let link_idx_next_alt = network[link_idx_prev.idx()].idx_next_alt;
                ensure!(
                    link_idx_next.is_real(),
                    "Link idx next cannot be fake when making est times! link_idx_prev={link_idx_prev:?}"
                );

                // Collect the valid links that will allow reaching destination
                let link_idxs_next_valid = [link_idx_next, link_idx_next_alt]
                    .into_iter()
                    .filter(|link_idx| link_idx_options.contains(link_idx))
                    .collect::<Vec<_>>();
                let link_idx_next = match link_idxs_next_valid[..] {
                    // extract out the next link that this train will travel on
                    [link_idx_next] => link_idx_next,
                    // extract out the next link thit this train will travel on
                    // and create a new train sim that will travel the alternate branch
                    [link_idx_next, link_idx_next_alt] => {
                        let mut new_sim = sim.clone();
                        if let Err(err) = new_sim
                            .train_sim
                            .extend_path_tpc(network, &[link_idx_next_alt])
                            .with_context(|| format_dbg!())
                        {
                            if let Some(save_path) = path_for_failed_sim {
                                new_sim.to_file(save_path)?;
                            }

                            bail!(err)
                        }
                        new_sim.check_dests(dests);
                        saved_sims.push(new_sim);
                        link_idx_next
                    }
                    _ => {
                        bail!(
                                            "{}
                Unexpected end of path reached! prev={link_idx_prev:?}, next={link_idx_next:?}, next_alt={link_idx_next_alt:?}",
                                            format_dbg!()
                                        );
                    }
                };
                // Extend the path for the current sim with the chosen next link.
                sim.train_sim
                    .extend_path_tpc(network, &[link_idx_next])
                    .with_context(|| format_dbg!())?;
                sim.check_dests(dests);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Link up the final "end" nodes into the estimated time network.
    // Here, we add one node per end index and link them together as a chain,
    // then mark the final arrival distances/times as zero.
    // -----------------------------------------------------------------------
    ensure!(est_times.len() < (EstIdx::MAX as usize) - est_idxs_end.len());

    let mut est_idx_alt = EST_IDX_NA;
    for est_idx_end in est_idxs_end.iter().rev() {
        est_times.push(EstTime {
            idx_next: est_times.len() as EstIdx + 1,
            idx_prev: *est_idx_end,
            idx_prev_alt: est_idx_alt,
            ..Default::default()
        });
        est_idx_alt = est_times.len() as EstIdx - 1;
        est_times[est_idx_end.idx()].idx_next = est_idx_alt;
        est_times[est_idx_end.idx()].time_to_next = si::Time::ZERO;
        est_times[est_idx_end.idx()].dist_to_next = si::Length::ZERO;
    }

    // Add the final (fake) node and shrink `est_times` to free unused capacity.
    est_times.push(EstTime {
        idx_prev: est_times.len() as EstIdx - 1,
        ..Default::default()
    });
    est_times.shrink_to_fit();

    // -----------------------------------------------------------------------
    // Validate that all linked indices are correct and consistent.
    // -----------------------------------------------------------------------
    for (idx, est_time) in est_times.iter().enumerate() {
        // Verify that all prev idxs are valid
        assert!((est_time.idx_prev != EST_IDX_NA) != (idx <= 1));
        // Verify that the next idxs are valid
        assert!((est_time.idx_next != EST_IDX_NA) != (idx == est_times.len() - 1));
        // Verify that no fake nodes have both idx prev alt and idx next alt
        assert!(
            est_time.link_event.est_type != EstType::Fake
                || est_time.idx_prev_alt == EST_IDX_NA
                || est_time.idx_next_alt == EST_IDX_NA
        );

        let est_time_prev = est_times[est_time.idx_prev.idx()];
        let est_time_next = est_times[est_time.idx_next.idx()];
        let est_idx = idx as EstIdx;
        // Verify that prev est time is linked to current est time
        ensure!(est_time_prev.idx_next == est_idx || est_time_prev.idx_next_alt == est_idx);
        // Verify that next est time is linked to current est time
        ensure!(
            est_time_next.idx_prev == est_idx
                || est_time_next.idx_prev_alt == est_idx
                || idx == est_times.len() - 1
        );

        // Verify that current est time is not the alternate of both the previous and next est times
        ensure!(
            est_time_prev.idx_next_alt != est_idx
                || est_time_next.idx_prev_alt != est_idx
                || idx == 0
                || idx == est_times.len() - 1
        );
    }

    // -----------------------------------------------------------------------
    // Update times forward and backward to generate final scheduling info.
    // -----------------------------------------------------------------------
    update_times_forward(&mut est_times, time_depart);
    update_times_backward(&mut est_times);

    // Construct the final EstTimeNet.
    let est_time_net = EstTimeNet::new(est_times);

    // Sanity check: ensure not all times are zero.
    ensure!(
        !est_time_net.val.iter().all(|x| x.time_sched == 0. * uc::S),
        "All times are 0.0 so something went wrong.\n{}",
        format_dbg!()
    );
    // Return the finished network and the locomotive consist.
    Ok((est_time_net, consist_out.unwrap()))
}

#[cfg(feature = "pyo3")]
#[pyfunction(name = "make_est_times")]
#[pyo3(signature=(speed_limit_train_sim, network, path_for_failed_sim=None))]
pub fn make_est_times_py(
    speed_limit_train_sim: SpeedLimitTrainSim,
    network: &Bound<PyAny>,
    path_for_failed_sim: Option<&Bound<PyAny>>,
) -> anyhow::Result<(EstTimeNet, Consist)> {
    let network = match network.extract::<Network>() {
        Ok(n) => n,
        Err(_) => {
            let n = network
                .extract::<Vec<Link>>()
                .map_err(|_| anyhow!("{}", format_dbg!()))?;
            Network(Default::default(), n)
        }
    };

    let path_for_failed_sim = match path_for_failed_sim {
        Some(pffs) => Some(PathBuf::extract_bound(pffs)?),
        None => None,
    };

    make_est_times(speed_limit_train_sim, network, path_for_failed_sim)
}
