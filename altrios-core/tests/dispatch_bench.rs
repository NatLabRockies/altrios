//! Scaling benchmark and output-stability check for `run_dispatch`.
//!
//! Ignored by default because it takes tens of seconds. Run it with:
//!
//! ```text
//! BENCH_NS=32,64,128 cargo test --release --test dispatch_bench -- --ignored --nocapture
//! ```
//!
//! `BENCH_NS` (train counts), `BENCH_REPS` (timed repetitions, best is reported) and
//! `BENCH_SPACING_MIN` (departure spacing) are all overridable.
//!
//! The printed `chk[...]` checksum is the real point: dispatch optimizations are supposed
//! to be output-identical, so any change to these values for a given `BENCH_NS` is a
//! regression. Reference values on the Taconite network at 30 min spacing:
//!
//! ```text
//! n= 32  links=1378615  time_sum=196454511.1
//! n= 64  links=2756167  time_sum=805639571.0
//! n=128  links=5511271  time_sum=3267674528.9
//! ```
//!
//! Timing note: `--release` matters enormously here, and a debug build additionally runs
//! `validate_free_path` (see the `meet_pass` module docs), which is worth doing
//! occasionally at a small `BENCH_NS` to confirm dispatch state stays consistent.

use altrios_core::meet_pass::dispatch::run_dispatch;
use altrios_core::prelude::*;
use altrios_core::train::{speed_limit_train_sim_fwd, speed_limit_train_sim_rev};
use altrios_core::traits::*;
use altrios_core::uc;
use std::time::Instant;

/// Locations for the selected corridor, as (id, link idx, flip link idx). Sourced from
/// each network's `*_locations.csv`.
const TACONITE_LOCS: [(&str, u32, u32); 4] = [
    ("Minneapolis", 71, 606),
    ("Allouez", 4, 539),
    ("Superior", 22, 557),
    ("Hibbing", 148, 683),
];
const WINNIPEG_CHICAGO_LOCS: [(&str, u32, u32); 2] =
    [("Winnipeg", 1, 49280), ("Chicago", 24640, 24641)];
const SPOKANE_LETHBRIDGE_LOCS: [(&str, u32, u32); 2] =
    [("Spokane", 1, 4), ("Lethbridge", 15765, 20748)];

/// Selected by `BENCH_NETWORK` (`taconite`, the default, or `winnipeg`). Corridors differ
/// enormously in length, siding layout and single/double-track mix, so results are only
/// ever a statement about the corridor they were measured on.
fn corridor() -> (Network, &'static [(&'static str, u32, u32)]) {
    let which = std::env::var("BENCH_NETWORK").unwrap_or_else(|_| "taconite".into());
    let (file, locs): (&str, &'static [(&'static str, u32, u32)]) = match which.as_str() {
        "winnipeg" => ("Winnipeg_Chicago.msgpack", &WINNIPEG_CHICAGO_LOCS),
        // The with-sidings variant is what applications/catenary_optimization uses.
        "spokane_plain" => (
            "Spokane_Lethbridge/spokane_lethbridge.msgpack",
            &SPOKANE_LETHBRIDGE_LOCS,
        ),
        "spokane" => (
            "spokane_lethbridge_with_sidings_speed_reduced_059.msgpack",
            &SPOKANE_LETHBRIDGE_LOCS,
        ),
        _ => ("Taconite.msgpack", &TACONITE_LOCS),
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../python/altrios/resources/networks")
        .join(file);
    println!("BENCH_INFO network={file}");
    (Network::from_file(path, false).unwrap(), locs)
}

/// n trains alternating fwd/rev, departures spaced `spacing_min` apart.
/// Dispatch never reads `time_sched`, so one est-time net per direction is reused.
fn build(
    network: &Network,
    n: usize,
    spacing_min: f64,
) -> (Vec<SpeedLimitTrainSim>, Vec<EstTimeNet>) {
    let base_fwd = speed_limit_train_sim_fwd();
    let base_rev = speed_limit_train_sim_rev();
    let est_fwd = make_est_times(base_fwd.clone(), network, None).unwrap().0;
    let est_rev = make_est_times(base_rev.clone(), network, None).unwrap().0;
    let (mut sims, mut ests) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let mut s = if i % 2 == 0 {
            base_fwd.clone()
        } else {
            base_rev.clone()
        };
        s.state
            .time
            .update_unchecked((i as f64) * spacing_min * uc::MIN, || String::new())
            .unwrap();
        sims.push(s);
        ests.push(if i % 2 == 0 {
            est_fwd.clone()
        } else {
            est_rev.clone()
        });
    }
    (sims, ests)
}

fn loc_pair(id: &str, link_fwd: u32, link_rev: u32) -> Vec<Location> {
    [link_fwd, link_rev]
        .iter()
        .map(|link| Location {
            location_id: id.into(),
            offset: 0.0 * uc::M,
            link_idx: LinkIdx::new(*link),
            is_front_end: Default::default(),
            grid_emissions_region: "MISO_North".into(),
            electricity_price_region: "MN".into(),
            liquid_fuel_price_region: "MN".into(),
        })
        .collect()
}

/// Heterogeneous variant: trains spread over every feasible ordered pair of Taconite
/// locations, so origins and route lengths differ instead of every train being a clone of
/// one fixture. Corridors vary widely in siding layout and single/double-track mix, so
/// treat both this and the homogeneous case as two points on one network, not as
/// generalizable numbers.
fn build_mixed(
    network: &Network,
    locs: &[(&str, u32, u32)],
    n: usize,
    spacing_min: f64,
) -> (Vec<SpeedLimitTrainSim>, Vec<EstTimeNet>) {
    let base = speed_limit_train_sim_fwd();
    let mut routes: Vec<(SpeedLimitTrainSim, EstTimeNet)> = Vec::new();
    for (i, (id_o, of, orr)) in locs.iter().enumerate() {
        for (j, (id_d, df, dr)) in locs.iter().enumerate() {
            if i == j {
                continue;
            }
            let mut sim = base.clone();
            sim.origs = loc_pair(id_o, *of, *orr);
            sim.dests = loc_pair(id_d, *df, *dr);
            match make_est_times(sim.clone(), network, None) {
                Ok((est, _)) => routes.push((sim, est)),
                Err(e) => println!("BENCH_INFO route {id_o}->{id_d} infeasible: {e}"),
            }
        }
    }
    assert!(
        !routes.is_empty(),
        "no feasible O-D pairs for this corridor"
    );
    println!("BENCH_INFO feasible_routes={}", routes.len());
    let (mut sims, mut ests) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let (sim, est) = &routes[i % routes.len()];
        let mut s = sim.clone();
        s.state
            .time
            .update_unchecked((i as f64) * spacing_min * uc::MIN, || String::new())
            .unwrap();
        sims.push(s);
        ests.push(est.clone());
    }
    (sims, ests)
}

fn run_case(network: &Network, locs: &[(&str, u32, u32)], n: usize, spacing_min: f64) {
    let mixed = std::env::var("BENCH_ROUTES")
        .map(|v| v == "mixed")
        .unwrap_or(false);
    let (sims, ests) = if mixed {
        build_mixed(network, locs, n, spacing_min)
    } else {
        build(network, n, spacing_min)
    };
    let reps: usize = std::env::var("BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let mut best = f64::INFINITY;
    let mut chk = String::new();
    for _ in 0..reps {
        altrios_core::meet_pass::counters::reset();
        let t0 = Instant::now();
        let out = run_dispatch(network, &sims, ests.clone(), false, false).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        let (mut sl, mut st) = (0u64, 0f64);
        for p in &out {
            for x in p {
                sl += x.link_idx.idx() as u64;
                st += x.time.value;
            }
        }
        chk = format!("links={sl} time_sum={st:.1}");
    }
    println!("BENCH n={n:4} spacing_min={spacing_min:6.1} min_s={best:9.3} chk[{chk}]");
    for (name, value) in altrios_core::meet_pass::counters::report() {
        println!("COUNT n={n:4} spacing_min={spacing_min:6.1} {name:>20} {value:>14}");
    }
}

#[test]
#[ignore]
fn bench_dispatch_scaling() {
    let (network, locs) = corridor();
    let spacing: f64 = std::env::var("BENCH_SPACING_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let ns: Vec<usize> = std::env::var("BENCH_NS")
        .ok()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16, 32, 64]);
    for n in ns {
        run_case(&network, locs, n, spacing);
    }
}
