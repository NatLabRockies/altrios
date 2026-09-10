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
use altrios_core::traits::*;
use altrios_core::train::{speed_limit_train_sim_fwd, speed_limit_train_sim_rev};
use altrios_core::uc;
use std::time::Instant;

fn taconite() -> Network {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../python/altrios/resources/networks/Taconite.yaml");
    Network::from_file(path, false).unwrap()
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

fn run_case(network: &Network, n: usize, spacing_min: f64) {
    let (sims, ests) = build(network, n, spacing_min);
    let reps: usize = std::env::var("BENCH_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let mut best = f64::INFINITY;
    let mut chk = String::new();
    for _ in 0..reps {
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
}

#[test]
#[ignore]
fn bench_dispatch_scaling() {
    let network = taconite();
    let spacing: f64 = std::env::var("BENCH_SPACING_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let ns: Vec<usize> = std::env::var("BENCH_NS")
        .ok()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16, 32, 64]);
    for n in ns {
        run_case(&network, n, spacing);
    }
}
