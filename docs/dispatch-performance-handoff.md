# Handoff: speed up meet-pass dispatch (`altrios-core/src/meet_pass/`)

Status: investigation done, no code changed yet. This document is the spec for the
implementation. Read `CLAUDE.md` first (build commands, conventions, docstring rules).

## Goal

Reduce runtime of `run_dispatch` (`altrios-core/src/meet_pass/dispatch.rs`) for large
multi-train simulations **without changing its output**. Two changes are verified to be
output-identical and are the deliverable. A third, larger change is optional and is a
behavioral change that must be validated separately.

## Where the time goes (measured)

Benchmark: Taconite network, N trains alternating fwd/rev fixtures, departures 30 min
apart, release build, best of 3 runs.

| N trains | baseline | validation gated off | + affected-train precheck |
|---|---|---|---|
| 32  | 0.176 s | 0.067 s | 0.063 s |
| 64  | 1.67 s  | 0.63 s  | 0.62 s  |
| 128 | 17.2 s  | 7.1 s   | 5.4 s   |
| 256 | 121 s   | —       | —       |

Output checksums (sum of link idxs, sum of times over all timed paths) were identical
across all three columns:

- N=32:  `links=1378615 time_sum=196454511.1`
- N=64:  `links=2756167 time_sum=805639571.0`
- N=128: `links=5511271 time_sum=3267674528.9`

Counters at N=128: ~164k `advance` steps, ~180k `check_deadlock` calls, 12.4M
`update_free_path` calls; 94% of those calls are on trains that have not departed yet
(`disp_node_idx_free == None`); 46% enter the path-search loop. Every advance step
re-plans every unfinished train against the moved train — that is the O(steps × trains)
core and is *not* changed by the two deliverables below.

Negative result: parallelizing the per-train loop in `check_deadlock` with rayon was
2–4x **slower** (per-call work < 1 µs, 180k invocations). Do not do this.

## Deliverable 1: gate `validate_free_path` to debug builds

File: `altrios-core/src/meet_pass/train_disp/free_path.rs`, end of `update_free_path`:

```rust
        self.validate_free_path(links_blocked)?;
        Ok(FreePathStatus::UpdateSuccess)
```

`validate_free_path` is a pure `&self` consistency check (walks the whole `disp_path`
with `windows(2)` plus an occupancy scan) that only ever produces errors; it runs on every
call. Gate it so it still runs in debug/test builds:

```rust
        #[cfg(debug_assertions)]
        self.validate_free_path(links_blocked)?;
        Ok(FreePathStatus::UpdateSuccess)
```

Then silence the resulting dead-code warning in release (`#[cfg(debug_assertions)]` on the
fn, or `#[cfg_attr(not(debug_assertions), allow(dead_code))]`). A cargo feature
(`validate-dispatch`) instead of `debug_assertions` is also acceptable if you want it
switchable in release builds; pick one and document it in the module doc of
`meet_pass/mod.rs`.

## Deliverable 2: skip trains that cannot be affected

`update_free_path` is a no-op when (a) none of the moved train's blocking links are on this
train's path **and** (b) no diverge node references the moved train. (With
`LinkOptType::None` the walk jumps straight to the ending sentinel and returns
`UpdateSuccess`; the only side effect is overwriting the sentinel's `train_idx`, which is
rewritten on every call anyway.)

Add to `impl TrainDisp` in `free_path.rs` (Google-style doc comment per CLAUDE.md):

```rust
    /// True if `update_free_path` could do anything for this moved train: either one of
    /// the moved train's blocking links lies on this train's path, or this train has
    /// previously diverged because of the moved train. Otherwise the update is a no-op.
    pub fn may_be_affected(&self, train_idx_moved: TrainIdx, link_idxs_blocking: &[LinkIdx]) -> bool {
        link_idxs_blocking.iter().any(|l| self.links_on_path.contains(l))
            || self.div_nodes[..self.div_nodes.len() - 1]
                .iter()
                .any(|d| d.train_idx == train_idx_moved)
    }
```

In `check_deadlock` (`dispatch.rs`), change

```rust
            if idx != train_idx_moved.idx() {
                match train_disp.update_free_path(
```
to
```rust
            if idx != train_idx_moved.idx()
                && train_disp.may_be_affected(train_idx_moved, &link_idxs_blocked)
            {
                match train_disp.update_free_path(
```

Optional extension (bigger payoff at thousands of trains): a global inverted index
`Vec<Vec<TrainIdx>>` from link idx → trains whose `links_on_path` contains it, maintained
where `links_on_path` is mutated (`update_free_path_helpers`), so `check_deadlock` visits
only candidate trains instead of probing every train. Only do this if the simple precheck
proves insufficient on the real workload.

## Optional / minor

- `train_idx_begin` in `check_deadlock` only skips a *contiguous* finished prefix.
  Replacing it with an explicit active-train list (swap-remove on finish) is equivalent
  and small; low priority.

## NOT in scope (behavioral change — flag, don't do silently)

Deferring re-planning for not-yet-departed trains (94% of calls) until they are popped
from the queue would be the order-of-magnitude win, but it changes deadlock semantics:
today an undeparted train reporting `Blocked` counts as deadlock and forces the moving
train to rewind. It would also need a "catch-up" full re-plan on first pop. Treat as a
separate, validated project.

## Verification

1. `cargo test --workspace` (Rust) and `pytest python/altrios/tests -v` must pass.
   `test_simple_dispatch` in `dispatch.rs` exercises a 2-train Taconite case.
2. Rebuild the extension: `maturin develop --release`, then run
   `pytest python/altrios/demos -v` (slow) or at least `speed_limit_train_sim_demo.py`.
3. Re-create the scaling benchmark as an integration test (below), run it **before** the
   change (baseline) and **after**, and confirm the output checksums above (or your own
   baseline checksums) are bit-identical. Delete the benchmark file afterwards or commit
   it as an `#[ignore]`d test — your call.

### Benchmark harness (put at `altrios-core/tests/dispatch_bench_tmp.rs`)

Run with:
```
cd altrios-core
BENCH_NS=32,64,128 cargo test --release -p altrios-core --test dispatch_bench_tmp bench_dispatch_scaling -- --ignored --nocapture
```

```rust
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
fn build(network: &Network, n: usize, spacing_min: f64) -> (Vec<SpeedLimitTrainSim>, Vec<EstTimeNet>) {
    let base_fwd = speed_limit_train_sim_fwd();
    let base_rev = speed_limit_train_sim_rev();
    let est_fwd = make_est_times(base_fwd.clone(), network, None).unwrap().0;
    let est_rev = make_est_times(base_rev.clone(), network, None).unwrap().0;
    let (mut sims, mut ests) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let mut s = if i % 2 == 0 { base_fwd.clone() } else { base_rev.clone() };
        s.state.time.update_unchecked((i as f64) * spacing_min * uc::MIN, || String::new()).unwrap();
        sims.push(s);
        ests.push(if i % 2 == 0 { est_fwd.clone() } else { est_rev.clone() });
    }
    (sims, ests)
}

fn run_case(network: &Network, n: usize, spacing_min: f64) {
    let (sims, ests) = build(network, n, spacing_min);
    let reps: usize = std::env::var("BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut best = f64::INFINITY;
    let mut chk = String::new();
    for _ in 0..reps {
        let t0 = Instant::now();
        let out = run_dispatch(network, &sims, ests.clone(), false, false).unwrap();
        best = best.min(t0.elapsed().as_secs_f64());
        let (mut sl, mut st) = (0u64, 0f64);
        for p in &out { for x in p { sl += x.link_idx.idx() as u64; st += x.time.value; } }
        chk = format!("links={sl} time_sum={st:.1}");
    }
    println!("BENCH n={n:4} spacing_min={spacing_min:6.1} min_s={best:9.3} chk[{chk}]");
}

#[test]
#[ignore]
fn bench_dispatch_scaling() {
    let network = taconite();
    let spacing: f64 = std::env::var("BENCH_SPACING_MIN").ok().and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let ns: Vec<usize> = std::env::var("BENCH_NS").ok()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![2, 4, 8, 16, 32, 64]);
    for n in ns { run_case(&network, n, spacing); }
}
```

Timing pitfalls seen during investigation: right after editing Rust sources,
rust-analyzer re-indexes and can inflate timings 3x for a minute or two. Use best-of-3
and re-run the baseline back-to-back with the patched build.

## Context worth knowing

- This fork's `python/altrios/sim_manager.py` (line ~123, commit 2b52a8c) dispatches each
  train **alone** in a loop; upstream passes all trains to one `run_dispatch` call. Only
  the all-trains form exercises meet-pass and hits this bottleneck. Don't "fix" that loop
  as part of this task, but be aware the Python demos won't show a speedup.
- `*.rs.bak` files are stale merge artifacts; ignore them.
- Rebuild the extension with `maturin develop --release` after any Rust change.
