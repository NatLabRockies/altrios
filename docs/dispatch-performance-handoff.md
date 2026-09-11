# Handoff: speed up meet-pass dispatch (`altrios-core/src/meet_pass/`)

Status: **deliverables 1 and 2 are implemented and merged.** What remains open is the
optional inverted index, the minor `train_idx_begin` cleanup, and the out-of-scope
deferred-re-planning project. Read `CLAUDE.md` first (build commands, conventions,
docstring rules).

Measured on the committed benchmark (Taconite, 30 min spacing, release, best of 3), with
checksums bit-identical to the baseline at every N:

| N trains | before | after |
|---|---|---|
| 32  | 0.161 s | 0.061 s |
| 64  | 1.546 s | 0.566 s |
| 128 | 13.85 s | 5.27 s  |

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

## Deliverable 1: gate `validate_free_path` to debug builds — DONE

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

Implemented with `debug_assertions` (no new cargo feature): `#[cfg(debug_assertions)]` on
both the call site and the function itself, documented under "Dispatch path validation" in
the `meet_pass` module doc. Release and debug builds both still emit the same 6
pre-existing crate warnings, so nothing was left dead.

## Deliverable 2: skip trains that cannot be affected — DONE

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

### Benchmark harness

Committed as an `#[ignore]`d integration test at `altrios-core/tests/dispatch_bench.rs`,
with the reference checksums recorded in its module doc. Run with:
```
cd altrios-core
BENCH_NS=32,64,128 cargo test --release --test dispatch_bench -- --ignored --nocapture
```

`BENCH_NETWORK` selects the corridor (`taconite` default, `spokane`, `spokane_plain`,
`winnipeg`) and `BENCH_ROUTES=mixed` spreads trains over every feasible O-D pair instead of
cloning one fixture. Corridors behave very differently — see below — so a result is only ever
a statement about the corridor it was measured on.

Adding `--features dispatch-counters` enables the instrumentation in `meet_pass::counters`
(off by default, zero cost when off) and prints a `COUNT` line per counter. That is how the
numbers in this document were obtained; counters are deterministic, so they are unaffected by
machine load, but they inflate wall-clock times and must not be used for timing.

### Corridor dependence (measured)

Taconite is not representative. Link granularity drives how many dispatch nodes a fixed
search distance covers, and both the share of work attributable to not-yet-departed trains
and the cost of the offset suffix rewrite vary by orders of magnitude:

| | Taconite (N=128) | Spokane-Lethbridge (N=16) | Winnipeg-Chicago (N=16) |
|---|---|---|---|
| links / route-miles | 1,077 / ~370 | 20,749 / ~490 | 60,931 / ~1,060 |
| mi per link | 0.685 | 0.047 | 0.035 |
| `SEARCH_ITERS` per `UPDATE_CALLS` | 134 | 1,196 | 1,160 |
| undeparted share of `SEARCH_ITERS` | 96.8% | 23% | 12% |
| `OFFSET_SUFFIX_ELEMS` / `SEARCH_ITERS` | 0.09% | 12% | **52%** |

Two consequences. Any optimization aimed at not-yet-departed trains is close to worthless
off Taconite - that 96.8% is an artifact of train density against a short corridor, not a
property of dispatch. And the offset suffix rewrite in `update_free_path_helpers`, dismissed
as negligible at 0.09% on Taconite, is half the search work on Winnipeg-Chicago; replacing
the eager `disp_path[idx_join_base..]` rewrite with a range-add structure is worth
revisiting on fine-grained networks.

Scaling is `O(N^3)` on every corridor measured, decomposing as `O(N^2)` advance steps
(sequential, one train at a time) times `O(N)` trains scanned per step. Only the second
factor is parallelizable, and it is also the one the affected-train precheck and any
inverted index attack directly, so parallelism is the wrong lever here.

### Negative result: memoizing the `Blocked` verdict

Measured and rejected. The idea was to cache the `Blocked` verdict for not-yet-departed
trains, keyed on the links the failed search actually consulted, and skip the re-search while
none of those links changed occupant. A shadow implementation (recording witnesses and
re-checking them without acting on the result) gave, per corridor:

| | hit rate of eligible calls | eligible share of `SEARCH_ITERS` | savings ceiling |
|---|---|---|---|
| Taconite (N=64) | 17.6% | 96.8% | ~17% |
| Winnipeg-Chicago (N=16) | 22.5% | 12% | ~2.7% |
| Spokane-Lethbridge (N=16) | 0% | 23% | 0% |

On Spokane 7,018 of 7,303 eligible calls had a memo present and every one was invalidated
before it could be used, so the zero is real invalidation churn rather than missing state. The
predicate itself is sound - a mispredict counter stayed at zero across ~103k hits - it simply
does not fire often enough anywhere that matters. The one corridor where it would pay is the
one that is not representative.

This is an upper bound on the whole family: the witness used was the exact set of links the
search read, which is the tightest possible predicate. Any coarser key (an offset range, for
instance) can only invalidate more often.

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
