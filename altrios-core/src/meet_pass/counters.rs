//! Optional instrumentation counters for the meet-pass dispatch hot path.
//!
//! Enabled by the non-default `dispatch-counters` cargo feature. With the feature off the
//! `dcount!` macro expands to nothing, so there is no cost in normal builds; `reset` and
//! `report` remain callable and simply do nothing. Counting perturbs timing by a few
//! percent, so treat a counting run as a source of ratios, not of benchmark timings.
//!
//! See `altrios-core/tests/dispatch_bench.rs` for the reporting harness.

/// Increment a counter, by 1 or by an explicit amount. Expands to nothing without the
/// `dispatch-counters` feature.
macro_rules! dcount {
    ($name:ident) => {
        dcount!($name, 1)
    };
    ($name:ident, $n:expr) => {{
        #[cfg(feature = "dispatch-counters")]
        {
            $crate::meet_pass::counters::inner::$name
                .fetch_add($n as u64, std::sync::atomic::Ordering::Relaxed);
        }
    }};
}

#[cfg(feature = "dispatch-counters")]
pub mod inner {
    use std::sync::atomic::{AtomicU64, Ordering};

    macro_rules! declare_counters {
        ($($name:ident),* $(,)?) => {
            $(pub static $name: AtomicU64 = AtomicU64::new(0);)*

            pub fn reset() {
                $($name.store(0, Ordering::Relaxed);)*
            }

            pub fn report() -> Vec<(&'static str, u64)> {
                vec![$((stringify!($name), $name.load(Ordering::Relaxed))),*]
            }
        };
    }

    declare_counters!(
        // Outer dispatch loop
        ADVANCE_CALLS,
        REWIND_CALLS,
        CHECK_DEADLOCK_CALLS,
        // Affected-train precheck (see TrainDisp::may_be_affected)
        PRECHECK_TESTED,
        PRECHECK_SKIPPED,
        // Free path update. SEARCH_ITERS dominates runtime; the *_UNDEPARTED variants
        // isolate trains that have not left their origin, whose share varies enormously
        // by corridor (97% of search work on Taconite, ~19% on Spokane-Lethbridge).
        UPDATE_CALLS,
        WALK_ITERS,
        SEARCH_ITERS,
        BLOCKED_RETURNS,
        REPLAN_HELPERS,
        UPDATE_CALLS_UNDEPARTED,
        SEARCH_ITERS_UNDEPARTED,
        BLOCKED_UNDEPARTED,
        // Suffix rewrites on a successful re-plan. Negligible on coarse networks,
        // 12-30% of SEARCH_ITERS on fine-grained ones.
        OFFSET_SUFFIX_ELEMS,
        DIV_SUFFIX_ELEMS,
        DIV_SPLICE_ELEMS,
    );
}

/// Zero every counter. No-op without the `dispatch-counters` feature.
pub fn reset() {
    #[cfg(feature = "dispatch-counters")]
    inner::reset();
}

/// Current counter values as `(name, value)` pairs, in declaration order. Empty without
/// the `dispatch-counters` feature.
pub fn report() -> Vec<(&'static str, u64)> {
    #[cfg(feature = "dispatch-counters")]
    {
        inner::report()
    }
    #[cfg(not(feature = "dispatch-counters"))]
    {
        Vec::new()
    }
}
