//! Meet-pass
//!
//! # Nomenclature
//! Blocked: when searching along a path, a train is blocked when it
//! reaches a link occupied by a train travelling in the opposite direction.
//!
//! # Dispatch path validation
//! `TrainDisp::validate_free_path` re-walks the whole dispatch path after every
//! `update_free_path` call to check it for internal consistency. It never changes state and
//! only ever reports errors, but it runs often enough to dominate dispatch runtime, so it
//! is gated on `debug_assertions`: it runs in debug and test builds (including
//! `cargo test`) and is compiled out of release builds, which is what `maturin develop
//! --release` and the Python extension use. Run `cargo test` (debug) to exercise it.

mod disp_imports;
#[macro_use]
pub mod counters;
pub mod disp_structs;
pub mod dispatch;
pub mod est_times;
pub mod train_disp;
