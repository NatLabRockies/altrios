// #![warn(missing_docs)]
// #![warn(missing_docs_in_private_items)]

//! Crate containing models for second-by-second fuel and energy consumption of simulation
//! of locomotive consists comprising collections of individual locomotives, which comprise
//! various powertrain components (engine, generator/alternator, battery, and electric drivetrain)
//! -- all connected to a detailed train model including drag, grade, curvature, bearing, and
//! rolling resistances.  
//!
//! # Helpful Tips
//! Every struct in this crate implements methods for serializing/deserializing itself to/from a
//! handful of standard data formats as strings or file read/write operations using
//! [traits::SerdeAPI].   
//!
//! # Feature Flags
#![doc = document_features::document_features!()]

#[macro_use]
pub mod macros;

#[cfg(test)]
pub mod testing;

pub mod combo_error;
pub mod consist;
pub mod error;
pub mod imports;
pub mod lin_search_hint;
pub mod meet_pass;
pub mod prelude;
pub mod si;
pub mod track;
pub mod train;
pub mod traits;
pub mod uc;
pub mod utils;
pub mod validate;

#[cfg(feature = "pyo3")]
pub mod pyo3;
