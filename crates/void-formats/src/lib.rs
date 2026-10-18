//! Decoders for Dishonored 2 resource formats.
//!
//! Each module documents its provenance in `docs/formats/`. Decoders keep
//! fields whose meaning is unknown so callers can report them, and they
//! never depend on a particular engine or renderer.

pub mod arkmat;
pub mod bim;
pub mod bmodel;
pub mod bwm;
pub mod collision;
pub mod dds;
pub mod decl;
pub mod entities;
pub mod envdecl;
mod error;
pub mod havok;
pub mod material;
pub mod md6;
pub mod physworld;
pub mod probes;
mod reader;
pub mod texmod;

pub use error::Error;
