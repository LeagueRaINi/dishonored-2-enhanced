//! Read-only access to Dishonored 2 (Void engine) resource archives.
//!
//! Format notes and their provenance live in `docs/formats/archives.md`.
//! Opaque resource contents are returned byte-exact; this crate does not
//! interpret resource types.

mod error;
pub mod index;
pub mod install;
pub mod mapres;
pub mod master;
pub mod path;
mod reader;

pub use error::Error;
pub use index::{Entry, Index, Location, Lookup};
pub use install::{Group, Installation};
pub use mapres::MapResources;
pub use master::MasterIndex;
