// Responsibility: Define the public crate surface and connect the application modules.
// Input: Validated runtime configuration supplied by the executable or another Rust caller.
// Output: Application lifecycle results and public domain types needed by callers.

pub mod capture;
pub mod config;
pub mod domain;
