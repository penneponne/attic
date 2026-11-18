//! The engine as a WebAssembly module.
//!
//! A thin C ABI over [`attic_protocol::UsiDriver`]: the host feeds it USI lines
//! and reads USI lines back, exactly like a process on stdin and stdout. Two
//! shapes of build exist, picked by the compilation target:
//!
//! * `wasm32-wasip1-threads` — `attic_start` spawns the driver thread, which
//!   pulls its command lines from the host through a blocking `input_read`
//!   import. Everything the engine writes goes out through `emit`. See [`mt`].
//! * `wasm32-wasip1` (built with `--cfg attic_single_thread`, which
//!   `.cargo/config.toml` sets) — no threads exist, so the host calls
//!   `attic_st_command` once per line and every `go` runs to completion inside
//!   that call. See [`st`].
//!
//! The module has no `main`. Everything else — the file system the engine
//! opens its network through, the clock, the thread-spawn import — is the
//! host's WASI implementation.
//!
//! On any other target the crate is empty.
#![cfg(target_family = "wasm")]

/// Bumped when an export's or import's meaning changes, so a loader built for
/// another revision refuses the module instead of misreading it.
pub const ABI_VERSION: u32 = 1;

/// [`ABI_VERSION`], for the host to check before it does anything else.
#[unsafe(no_mangle)]
pub extern "C" fn attic_abi_version() -> u32 {
    ABI_VERSION
}

#[cfg(not(attic_single_thread))]
mod mt;

#[cfg(attic_single_thread)]
mod st;
