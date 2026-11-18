//! The single-threaded build: no thread exists, so the host hands the driver one
//! line per call and a `go` runs to completion inside the call that issued it.
//!
//! The driver's output goes to the host through the `emit` import, synchronously
//! and in the order it is written. The host calls into the module only from one
//! thread, one call at a time, which is what makes the `static mut`s below sound.

use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use attic_protocol::UsiDriver;

/// The most the host can pass as one command line.
const STAGING_CAP: usize = 1 << 16;

#[link(wasm_import_module = "attic_host")]
unsafe extern "C" {
    /// Output bytes, as written: not necessarily whole lines.
    fn emit(ptr: *const u8, len: usize);
}

struct Staging([u8; STAGING_CAP]);

static mut STAGING: Staging = Staging([0; STAGING_CAP]);

type Driver = UsiDriver<io::Empty, EmitWriter>;

static mut DRIVER: Option<Driver> = None;

/// The address of the staging buffer, where the host writes one command line
/// before calling [`attic_st_command`]. Stable for the life of the module.
#[unsafe(no_mangle)]
pub extern "C" fn attic_st_input() -> *mut u8 {
    // SAFETY: single-threaded; see the module comment.
    unsafe { (*std::ptr::addr_of_mut!(STAGING)).0.as_mut_ptr() }
}

/// The staging buffer's capacity in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn attic_st_input_capacity() -> u32 {
    STAGING_CAP as u32
}

/// Create the driver. Returns 0, or 1 if it already exists.
#[unsafe(no_mangle)]
pub extern "C" fn attic_st_start() -> i32 {
    // SAFETY: single-threaded; see the module comment.
    let driver = unsafe { &mut *std::ptr::addr_of_mut!(DRIVER) };
    if driver.is_some() {
        return 1;
    }
    let writer = Arc::new(Mutex::new(EmitWriter));
    *driver = Some(UsiDriver::new(io::empty(), writer));
    0
}

/// Run the `len` bytes the host wrote to the staging buffer as one command.
/// Returns 1 once the command was `quit` (the driver is gone then), 0 while it
/// is still running, and -1 if the driver does not exist or the line is not
/// UTF-8.
#[unsafe(no_mangle)]
pub extern "C" fn attic_st_command(len: u32) -> i32 {
    // SAFETY: single-threaded; see the module comment.
    let (driver, staging) = unsafe {
        (
            &mut *std::ptr::addr_of_mut!(DRIVER),
            &*std::ptr::addr_of!(STAGING),
        )
    };
    let Some(running) = driver.as_mut() else {
        return -1;
    };
    let Some(bytes) = staging.0.get(..len as usize) else {
        return -1;
    };
    let Ok(line) = std::str::from_utf8(bytes) else {
        return -1;
    };
    match running.run_line(line) {
        Ok(false) => 0,
        Ok(true) | Err(_) => {
            *driver = None;
            1
        }
    }
}

struct EmitWriter;

impl Write for EmitWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        // SAFETY: `data` is a live slice; the import only reads it.
        unsafe { emit(data.as_ptr(), data.len()) };
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
