//! The threaded build: the driver runs on its own thread and reads its command
//! lines from the host, which also takes every byte of output.
//!
//! The host's side of both streams is a ring buffer in a `SharedArrayBuffer` it
//! owns, so a host thread that must not block (a browser's main thread) can feed
//! and drain the engine without a call into the module. The module sees them as
//! two imports: [`input_read`] blocks the driver thread until a command line is
//! there, and [`emit`] hands over output, blocking only while the host has not
//! made room.

use std::io::{self, BufRead, Read, Write};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use attic_protocol::UsiDriver;

/// Stack of the driver thread. Everything it runs on its own is the command
/// loop, `isready`'s network load and the table clear; a search runs on a
/// thread of its own.
const DRIVER_STACK: usize = 8 << 20;

/// How much the driver asks the host for at a time.
const READ_CHUNK: usize = 1 << 16;

/// `STATE` values.
const STATE_IDLE: u32 = 0;
const STATE_RUNNING: u32 = 1;

static STATE: AtomicU32 = AtomicU32::new(STATE_IDLE);

#[link(wasm_import_module = "attic_host")]
unsafe extern "C" {
    /// Copy up to `cap` bytes of the host's input into `ptr`, waiting until there
    /// is at least one. Returns the count, or 0 once the host has closed the
    /// input and every byte before that has been read.
    fn input_read(ptr: *mut u8, cap: usize) -> usize;
    /// Output bytes, as written: not necessarily whole lines. The host takes
    /// all `len` bytes before it returns.
    fn emit(ptr: *const u8, len: usize);
    /// The driver thread has returned. 0 for a clean `quit` or end of input.
    fn driver_exit(code: i32);
}

/// Start the driver thread. Returns 0, or 1 if it could not be spawned or was
/// already started.
#[unsafe(no_mangle)]
pub extern "C" fn attic_start() -> i32 {
    if STATE
        .compare_exchange(
            STATE_IDLE,
            STATE_RUNNING,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        return 1;
    }
    let spawned = std::thread::Builder::new()
        .name("attic-usi".into())
        .stack_size(DRIVER_STACK)
        .spawn(driver_main);
    match spawned {
        Ok(_detached) => 0,
        Err(_) => 1,
    }
}

fn driver_main() {
    let writer = Arc::new(Mutex::new(HostOut));
    let code = match UsiDriver::new(HostIn::new(), writer).run() {
        Ok(()) => 0,
        Err(_) => 1,
    };
    // SAFETY: the import takes no pointers.
    unsafe { driver_exit(code) };
}

/// The driver's input: the host's byte stream, buffered a chunk at a time.
struct HostIn {
    chunk: Box<[u8]>,
    pos: usize,
    len: usize,
}

impl HostIn {
    fn new() -> Self {
        HostIn {
            chunk: vec![0; READ_CHUNK].into_boxed_slice(),
            pos: 0,
            len: 0,
        }
    }
}

impl Read for HostIn {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let n = available.len().min(out.len());
        out[..n].copy_from_slice(&available[..n]);
        self.consume(n);
        Ok(n)
    }
}

impl BufRead for HostIn {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos == self.len {
            // SAFETY: `chunk` is a live buffer of `READ_CHUNK` bytes the host
            // writes at most that many into.
            self.len = unsafe { input_read(self.chunk.as_mut_ptr(), self.chunk.len()) };
            self.pos = 0;
        }
        Ok(&self.chunk[self.pos..self.len])
    }

    fn consume(&mut self, amount: usize) {
        self.pos = (self.pos + amount).min(self.len);
    }
}

/// The engine's output. Every thread writes through the driver's writer mutex,
/// so the host sees the lines in the order that serialised them.
struct HostOut;

impl Write for HostOut {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        // SAFETY: `data` is a live slice; the host only reads it.
        unsafe { emit(data.as_ptr(), data.len()) };
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
