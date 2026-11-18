/**
 * Everything one thread of the engine needs to run its share of the module: the WASI host, the
 * `attic_host` imports, and the `wasi.thread-spawn` import. The same code runs on the thread that
 * created the engine and on every worker, so each of them differs only in how it reaches the
 * thread that owns the pool ({@link ThreadHost}).
 */

import { Vfs, VfsSnapshot } from "./vfs.js";
import { Wasi } from "./wasi.js";

/** Capacity of each byte ring between the engine and its host, in bytes (a power of two). */
export const RING_CAP = 1 << 20;

/** Indices into the shared control block (an `Int32Array` over a `SharedArrayBuffer`). */
export const CTRL = {
  IN_HEAD: 0,
  IN_TAIL: 1,
  IN_SEQ: 2,
  IN_CLOSED: 3,
  OUT_HEAD: 4,
  OUT_TAIL: 5,
  OUT_PENDING: 6,
  TID: 7,
  EXITED: 8,
  EXIT_CODE: 9,
  LENGTH: 16,
} as const;

/** What a worker is given to join the engine: all of it shared, none of it copied. */
export interface ThreadBundle {
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
  ctrl: SharedArrayBuffer;
  inRing: SharedArrayBuffer;
  outRing: SharedArrayBuffer;
  vfs: VfsSnapshot;
  env: Record<string, string>;
}

/** How a thread reaches the one that owns the pool. */
export interface ThreadHost {
  /** Ask for a new thread running `wasi_thread_start(tid, arg)`. */
  spawn(tid: number, arg: number): void;
  /** The output ring has bytes the host has not read. */
  outputReady(): void;
  /** The driver thread has returned. */
  driverExit(code: number): void;
  /** Text the engine wrote to stdout (1) or stderr (2). */
  log(fd: 1 | 2, text: string): void;
}

export interface ThreadInstance {
  exports: WebAssembly.Exports;
  /** Run one thread's entry point to its end. */
  runThread(tid: number, arg: number): void;
}

export async function instantiateThread(bundle: ThreadBundle, host: ThreadHost): Promise<ThreadInstance> {
  const ctrl = new Int32Array(bundle.ctrl);
  const inRing = new Uint8Array(bundle.inRing);
  const outRing = new Uint8Array(bundle.outRing);
  const mask = RING_CAP - 1;
  const sleepCell = new Int32Array(new SharedArrayBuffer(4));
  const decoder = new TextDecoder();
  const partial: { 1: string; 2: string } = { 1: "", 2: "" };

  const wasi = new Wasi({
    memory: () => bundle.memory,
    vfs: Vfs.fromSnapshot(bundle.vfs),
    env: bundle.env,
    write: (fd, bytes) => {
      // Whole lines go out as they complete, so a panic message arrives as one piece.
      partial[fd] += decoder.decode(bytes, { stream: true });
      let nl: number;
      while ((nl = partial[fd].indexOf("\n")) >= 0) {
        host.log(fd, partial[fd].slice(0, nl));
        partial[fd] = partial[fd].slice(nl + 1);
      }
    },
    sleep: (ns) => {
      Atomics.wait(sleepCell, 0, 0, Number(ns) / 1e6);
    },
  });

  const mem = () => new Uint8Array(bundle.memory.buffer);

  const attic_host = {
    input_read(ptr: number, cap: number): number {
      for (;;) {
        const seq = Atomics.load(ctrl, CTRL.IN_SEQ);
        const head = Atomics.load(ctrl, CTRL.IN_HEAD);
        const tail = Atomics.load(ctrl, CTRL.IN_TAIL);
        const avail = (head - tail) >>> 0;
        if (avail > 0) {
          const n = Math.min(avail, cap);
          const off = tail & mask;
          const first = Math.min(n, RING_CAP - off);
          const m = mem();
          m.set(inRing.subarray(off, off + first), ptr);
          if (first < n) m.set(inRing.subarray(0, n - first), ptr + first);
          Atomics.store(ctrl, CTRL.IN_TAIL, (tail + n) | 0);
          return n;
        }
        if (Atomics.load(ctrl, CTRL.IN_CLOSED) !== 0) return 0;
        Atomics.wait(ctrl, CTRL.IN_SEQ, seq);
      }
    },
    emit(ptr: number, len: number): void {
      const m = mem();
      let done = 0;
      while (done < len) {
        const head = Atomics.load(ctrl, CTRL.OUT_HEAD);
        const tail = Atomics.load(ctrl, CTRL.OUT_TAIL);
        const free = RING_CAP - ((head - tail) >>> 0);
        if (free === 0) {
          Atomics.wait(ctrl, CTRL.OUT_TAIL, tail);
          continue;
        }
        const n = Math.min(free, len - done);
        const off = head & mask;
        const first = Math.min(n, RING_CAP - off);
        outRing.set(m.subarray(ptr + done, ptr + done + first), off);
        if (first < n) outRing.set(m.subarray(ptr + done + first, ptr + done + n), 0);
        Atomics.store(ctrl, CTRL.OUT_HEAD, (head + n) | 0);
        done += n;
        if (Atomics.compareExchange(ctrl, CTRL.OUT_PENDING, 0, 1) === 0) host.outputReady();
      }
    },
    driver_exit(code: number): void {
      Atomics.store(ctrl, CTRL.EXIT_CODE, code);
      Atomics.store(ctrl, CTRL.EXITED, 1);
      host.driverExit(code);
    },
  };

  const threadSpawn = (arg: number): number => {
    const tid = Atomics.add(ctrl, CTRL.TID, 1) + 1;
    host.spawn(tid, arg);
    return tid;
  };

  const wanted = WebAssembly.Module.imports(bundle.module);
  const imports: WebAssembly.Imports = {
    env: { memory: bundle.memory },
    attic_host: attic_host as unknown as WebAssembly.ModuleImports,
    wasi: { "thread-spawn": threadSpawn },
    wasi_snapshot_preview1: wasi.imports(
      wanted.filter((i) => i.module === "wasi_snapshot_preview1").map((i) => i.name),
    ) as unknown as WebAssembly.ModuleImports,
  };

  const instance = await WebAssembly.instantiate(bundle.module, imports);
  const exports = instance.exports;

  return {
    exports,
    runThread(tid: number, arg: number): void {
      const start = exports.wasi_thread_start as ((tid: number, arg: number) => void) | undefined;
      if (!start) throw new Error("the module does not export wasi_thread_start");
      start(tid, arg);
    },
  };
}
