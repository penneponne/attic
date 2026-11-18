/**
 * The multi-threaded engine: one `WebAssembly.Memory` shared by a pool of workers, one of which runs
 * the USI driver loop. The host talks to the driver through two byte rings (see `threads.ts`) and never
 * calls into the module after `attic_start`, so it can be a browser's main thread.
 */

import { WorkerLike, WorkerSpawner } from "./env.js";
import { CTRL, instantiateThread, RING_CAP, ThreadBundle, ThreadInstance } from "./threads.js";
import { Vfs } from "./vfs.js";

export interface ThreadedOptions {
  module: WebAssembly.Module;
  memory: WebAssembly.Memory;
  vfs: Vfs;
  env: Record<string, string>;
  spawner: WorkerSpawner;
  print: (line: string) => void;
  printErr: (line: string) => void;
}

export class ThreadedRuntime {
  private readonly ctrl: Int32Array;
  private readonly inRing: Uint8Array;
  private readonly outRing: Uint8Array;
  private readonly bundle: ThreadBundle;
  private readonly workers: WorkerLike[] = [];
  private readonly idle: WorkerLike[] = [];
  private readonly lineListeners = new Set<(line: string) => void>();
  private readonly decoder = new TextDecoder();
  private readonly encoder = new TextEncoder();
  private partial = "";
  private main!: ThreadInstance;
  private started = false;
  private disposed = false;
  private disposing: Promise<void> | undefined;
  private exitWaiters: Array<() => void> = [];
  private fatalHandlers: Array<(error: Error) => void> = [];
  private spawned = 0;

  /** The first fatal error: a trap in any thread, or a worker that failed to start. */
  fatal: Error | null = null;

  private constructor(private readonly opts: ThreadedOptions) {
    const ctrl = new SharedArrayBuffer(CTRL.LENGTH * 4);
    const inRing = new SharedArrayBuffer(RING_CAP);
    const outRing = new SharedArrayBuffer(RING_CAP);
    this.ctrl = new Int32Array(ctrl);
    this.inRing = new Uint8Array(inRing);
    this.outRing = new Uint8Array(outRing);
    this.bundle = {
      module: opts.module,
      memory: opts.memory,
      ctrl,
      inRing,
      outRing,
      vfs: opts.vfs.snapshot(),
      env: opts.env,
    };
  }

  static async create(opts: ThreadedOptions): Promise<ThreadedRuntime> {
    const rt = new ThreadedRuntime(opts);
    rt.main = await instantiateThread(rt.bundle, {
      spawn: (tid, arg) => rt.startThread(tid, arg),
      outputReady: () => rt.drain(),
      driverExit: (code) => rt.onDriverExit(code),
      log: (fd, text) => rt.log(fd, text),
    });
    const version = (rt.main.exports.attic_abi_version as (() => number) | undefined)?.();
    if (version !== 1) throw new Error(`attic-wasm: the module speaks ABI ${version}, this loader speaks 1`);
    // Workers enter through wasi_thread_start, which initializes their own TLS.
    (rt.main.exports._initialize as () => void)();
    return rt;
  }

  /** Workers created so far (idle ones included). */
  get workerCount(): number {
    return this.workers.length;
  }

  /** Engine threads started so far, the driver's included. */
  get threadsSpawned(): number {
    return this.spawned;
  }

  get memoryBytes(): number {
    return this.opts.memory.buffer.byteLength;
  }

  addLineListener(fn: (line: string) => void): void {
    this.lineListeners.add(fn);
  }

  removeLineListener(fn: (line: string) => void): void {
    this.lineListeners.delete(fn);
  }

  onFatal(fn: (error: Error) => void): void {
    this.fatalHandlers.push(fn);
    if (this.fatal) fn(this.fatal);
  }

  /** Spawn the driver thread. Called once, by the first command. */
  private start(): void {
    try {
      const rc = (this.main.exports.attic_start as () => number)();
      if (rc !== 0) throw new Error(`attic_start failed (${rc})`);
      this.started = true;
    } catch (cause) {
      const error = new Error("attic-wasm: could not start the driver", { cause });
      this.fail(error);
      throw error;
    }
  }

  send(line: string): void {
    if (this.fatal) throw this.fatal;
    if (this.disposed || Atomics.load(this.ctrl, CTRL.EXITED) !== 0) {
      throw new Error("attic-wasm: the engine has exited");
    }
    if (!this.started) this.start();
    const bytes = this.encoder.encode(line + "\n");
    const head = Atomics.load(this.ctrl, CTRL.IN_HEAD);
    const tail = Atomics.load(this.ctrl, CTRL.IN_TAIL);
    if (bytes.length > RING_CAP - ((head - tail) >>> 0)) {
      throw new Error("attic-wasm: the engine's input is full; it is not reading commands");
    }
    const off = head & (RING_CAP - 1);
    const first = Math.min(bytes.length, RING_CAP - off);
    this.inRing.set(bytes.subarray(0, first), off);
    if (first < bytes.length) this.inRing.set(bytes.subarray(first), 0);
    Atomics.store(this.ctrl, CTRL.IN_HEAD, (head + bytes.length) | 0);
    Atomics.add(this.ctrl, CTRL.IN_SEQ, 1);
    Atomics.notify(this.ctrl, CTRL.IN_SEQ);
  }

  /** Stop the engine: `quit`, a bounded wait for the driver to return, then every worker is terminated. */
  dispose(timeoutMs = 3000): Promise<void> {
    return this.disposing ??= this.close(timeoutMs);
  }

  private async close(timeoutMs: number): Promise<void> {
    if (this.started && Atomics.load(this.ctrl, CTRL.EXITED) === 0 && !this.fatal) {
      try {
        this.send("quit");
        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, timeoutMs);
          this.exitWaiters.push(() => {
            clearTimeout(timer);
            resolve();
          });
        });
      } catch {
        /* terminated below regardless */
      }
    }
    this.disposed = true;
    this.exitWaiters.length = 0;
    Atomics.store(this.ctrl, CTRL.IN_CLOSED, 1);
    Atomics.add(this.ctrl, CTRL.IN_SEQ, 1);
    Atomics.notify(this.ctrl, CTRL.IN_SEQ);
    await Promise.allSettled(this.workers.map(async (worker) => worker.terminate()));
    this.workers.length = 0;
    this.idle.length = 0;
    this.lineListeners.clear();
  }

  // -- the pool -----------------------------------------------------------------------------

  private startThread(tid: number, arg: number): void {
    this.spawned++;
    const worker = this.idle.pop() ?? this.newWorker();
    worker.postMessage({ t: "start", tid, arg });
  }

  private newWorker(): WorkerLike {
    const worker = this.opts.spawner();
    worker.onMessage((m) => this.onWorkerMessage(worker, m));
    worker.onError((e) => this.fail(e instanceof Error ? e : new Error(String((e as { message?: string })?.message ?? e))));
    this.workers.push(worker);
    worker.postMessage({ t: "init", bundle: this.bundle });
    return worker;
  }

  private onWorkerMessage(worker: WorkerLike, m: { t: string; [k: string]: any }): void {
    if (this.disposed || this.fatal) return;
    switch (m.t) {
      case "spawn":
        try {
          this.startThread(m.tid, m.arg);
        } catch (cause) {
          this.fail(new Error("attic-wasm: could not start a worker", { cause }));
        }
        break;
      case "out":
        this.drain();
        break;
      case "done":
        if (!this.disposed) this.idle.push(worker);
        break;
      case "exit":
        this.onDriverExit(m.code);
        break;
      case "log":
        this.log(m.fd, m.text);
        break;
      case "trap":
        this.fail(new Error(`attic-wasm: a thread trapped: ${m.message}`));
        break;
    }
  }

  private fail(error: Error): void {
    if (this.fatal || this.disposed) return;
    this.fatal = error;
    this.onDriverExit(0);
    void this.dispose();
    this.opts.printErr(error.message);
    for (const fn of this.fatalHandlers) fn(error);
  }

  private onDriverExit(code: number): void {
    if (code !== 0) {
      this.fail(new Error(`attic-wasm: the driver exited with code ${code}`));
      return;
    }
    this.drain();
    const waiters = this.exitWaiters;
    this.exitWaiters = [];
    for (const fn of waiters) fn();
  }

  private log(fd: 1 | 2, text: string): void {
    (fd === 1 ? this.opts.print : this.opts.printErr)(text);
  }

  // -- output ---------------------------------------------------------------------------------

  /** Move whatever the engine has written out of the ring and split it into lines. */
  private drain(): void {
    // Clear the flag before reading the head, so a writer that publishes after this point raises it again.
    Atomics.store(this.ctrl, CTRL.OUT_PENDING, 0);
    for (;;) {
      const head = Atomics.load(this.ctrl, CTRL.OUT_HEAD);
      const tail = Atomics.load(this.ctrl, CTRL.OUT_TAIL);
      const avail = (head - tail) >>> 0;
      if (avail === 0) return;
      const off = tail & (RING_CAP - 1);
      const first = Math.min(avail, RING_CAP - off);
      // `slice` copies out of shared memory; a decoder will not read a shared view.
      this.partial += this.decoder.decode(this.outRing.slice(off, off + first), { stream: true });
      if (first < avail) this.partial += this.decoder.decode(this.outRing.slice(0, avail - first), { stream: true });
      Atomics.store(this.ctrl, CTRL.OUT_TAIL, head);
      Atomics.notify(this.ctrl, CTRL.OUT_TAIL);
      let nl: number;
      while ((nl = this.partial.indexOf("\n")) >= 0) {
        const line = this.partial.slice(0, nl);
        this.partial = this.partial.slice(nl + 1);
        this.opts.print(line);
        for (const fn of this.lineListeners) fn(line);
      }
    }
  }
}
