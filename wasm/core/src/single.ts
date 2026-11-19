/**
 * The single-threaded engine, for hosts that have no threads (Cloudflare Workers): the module is
 * instantiated in the calling thread, and a `go` runs to completion inside the `send` that issued it.
 * Output reaches the line listeners synchronously, before `send` returns.
 */

import { Vfs } from "./vfs.js";
import { Wasi, WasiExit } from "./wasi.js";

export interface SingleOptions {
  module: WebAssembly.Module;
  vfs: Vfs;
  env: Record<string, string>;
  print: (line: string) => void;
  printErr: (line: string) => void;
}

export class SingleThreadRuntime {
  private readonly listeners = new Set<(line: string) => void>();
  private readonly decoder = new TextDecoder();
  private readonly errDecoder = new TextDecoder();
  private readonly encoder = new TextEncoder();
  private partial = "";
  private errPartial = "";
  private memory!: WebAssembly.Memory;
  private exports!: WebAssembly.Exports;
  private started = false;
  private quit = false;
  private output: Array<{ fd: 1 | 2; line: string }> = [];

  private constructor(private readonly opts: SingleOptions) {}

  static async create(opts: SingleOptions): Promise<SingleThreadRuntime> {
    const rt = new SingleThreadRuntime(opts);
    const wanted = WebAssembly.Module.imports(opts.module);
    const wasi = new Wasi({
      memory: () => rt.memory,
      vfs: opts.vfs,
      env: opts.env,
      write: (fd, bytes) => {
        if (fd === 2) {
          rt.errPartial += rt.errDecoder.decode(bytes, { stream: true });
          let nl: number;
          while ((nl = rt.errPartial.indexOf("\n")) >= 0) {
            rt.output.push({ fd: 2, line: rt.errPartial.slice(0, nl) });
            rt.errPartial = rt.errPartial.slice(nl + 1);
          }
        } else {
          rt.accept(bytes);
        }
      },
      // The only thing that sleeps is a `go infinite`, which this build refuses; and a sleep that
      // cannot wake (Workers freeze the clock while code runs) would never end.
      sleep: null,
    });
    const instance = await WebAssembly.instantiate(opts.module, {
      attic_host: {
        emit: (ptr: number, len: number) => rt.accept(new Uint8Array(rt.memory.buffer, ptr, len).slice()),
      },
      wasi_snapshot_preview1: wasi.imports(
        wanted.filter((i) => i.module === "wasi_snapshot_preview1").map((i) => i.name),
      ) as unknown as WebAssembly.ModuleImports,
    });
    rt.exports = instance.exports;
    rt.memory = instance.exports.memory as WebAssembly.Memory;
    const version = (rt.exports.attic_abi_version as (() => number) | undefined)?.();
    if (version !== 1) throw new Error(`attic-wasm: the module speaks ABI ${version}, this loader speaks 1`);
    (rt.exports._initialize as () => void)();
    return rt;
  }

  get memoryBytes(): number {
    return this.memory.buffer.byteLength;
  }

  addLineListener(fn: (line: string) => void): void {
    this.listeners.add(fn);
  }

  removeLineListener(fn: (line: string) => void): void {
    this.listeners.delete(fn);
  }

  send(line: string): void {
    if (this.quit) throw new Error("attic-wasm: the engine has been disposed");
    if (!this.started) {
      this.started = true;
      if ((this.exports.attic_st_start as () => number)() !== 0) throw new Error("attic_st_start failed");
    }
    const bytes = this.encoder.encode(line);
    const cap = (this.exports.attic_st_input_capacity as () => number)();
    if (bytes.length > cap) throw new Error(`attic-wasm: a command of ${bytes.length} bytes does not fit the ${cap}-byte input buffer`);
    const at = (this.exports.attic_st_input as () => number)();
    new Uint8Array(this.memory.buffer, at, bytes.length).set(bytes);
    let rc: number;
    try {
      rc = (this.exports.attic_st_command as (len: number) => number)(bytes.length);
    } catch (e) {
      this.quit = true;
      if (e instanceof WasiExit) throw new Error(`attic-wasm: the engine exited (${e.code})`);
      throw e;
    }
    if (rc < 0) throw new Error("attic-wasm: the engine refused the command");
    if (rc === 1) this.quit = true;
    // Rust still borrows the driver while it calls emit. Deliver lines after
    // it returns so a listener can safely submit another command.
    const output = this.output;
    this.output = [];
    for (const { fd, line } of output) {
      if (fd === 2) {
        this.opts.printErr(line);
      } else {
        this.opts.print(line);
        for (const fn of this.listeners) fn(line);
      }
    }
  }

  async dispose(): Promise<void> {
    if (this.started && !this.quit) {
      try {
        this.send("quit");
      } catch {
        /* gone already */
      }
    }
    this.quit = true;
    this.listeners.clear();
  }

  private accept(bytes: Uint8Array): void {
    this.partial += this.decoder.decode(bytes, { stream: true });
    let nl: number;
    while ((nl = this.partial.indexOf("\n")) >= 0) {
      const line = this.partial.slice(0, nl);
      this.partial = this.partial.slice(nl + 1);
      this.output.push({ fd: 1, line });
    }
  }
}
