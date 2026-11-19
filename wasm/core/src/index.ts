import { makeWorkerSpawner } from "./env.js";
import { SingleThreadRuntime } from "./single.js";
import { ThreadedRuntime } from "./threaded.js";
import { Vfs } from "./vfs.js";
import { readMemoryImport } from "./wasm-info.js";

export interface EngineOptions {
  /** Supply bytes for either build, or a precompiled module for the single-threaded build. */
  wasm: ArrayBuffer | Uint8Array | WebAssembly.Module;
  files?: Readonly<Record<string, ArrayBuffer | ArrayBufferView>>;
  env?: Readonly<Record<string, string>>;
  print?: (line: string) => void;
  printErr?: (line: string) => void;
  workerUrl?: URL;
}

export interface Engine {
  send(line: string): void;
  addLineListener(listener: (line: string) => void): void;
  removeLineListener(listener: (line: string) => void): void;
  dispose(): Promise<void>;
}

/** Instantiate an Attic WASI module. Virtual files must be supplied before the first command. */
export async function createEngine(options: EngineOptions): Promise<Engine> {
  const source = options.wasm;
  const bytes = source instanceof WebAssembly.Module ? null
    : source instanceof ArrayBuffer ? new Uint8Array(source) : source;
  const module = source instanceof WebAssembly.Module
    ? source
    : await WebAssembly.compile(source as BufferSource);
  const limits = bytes ? readMemoryImport(bytes) : null;
  if (!bytes && WebAssembly.Module.imports(module).some((entry) => entry.kind === "memory")) {
    throw new Error("attic-wasm: pass the WASM bytes for the threaded module");
  }
  const vfs = new Vfs(limits?.shared === true);
  for (const [path, data] of Object.entries(options.files ?? {})) vfs.writeFile(path, data);

  const config = {
    module,
    vfs,
    env: { ...options.env },
    print: options.print ?? (() => {}),
    printErr: options.printErr ?? ((line: string) => console.error(line)),
  };
  if (!limits) return SingleThreadRuntime.create(config);
  if (!limits.shared || limits.maximumPages === null) {
    throw new Error("attic-wasm: expected a shared memory import with a maximum");
  }

  const memory = new WebAssembly.Memory({
    initial: limits.initialPages,
    maximum: limits.maximumPages,
    shared: true,
  });
  const spawner = await makeWorkerSpawner(options.workerUrl ?? new URL("./worker.js", import.meta.url));
  return ThreadedRuntime.create({ ...config, memory, spawner });
}
