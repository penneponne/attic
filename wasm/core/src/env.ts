/** The few places the three hosts (Node, a browser, a Worker isolate) differ. */

declare const process: { versions?: { node?: string } } | undefined;

export const isNode: boolean = typeof process !== "undefined" && !!process?.versions?.node;

/** One end of a message channel between a worker and the thread that created it. */
export interface WorkerLike {
  postMessage(message: unknown): void;
  onMessage(fn: (message: any) => void): void;
  onError(fn: (error: unknown) => void): void;
  terminate(): Promise<unknown> | void;
}

export type WorkerSpawner = () => WorkerLike;

/**
 * A function that starts a worker running `workerUrl` (the package's `worker.js`), resolved once so
 * that the spawn itself, which `wasi.thread-spawn` needs to be synchronous, is.
 */
export async function makeWorkerSpawner(workerUrl: string | URL): Promise<WorkerSpawner> {
  if (isNode) {
    const wt = (await import(/* @vite-ignore */ "node:" + "worker_threads")) as {
      Worker: new (url: URL | string) => {
        postMessage(m: unknown): void;
        on(event: string, fn: (...a: any[]) => void): void;
        terminate(): Promise<number>;
      };
    };
    return () => {
      const w = new wt.Worker(workerUrl);
      return {
        postMessage: (m) => w.postMessage(m),
        onMessage: (fn) => w.on("message", fn),
        onError: (fn) => w.on("error", fn),
        terminate: () => w.terminate(),
      };
    };
  }
  const W = (globalThis as unknown as { Worker?: new (url: string | URL, o?: object) => any }).Worker;
  if (!W) throw new Error("attic-wasm: this environment has no Worker, so the engine cannot run several threads");
  return () => {
    const w = new W(workerUrl, { type: "module" });
    return {
      postMessage: (m) => w.postMessage(m),
      onMessage: (fn) => {
        w.onmessage = (e: { data: unknown }) => fn(e.data);
      },
      onError: (fn) => {
        w.onerror = (e: unknown) => fn(e);
      },
      terminate: () => w.terminate(),
    };
  };
}

/** The worker's side of the channel: where its messages come from and go to. */
export async function workerPort(): Promise<{ post(m: unknown): void; on(fn: (m: any) => void): void }> {
  if (isNode) {
    const wt = (await import(/* @vite-ignore */ "node:" + "worker_threads")) as {
      parentPort: { postMessage(m: unknown): void; on(e: string, fn: (m: any) => void): void } | null;
    };
    const port = wt.parentPort;
    if (!port) throw new Error("attic-wasm worker.js must be run as a worker");
    return { post: (m) => port.postMessage(m), on: (fn) => port.on("message", fn) };
  }
  const scope = globalThis as unknown as { postMessage(m: unknown): void; onmessage: ((e: { data: unknown }) => void) | null };
  return {
    post: (m) => scope.postMessage(m),
    on: (fn) => {
      scope.onmessage = (e) => fn(e.data);
    },
  };
}
