/**
 * Entry point of a worker thread of the engine. The thread that owns the pool sends `init` once,
 * and `start` for each engine thread it wants run here; when the thread's function returns the
 * worker reports `done` and waits for the next `start`, so a worker is reused instead of being
 * created again for every `go`.
 */

import { workerPort } from "./env.js";
import { instantiateThread, ThreadBundle, ThreadInstance } from "./threads.js";

const port = await workerPort();
let instance: Promise<ThreadInstance> | null = null;

function describe(e: unknown): string {
  return e instanceof Error ? `${e.name}: ${e.message}\n${e.stack ?? ""}` : String(e);
}

port.on(async (msg: { t: string; bundle?: ThreadBundle; tid?: number; arg?: number }) => {
  try {
    if (msg.t === "init") {
      instance = instantiateThread(msg.bundle!, {
        spawn: (tid, arg) => port.post({ t: "spawn", tid, arg }),
        outputReady: () => port.post({ t: "out" }),
        driverExit: (code) => port.post({ t: "exit", code }),
        log: (fd, text) => port.post({ t: "log", fd, text }),
      });
      await instance;
    } else if (msg.t === "start") {
      const inst = await instance!;
      inst.runThread(msg.tid!, msg.arg!);
      port.post({ t: "done" });
    }
  } catch (e) {
    port.post({ t: "trap", message: describe(e) });
  }
});
