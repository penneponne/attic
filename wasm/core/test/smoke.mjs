import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { dirname } from "node:path";
import { createInterface } from "node:readline";
import { createEngine } from "../dist/index.js";

const [single, threaded, native, network] = process.argv.slice(2);
assert(single && threaded && native, "pass both WASM modules and the native binary");

const files = network ? { "/eval/nn.bin": await readFile(network) } : undefined;
const commands = [
  "usi",
  "setoption name USI_Hash value 16",
  "isready",
  "usinewgame",
  "position startpos moves 7g7f 3c3d",
  "go depth 3",
];
const bestmove = /^bestmove [1-9][a-i][1-9][a-i].*$/m;

async function search(path, threads, interrupt = false) {
  const bytes = await readFile(path);
  const wasm = path === single ? await WebAssembly.compile(bytes) : bytes;
  const engine = await createEngine({ wasm, files });
  const lines = [];
  let timer;
  try {
    // Submit the next command from the line callback. For inline WASM this
    // exercises the boundary between a Rust command and its host callbacks.
    await new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error(`Timed out in ${path}: ${lines.slice(-4).join("\n")}`)), 10000);
      let stopped = false;
      const listener = (line) => {
        lines.push(line);
        if (line === "usiok") {
          engine.send(`setoption name Threads value ${threads}`);
          engine.send(commands[1]);
          engine.send(commands[2]);
        } else if (line === "readyok") {
          for (const command of commands.slice(3, -1)) engine.send(command);
          if (interrupt) engine.send("setoption name PvInterval value 0");
          engine.send(interrupt ? "go infinite" : commands.at(-1));
        } else if (interrupt && !stopped && line.startsWith("info depth ")) {
          stopped = true;
          engine.send("stop");
        } else if (bestmove.test(line)) {
          if (interrupt && !stopped) reject(new Error("go infinite returned before stop"));
          else resolve();
        }
      };
      engine.addLineListener(listener);
      try {
        engine.send(commands[0]);
      } catch (error) {
        reject(error);
      }
    });
    return lines.find((line) => bestmove.test(line));
  } finally {
    clearTimeout(timer);
    await engine.dispose();
  }
}

async function searchNative() {
  // Give a native search time to finish before sending quit.
  const child = spawn(native, { stdio: ["pipe", "pipe", "inherit"] });
  try {
    const output = createInterface({ input: child.stdout });
    const evalDir = network ? `setoption name EvalDir value ${dirname(network)}` : "";
    child.stdin.write([commands[0], "setoption name Threads value 1", evalDir, ...commands.slice(1)].filter(Boolean).join("\n") + "\n");
    for await (const line of output) {
      if (bestmove.test(line)) return line;
    }
    assert.fail("native engine exited without a bestmove");
  } finally {
    child.stdin.end("quit\n");
  }
}

const expected = await searchNative();
assert.equal(await search(single, 1), expected);
assert.equal(await search(threaded, 1), expected);
assert.match(await search(threaded, 2), bestmove);
assert.match(await search(threaded, 2, true), bestmove);
console.log(`Native and one-thread WASM searches agree: ${expected}`);

const missingWorker = new URL("./missing-worker.js", import.meta.url);
let reportFailure;
const failed = new Promise((resolve) => { reportFailure = resolve; });
const broken = await createEngine({
  wasm: await readFile(threaded),
  workerUrl: missingWorker,
  printErr: reportFailure,
});
try {
  broken.send("usi");
  assert.match(await failed, /missing-worker/);
  assert.throws(() => broken.send("isready"), /missing-worker/);
} finally {
  await broken.dispose();
}
