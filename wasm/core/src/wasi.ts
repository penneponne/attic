/**
 * The slice of WASI preview 1 the engine uses, over a {@link Vfs}.
 *
 * The engine reads its network (and optionally a book) with `std::fs`, asks the clock for the time,
 * sleeps with `poll_oneoff`, and writes panic messages to stderr. Anything it does not use is not
 * implemented and answers `ENOSYS`; the table is built from the module's own import list, so a
 * syscall a later std version adds fails at the call, not at instantiation.
 */

import { Vfs, normalizePath } from "./vfs.js";

const ESUCCESS = 0;
const EACCES = 2;
const EBADF = 8;
const EINVAL = 28;
const EISDIR = 31;
const ENOENT = 44;
const ENOSYS = 52;
const ENOTDIR = 54;
const ENOTCAPABLE = 76;

const FILETYPE_DIRECTORY = 3;
const FILETYPE_REGULAR_FILE = 4;
const FILETYPE_CHARACTER_DEVICE = 2;

const OFLAGS_CREAT = 1;
const OFLAGS_DIRECTORY = 2;
const OFLAGS_EXCL = 4;
const OFLAGS_TRUNC = 8;

const ALL_RIGHTS = 0x1fffffffn;

/** Thrown by `proc_exit`; the host catches it where it called into the module. */
export class WasiExit extends Error {
  constructor(readonly code: number) {
    super(`wasi proc_exit(${code})`);
    this.name = "WasiExit";
  }
}

export interface WasiOptions {
  /** The module's memory; a function because a module that defines its own has none until it is instantiated. */
  memory: () => WebAssembly.Memory;
  vfs: Vfs;
  env: Record<string, string>;
  /** Called for each `fd_write` to stdout (1) or stderr (2). */
  write: (fd: 1 | 2, bytes: Uint8Array) => void;
  /** Block the calling thread for this long. `null` where the thread cannot block. */
  sleep: ((ns: bigint) => void) | null;
}

type FdEntry =
  | { kind: "stdio"; fd: number }
  | { kind: "dir"; path: string }
  | { kind: "file"; path: string; data: Uint8Array; pos: number };

export function nowNs(clock: number): bigint {
  if (clock === 0) return BigInt(Math.round(Date.now() * 1000)) * 1000n;
  // Monotonic, process-wide: `performance.now()` counts from the same origin on every thread of a
  // Node process or a page, which `Instant` needs for a time taken on one thread and read on another.
  return BigInt(Math.round((performance.timeOrigin + performance.now()) * 1000)) * 1000n;
}

export class Wasi {
  private readonly fds = new Map<number, FdEntry>();
  private nextFd = 4;
  private readonly opts: WasiOptions;
  private readonly envBlock: Uint8Array[];

  constructor(opts: WasiOptions) {
    this.opts = opts;
    this.fds.set(0, { kind: "stdio", fd: 0 });
    this.fds.set(1, { kind: "stdio", fd: 1 });
    this.fds.set(2, { kind: "stdio", fd: 2 });
    this.fds.set(3, { kind: "dir", path: "/" });
    const enc = new TextEncoder();
    this.envBlock = Object.entries(opts.env).map(([k, v]) => enc.encode(`${k}=${v}\0`));
  }

  /** The import object for `wasi_snapshot_preview1`, restricted to what `module` asks for. */
  imports(wanted: readonly string[]): Record<string, (...args: never[]) => unknown> {
    const table = this.table();
    const out: Record<string, (...args: never[]) => unknown> = {};
    for (const name of wanted) {
      out[name] = (table[name] ?? (() => ENOSYS)) as (...args: never[]) => unknown;
    }
    return out;
  }

  private view(): DataView {
    return new DataView(this.opts.memory().buffer);
  }

  private bytes(ptr: number, len: number): Uint8Array {
    return new Uint8Array(this.opts.memory().buffer, ptr, len);
  }

  private table(): Record<string, (...args: never[]) => unknown> {
    const t: Record<string, unknown> = {
      args_sizes_get: (argcPtr: number, bufSizePtr: number) => {
        const v = this.view();
        v.setUint32(argcPtr, 0, true);
        v.setUint32(bufSizePtr, 0, true);
        return ESUCCESS;
      },
      args_get: () => ESUCCESS,

      environ_sizes_get: (countPtr: number, sizePtr: number) => {
        const v = this.view();
        v.setUint32(countPtr, this.envBlock.length, true);
        v.setUint32(sizePtr, this.envBlock.reduce((n, e) => n + e.length, 0), true);
        return ESUCCESS;
      },
      environ_get: (environPtr: number, bufPtr: number) => {
        const v = this.view();
        let at = bufPtr;
        this.envBlock.forEach((entry, i) => {
          v.setUint32(environPtr + i * 4, at, true);
          this.bytes(at, entry.length).set(entry);
          at += entry.length;
        });
        return ESUCCESS;
      },

      clock_res_get: (_id: number, resPtr: number) => {
        this.view().setBigUint64(resPtr, 1000n, true);
        return ESUCCESS;
      },
      clock_time_get: (id: number, _precision: bigint, timePtr: number) => {
        if (id > 3) return EINVAL;
        this.view().setBigUint64(timePtr, nowNs(id === 0 ? 0 : 1), true);
        return ESUCCESS;
      },

      random_get: (ptr: number, len: number) => {
        // `getRandomValues` refuses a view over shared memory, so fill a private array and copy.
        const scratch = new Uint8Array(Math.min(len, 65536));
        let done = 0;
        while (done < len) {
          const n = Math.min(scratch.length, len - done);
          crypto.getRandomValues(n === scratch.length ? scratch : scratch.subarray(0, n));
          this.bytes(ptr + done, n).set(scratch.subarray(0, n));
          done += n;
        }
        return ESUCCESS;
      },

      sched_yield: () => ESUCCESS,
      proc_exit: (code: number) => {
        throw new WasiExit(code);
      },

      poll_oneoff: (inPtr: number, outPtr: number, nsubs: number, neventsPtr: number) =>
        this.pollOneoff(inPtr, outPtr, nsubs, neventsPtr),

      fd_prestat_get: (fd: number, bufPtr: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "dir" || fd !== 3) return EBADF;
        const v = this.view();
        v.setUint8(bufPtr, 0); // __WASI_PREOPENTYPE_DIR
        v.setUint32(bufPtr + 4, 1, true); // name length: "/"
        return ESUCCESS;
      },
      fd_prestat_dir_name: (fd: number, pathPtr: number, pathLen: number) => {
        if (fd !== 3) return EBADF;
        if (pathLen < 1) return EINVAL;
        this.bytes(pathPtr, 1)[0] = 0x2f;
        return ESUCCESS;
      },

      fd_fdstat_get: (fd: number, bufPtr: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        const v = this.view();
        v.setUint8(bufPtr, e.kind === "dir" ? FILETYPE_DIRECTORY : e.kind === "file" ? FILETYPE_REGULAR_FILE : FILETYPE_CHARACTER_DEVICE);
        v.setUint16(bufPtr + 2, 0, true);
        v.setBigUint64(bufPtr + 8, ALL_RIGHTS, true);
        v.setBigUint64(bufPtr + 16, ALL_RIGHTS, true);
        return ESUCCESS;
      },
      fd_fdstat_set_flags: () => ESUCCESS,

      fd_filestat_get: (fd: number, bufPtr: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        if (e.kind === "stdio") return this.writeFilestat(bufPtr, FILETYPE_CHARACTER_DEVICE, 0);
        if (e.kind === "dir") return this.writeFilestat(bufPtr, FILETYPE_DIRECTORY, 0);
        return this.writeFilestat(bufPtr, FILETYPE_REGULAR_FILE, e.data.byteLength);
      },
      path_filestat_get: (fd: number, _flags: number, pathPtr: number, pathLen: number, bufPtr: number) => {
        const base = this.fds.get(fd);
        if (!base || base.kind !== "dir") return EBADF;
        const stat = this.opts.vfs.stat(this.resolve(base.path, pathPtr, pathLen));
        if (!stat) return ENOENT;
        return this.writeFilestat(bufPtr, stat.kind === "dir" ? FILETYPE_DIRECTORY : FILETYPE_REGULAR_FILE, stat.size);
      },

      path_open: (
        fd: number,
        _dirflags: number,
        pathPtr: number,
        pathLen: number,
        oflags: number,
        _rightsBase: bigint,
        _rightsInheriting: bigint,
        _fdflags: number,
        openedFdPtr: number,
      ) => {
        const base = this.fds.get(fd);
        if (!base || base.kind !== "dir") return EBADF;
        if (oflags & (OFLAGS_CREAT | OFLAGS_EXCL | OFLAGS_TRUNC)) return EACCES; // read-only tree
        const path = this.resolve(base.path, pathPtr, pathLen);
        const stat = this.opts.vfs.stat(path);
        if (!stat) return ENOENT;
        if (stat.kind === "dir") {
          const newFd = this.nextFd++;
          this.fds.set(newFd, { kind: "dir", path });
          this.view().setUint32(openedFdPtr, newFd, true);
          return ESUCCESS;
        }
        if (oflags & OFLAGS_DIRECTORY) return ENOTDIR;
        const data = this.opts.vfs.read(path);
        if (!data) return ENOENT;
        const newFd = this.nextFd++;
        this.fds.set(newFd, { kind: "file", path, data, pos: 0 });
        this.view().setUint32(openedFdPtr, newFd, true);
        return ESUCCESS;
      },

      fd_close: (fd: number) => {
        if (fd <= 3) return ESUCCESS;
        return this.fds.delete(fd) ? ESUCCESS : EBADF;
      },

      fd_read: (fd: number, iovsPtr: number, iovsLen: number, nreadPtr: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        const v = this.view();
        if (e.kind === "stdio") {
          v.setUint32(nreadPtr, 0, true); // stdin is empty: the engine is driven through the host
          return ESUCCESS;
        }
        if (e.kind === "dir") return EISDIR;
        let total = 0;
        for (let i = 0; i < iovsLen; i++) {
          const buf = v.getUint32(iovsPtr + i * 8, true);
          const len = v.getUint32(iovsPtr + i * 8 + 4, true);
          const n = Math.min(len, e.data.byteLength - e.pos);
          if (n > 0) {
            this.bytes(buf, n).set(e.data.subarray(e.pos, e.pos + n));
            e.pos += n;
            total += n;
          }
          if (n < len) break;
        }
        v.setUint32(nreadPtr, total, true);
        return ESUCCESS;
      },
      fd_pread: (fd: number, iovsPtr: number, iovsLen: number, offset: bigint, nreadPtr: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "file") return EBADF;
        const saved = e.pos;
        e.pos = Number(offset);
        const rc = (t.fd_read as (...a: number[]) => number)(fd, iovsPtr, iovsLen, nreadPtr);
        e.pos = saved;
        return rc;
      },

      fd_seek: (fd: number, offset: bigint, whence: number, newOffsetPtr: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "file") return e ? EINVAL : EBADF;
        const base = whence === 0 ? 0 : whence === 1 ? e.pos : e.data.byteLength;
        const next = base + Number(offset);
        if (next < 0) return EINVAL;
        e.pos = next;
        this.view().setBigUint64(newOffsetPtr, BigInt(next), true);
        return ESUCCESS;
      },
      fd_tell: (fd: number, offsetPtr: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "file") return EBADF;
        this.view().setBigUint64(offsetPtr, BigInt(e.pos), true);
        return ESUCCESS;
      },

      fd_write: (fd: number, iovsPtr: number, iovsLen: number, nwrittenPtr: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        if (e.kind !== "stdio" || fd === 0) return ENOTCAPABLE;
        const v = this.view();
        let total = 0;
        for (let i = 0; i < iovsLen; i++) {
          const buf = v.getUint32(iovsPtr + i * 8, true);
          const len = v.getUint32(iovsPtr + i * 8 + 4, true);
          if (len === 0) continue;
          // Copy out of shared memory before anything decodes it.
          this.opts.write(fd as 1 | 2, this.bytes(buf, len).slice());
          total += len;
        }
        v.setUint32(nwrittenPtr, total, true);
        return ESUCCESS;
      },

      fd_readdir: (fd: number, bufPtr: number, bufLen: number, cookie: bigint, usedPtr: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "dir") return EBADF;
        const names = this.opts.vfs.list(e.path);
        const enc = new TextEncoder();
        const v = this.view();
        let used = 0;
        for (let i = Number(cookie); i < names.length; i++) {
          const name = enc.encode(names[i]!);
          const entry = 24 + name.length;
          if (used + entry > bufLen) {
            used = bufLen; // a truncated entry tells the caller to come back with a larger buffer
            break;
          }
          const at = bufPtr + used;
          const full = this.opts.vfs.stat(e.path.replace(/\/$/, "") + "/" + names[i]);
          v.setBigUint64(at, BigInt(i + 1), true); // d_next
          v.setBigUint64(at + 8, 0n, true); // d_ino
          v.setUint32(at + 16, name.length, true);
          v.setUint8(at + 20, full?.kind === "dir" ? FILETYPE_DIRECTORY : FILETYPE_REGULAR_FILE);
          this.bytes(at + 24, name.length).set(name);
          used += entry;
        }
        v.setUint32(usedPtr, used, true);
        return ESUCCESS;
      },
    };
    return t as Record<string, (...args: never[]) => unknown>;
  }

  private resolve(base: string, pathPtr: number, pathLen: number): string {
    const rel = new TextDecoder().decode(this.bytes(pathPtr, pathLen).slice());
    return normalizePath(rel.startsWith("/") ? rel : base + "/" + rel);
  }

  private writeFilestat(bufPtr: number, filetype: number, size: number): number {
    const v = this.view();
    for (let o = 0; o < 64; o += 8) v.setBigUint64(bufPtr + o, 0n, true);
    v.setUint8(bufPtr + 16, filetype);
    v.setBigUint64(bufPtr + 24, 1n, true); // nlink
    v.setBigUint64(bufPtr + 32, BigInt(size), true);
    return ESUCCESS;
  }

  /**
   * Only clock subscriptions block, which is all `thread::sleep` and timed waits ask for; an fd
   * subscription is reported ready at once. Subscriptions are 48 bytes, events 32.
   */
  private pollOneoff(inPtr: number, outPtr: number, nsubs: number, neventsPtr: number): number {
    const v = this.view();
    let waitNs: bigint | null = null;
    const events: Array<{ userdata: bigint; type: number }> = [];
    for (let i = 0; i < nsubs; i++) {
      const at = inPtr + i * 48;
      const userdata = v.getBigUint64(at, true);
      const tag = v.getUint8(at + 8);
      if (tag === 0) {
        const clock = v.getUint32(at + 16, true);
        const timeout = v.getBigUint64(at + 24, true);
        const flags = v.getUint16(at + 40, true);
        const rel = flags & 1 ? timeout - nowNs(clock === 0 ? 0 : 1) : timeout;
        const ns = rel > 0n ? rel : 0n;
        if (waitNs === null || ns < waitNs) waitNs = ns;
        events.push({ userdata, type: 0 });
      } else {
        waitNs = 0n;
        events.push({ userdata, type: tag });
      }
    }
    if (waitNs !== null && waitNs > 0n && this.opts.sleep) this.opts.sleep(waitNs);
    for (let i = 0; i < events.length; i++) {
      const at = outPtr + i * 32;
      for (let o = 0; o < 32; o += 8) v.setBigUint64(at + o, 0n, true);
      v.setBigUint64(at, events[i]!.userdata, true);
      v.setUint8(at + 10, events[i]!.type);
    }
    v.setUint32(neventsPtr, events.length, true);
    return ESUCCESS;
  }
}
