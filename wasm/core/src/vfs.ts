/**
 * A read-only, in-memory file tree: what the engine's `std::fs` sees through WASI.
 *
 * Paths are absolute and `/`-separated. Files hold `Uint8Array`s that are used as given, with no
 * copy, so a caller can hand over an `ArrayBuffer` it already has. When threads are in play the
 * bytes must live in a `SharedArrayBuffer` (see {@link Vfs.shared}) so that every worker reads the
 * same memory instead of its own copy.
 */

export interface VfsStat {
  kind: "file" | "dir";
  size: number;
}

/** What crosses to a worker: a path and the (shared) bytes behind it. */
export type VfsSnapshot = ReadonlyArray<readonly [string, Uint8Array]>;

export function normalizePath(path: string): string {
  const out: string[] = [];
  for (const part of path.split("/")) {
    if (part === "" || part === ".") continue;
    if (part === "..") out.pop();
    else out.push(part);
  }
  return "/" + out.join("/");
}

export class Vfs {
  private readonly files = new Map<string, Uint8Array>();
  private readonly dirs = new Set<string>(["/"]);

  /** @param shared copy every written file into a `SharedArrayBuffer`, so workers can read it. */
  constructor(private readonly shared: boolean) {}

  mkdirTree(path: string): void {
    let at = "";
    for (const part of normalizePath(path).split("/").slice(1)) {
      at += "/" + part;
      this.dirs.add(at);
    }
  }

  writeFile(path: string, data: ArrayBuffer | ArrayBufferView): void {
    const norm = normalizePath(path);
    const slash = norm.lastIndexOf("/");
    this.mkdirTree(norm.slice(0, slash) || "/");
    const bytes =
      data instanceof ArrayBuffer
        ? new Uint8Array(data)
        : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    this.files.set(norm, this.shared ? toShared(bytes) : bytes);
  }

  stat(path: string): VfsStat | null {
    const norm = normalizePath(path);
    const file = this.files.get(norm);
    if (file) return { kind: "file", size: file.byteLength };
    if (this.dirs.has(norm)) return { kind: "dir", size: 0 };
    return null;
  }

  read(path: string): Uint8Array | null {
    return this.files.get(normalizePath(path)) ?? null;
  }

  /** The names directly inside `path`, files and directories alike. */
  list(path: string): string[] {
    const norm = normalizePath(path);
    const prefix = norm === "/" ? "/" : norm + "/";
    const names = new Set<string>();
    for (const key of [...this.files.keys(), ...this.dirs]) {
      if (key !== norm && key.startsWith(prefix)) names.add(key.slice(prefix.length).split("/")[0]!);
    }
    return [...names].sort();
  }

  snapshot(): VfsSnapshot {
    return [...this.files.entries()];
  }

  static fromSnapshot(snapshot: VfsSnapshot): Vfs {
    const vfs = new Vfs(false);
    for (const [path, bytes] of snapshot) {
      vfs.mkdirTree(path.slice(0, path.lastIndexOf("/")) || "/");
      vfs.files.set(path, bytes);
    }
    return vfs;
  }
}

function toShared(bytes: Uint8Array): Uint8Array {
  if (typeof SharedArrayBuffer !== "undefined" && bytes.buffer instanceof SharedArrayBuffer) return bytes;
  const copy = new Uint8Array(new SharedArrayBuffer(bytes.byteLength));
  copy.set(bytes);
  return copy;
}
