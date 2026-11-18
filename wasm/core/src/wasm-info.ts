/** What the loader reads off a module's bytes without instantiating it. */

export interface MemoryImport {
  initialPages: number;
  maximumPages: number | null;
  shared: boolean;
}

/** The limits of the module's imported memory, or `null` if it defines its own (no threads). */
export function readMemoryImport(bytes: Uint8Array): MemoryImport | null {
  let p = 8; // magic + version
  const u32 = (): number => {
    let result = 0;
    let shift = 0;
    let byte: number;
    do {
      byte = bytes[p++]!;
      result |= (byte & 0x7f) << shift;
      shift += 7;
    } while (byte & 0x80);
    return result >>> 0;
  };
  const skipName = (): void => {
    const length = u32();
    p += length;
  };
  while (p < bytes.length) {
    const id = bytes[p++]!;
    const size = u32();
    const end = p + size;
    if (id === 2) {
      const count = u32();
      for (let i = 0; i < count; i++) {
        skipName();
        skipName();
        const kind = bytes[p++]!;
        if (kind === 0) u32();
        else if (kind === 1) {
          p++;
          const flags = bytes[p++]!;
          u32();
          if (flags & 1) u32();
        } else if (kind === 2) {
          const flags = bytes[p++]!;
          const initialPages = u32();
          const maximumPages = flags & 1 ? u32() : null;
          return { initialPages, maximumPages, shared: (flags & 2) !== 0 };
        } else if (kind === 3) p += 2;
      }
    }
    p = end;
  }
  return null;
}
