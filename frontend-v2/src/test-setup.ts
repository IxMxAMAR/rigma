// jsdom, meet Node 25.
//
// Node 25 defines a global `localStorage` that only works when the process was
// started with `--localstorage-file`. Vitest's jsdom environment installs
// jsdom's window onto the Node global, but the built-in wins the NAME — so
// `globalThis.localStorage.getItem` is `undefined` (measured, not assumed), and
// `store.ts`'s module-scope read
// (`localStorage.getItem("rigma.sidebarWidth")`) throws before a single test
// runs.
//
// Repaired ONCE here rather than with a `vi.hoisted` shim copied into every
// jsdom file: the next jsdom test would have to remember the shim, and the
// failure it prevents reads as a bug in the test rather than in the
// environment. Wired in by `vite.config.ts`'s `test.setupFiles`.
//
// Only when the storage is actually broken: a working Storage (a future Node,
// or jsdom's own if it ever wins the name) is left untouched.
const storage = globalThis.localStorage as Storage | undefined;
if (typeof storage?.getItem !== "function") {
  const mem = new Map<string, string>();
  const shim: Storage = {
    get length() {
      return mem.size;
    },
    clear: () => {
      mem.clear();
    },
    getItem: (k) => (mem.has(k) ? mem.get(k)! : null),
    key: (i) => [...mem.keys()][i] ?? null,
    removeItem: (k) => {
      mem.delete(k);
    },
    setItem: (k, v) => {
      mem.set(k, String(v));
    },
  };
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    writable: true,
    value: shim,
  });
}
