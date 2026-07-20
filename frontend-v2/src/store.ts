// App-shell state. Zustand, functional updates only — see
// docs/design/UI-REWORK-PLAN.md "Bug-class defences": streaming state will
// always be keyed by id, never by "last thing rendered".
import { create } from "zustand";

export type Surface =
  | "chat"
  | "autonomous"
  | "models"
  | "engine"
  | "memory"
  | "settings";

export const SURFACES: { id: Surface; label: string; hint: string }[] = [
  { id: "chat", label: "Chat", hint: "converse with the loaded model" },
  { id: "autonomous", label: "Autonomous", hint: "unattended missions" },
  { id: "models", label: "Models", hint: "the hangar — install and manage" },
  { id: "engine", label: "Engine", hint: "load, switch, telemetry" },
  { id: "memory", label: "Memory", hint: "what the agent has learned" },
  { id: "settings", label: "Settings", hint: "presets and preferences" },
];

interface ServerStatus {
  model: string;
  quant: string;
  tps: number | null;
  healthy: boolean;
  ctx: number;
}

interface AppState {
  surface: Surface;
  paletteOpen: boolean;
  server: ServerStatus | null;
  setSurface: (s: Surface) => void;
  setPalette: (open: boolean) => void;
  setServer: (s: ServerStatus | null) => void;
}

export const useApp = create<AppState>((set) => ({
  surface: "chat",
  paletteOpen: false,
  server: null,
  setSurface: (surface) => set({ surface, paletteOpen: false }),
  setPalette: (paletteOpen) => set({ paletteOpen }),
  setServer: (server) => set({ server }),
}));
