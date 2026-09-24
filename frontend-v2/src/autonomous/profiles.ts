// The autonomous-run safety profiles (IMP-4).
//
// `POST /api/runs` has accepted `profile` all along (runs.PROFILES, applied in
// tools.tool_specs and the call gate), and the legacy UI offered a bare select
// of the four names with no explanation. The v2 launcher offered none, so a run
// always got "all" and the main safety control was invisible. These are the
// names plus one honest sentence each, read from tools.py — not from the name.

export interface ProfileDef {
  value: string;
  label: string;
  /** What choosing this profile withholds or restricts. */
  consequence: string;
}

export const PROFILES: readonly ProfileDef[] = [
  {
    value: "all",
    label: "all — full tools",
    consequence:
      "Code execution, deletes, the network tools and MCP servers are all "
      + "available.",
  },
  {
    value: "no-network",
    label: "no-network",
    consequence:
      "web_search, fetch_url, http_request and ask_gemini are withheld, and "
      + "MCP servers are not loaded; code execution and deletes still work.",
  },
  {
    value: "no-delete",
    label: "no-delete",
    consequence:
      "Deletion verbs in shell commands and Python are refused; everything "
      + "else still works.",
  },
  {
    value: "confined",
    label: "confined — no code execution",
    consequence:
      "run_shell, run_python and start_job are withheld, MCP servers are not "
      + "loaded, and absolute paths are refused so the run stays inside its "
      + "workspace.",
  },
];

/** The server's own default (runs.create) — an unknown name becomes this too. */
export const DEFAULT_PROFILE = "all";

/** The sentence for a profile value, or "" for one this build does not know
 *  (a newer server) — never a guess about what an unknown name permits. */
export function profileSentence(value: string): string {
  return PROFILES.find((p) => p.value === value)?.consequence ?? "";
}
