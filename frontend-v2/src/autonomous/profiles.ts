// The autonomous-run safety profiles (IMP-4).
//
// `POST /api/runs` has accepted `profile` all along (runs.PROFILES, applied in
// tools.tool_specs and the call gate), and the legacy UI offered a bare select
// of the four names with no explanation. The v2 launcher offered none, so a run
// always got "all" and the main safety control was invisible. These are the
// names plus one honest sentence each, read from tools.py — not from the name.
//
// R3-4 CORRECTION. Every sentence here used to say "code execution ... still
// works" for the profiles that do not withhold it. That was FALSE for a run:
// `start_run` sets `allow_code=True` but execution also needs `confirm_exec`,
// which defaulted OFF and which no surface could turn on. So "all" promised
// execution it never delivered, and a user choosing `confined` for safety got
// the same exec behaviour as `all` by accident. The sentences now state the
// profile's own contribution only, and the launcher carries a separate, explicit
// "allow code execution" switch — because a profile NARROWS a roster while that
// switch GRANTS a capability, and conflating the two is what produced the false
// sentence.

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
      "Nothing is withheld: the network tools, MCP servers and deletes are all "
      + "available. Code execution is a separate switch below.",
  },
  {
    value: "no-network",
    label: "no-network",
    consequence:
      "web_search, fetch_url, http_request and ask_gemini are withheld, and "
      + "MCP servers are not loaded; file edits and deletes still work.",
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
      + "workspace. Enforced regardless of the execution switch below.",
  },
];

/** The server's own default (runs.create) — an unknown name becomes this too. */
export const DEFAULT_PROFILE = "all";

/** The sentence for a profile value, or "" for one this build does not know
 *  (a newer server) — never a guess about what an unknown name permits. */
export function profileSentence(value: string): string {
  return PROFILES.find((p) => p.value === value)?.consequence ?? "";
}
