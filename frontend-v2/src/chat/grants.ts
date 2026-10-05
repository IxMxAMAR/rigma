// The four per-chat tool grants added by AUDIT 13-2 / 13-3 / R3-TOOL-4.
//
// The server gates run_shell / start_job / run_python behind the session field
// `confirm_exec`, absolute reads behind `allow_absolute_reads`, outbound POSTs
// carrying a body behind `allow_outbound_post`, and writes outside the
// workspace behind `allow_absolute_writes` (sessions.MUTABLE_FIELDS, read in
// serve.chat into the tool context). NO session defaults any of them
// and NO surface wrote them, so code execution was silently OFF for every
// existing chat: the owner's own tools were unreachable and nothing on screen
// said why. This module is the single place that names them, so the
// checkbox list cannot drift from the server's fields.
//
// IMP-4. The wording is deliberately a statement of what turning the grant ON
// permits, not a warning label — the honest default is off, and an owner who
// wants their own tools back should not have to feel they are doing something
// reckless to get them.

export type GrantKey =
  | "confirm_exec"
  | "allow_absolute_reads"
  | "allow_outbound_post"
  | "allow_absolute_writes";

export interface GrantDef {
  key: GrantKey;
  /** The checkbox label, phrased as what turning it ON permits. */
  label: string;
  /** One plain sentence about what the grant actually allows. */
  consequence: string;
}

export const GRANTS: readonly GrantDef[] = [
  {
    key: "confirm_exec",
    label: "Allow this chat to run commands",
    consequence:
      "The model may run shell commands and Python on this PC, confined to "
      + "this chat's workspace.",
  },
  {
    key: "allow_absolute_reads",
    label: "Allow reads outside the workspace",
    consequence:
      "The model may read files anywhere on this PC, not only inside this "
      + "chat's workspace.",
  },
  {
    key: "allow_outbound_post",
    label: "Allow sending data out",
    consequence:
      "The model may send data it has read to a web address — the one way "
      + "files can leave this machine.",
  },
  {
    // R3-TOOL-4. Separate from the reads grant on purpose: reading a file and
    // replacing one are different risks, and a user who granted the first did
    // not grant the second.
    key: "allow_absolute_writes",
    label: "Allow writes outside the workspace",
    // MEASURED 2026-10-05: this said "move or copy", and the tools agreed — a
    // `write_file`/`edit_file` target outside the workspace was refused even
    // with the grant on, while `move_files` reached it. The grant now answers
    // for every write the session can make, so the sentence names them.
    consequence:
      "The model may write, move or copy files anywhere on this PC, not only "
      + "inside this chat's workspace.",
  },
];

export type Grants = Record<GrantKey, boolean>;

/** All OFF — which is exactly what the server gives a chat, since it defaults
 *  none of them. This is the state every chat starts in. */
export const NO_GRANTS: Grants = {
  confirm_exec: false,
  allow_absolute_reads: false,
  allow_outbound_post: false,
  allow_absolute_writes: false,
};

/** Read the grant fields off a session body. Anything that is not exactly
 *  `true` reads as OFF: a missing field, a null, a string, or a server that
 *  predates the grants must never be mistaken for a granted permission. */
export function readGrants(s: unknown): Grants {
  const o = (s && typeof s === "object" ? s : {}) as Record<string, unknown>;
  const out: Grants = { ...NO_GRANTS };
  for (const g of GRANTS) out[g.key] = o[g.key] === true;
  return out;
}
