// Slash commands for the chat box.
//
// WHY RIGMA NEEDS ITS OWN, rather than passing them through. DSH ships three
// slash commands — `/goal`, `/compact`, `/feedback` — and Rigma can reach NONE
// of them, whatever it types:
//
//   `parseCommand` (packages/interaction/commands/src/index.ts:125) recognises
//   `/^\/[a-z][a-z0-9_-]*/`, and its only caller is `CommandRuntime.execute`
//   (`:367`), which is decorated **`@Remote`** — a web-layer remote service. The
//   SDK stdio wire that Rigma drives exposes exactly `initialize`,
//   `session/prompt` and `shutdown` (HarnessSdkRequestMap), so there is no
//   method to invoke a command with.
//
// So this is a Rigma feature, not a harness bridge, and it must be honest about
// that: every command below is backed by something Rigma itself can actually do.
// None of them pretend to be DSH's.
//
// This module is pure so the parsing and resolution rules — which are where the
// bugs live — are testable without a DOM.

export interface SlashCommand {
  /** Without the leading slash. */
  name: string;
  /** One line, shown in the menu and by `/help`. */
  summary: string;
  /** What the command needs beyond its own name. */
  takesArgs: boolean;
  /** Shown in the menu when the command needs args. */
  argHint?: string;
}

/** The roster, in menu order. Every one is backed by a real Rigma capability. */
export const COMMANDS: SlashCommand[] = [
  { name: "compact", summary: "summarise this chat to free context", takesArgs: false },
  { name: "new", summary: "start a new chat", takesArgs: false },
  { name: "stop", summary: "stop the reply in progress", takesArgs: false },
  { name: "skills", summary: "open the skills surface", takesArgs: false },
  {
    name: "permission",
    summary: "set what this chat may do",
    takesArgs: true,
    argHint: "off | smart | full",
  },
  {
    // R6-EXPORT: the route has existed all along (`GET /api/sessions/{sid}/export`)
    // and the per-session rail in ChatSurface already links it — but there was no way
    // to reach it FROM THE COMPOSER, which is where someone finishing a conversation
    // actually is. No backend work: this wraps a capability Rigma already had.
    name: "export",
    summary: "download this chat as markdown",
    takesArgs: false,
  },
  { name: "help", summary: "list these commands", takesArgs: false },
];

/** B6c: what to send to RUN a command the BACKEND advertised.
 *
 *  ACP's `available_commands_update` names commands mcode itself resolves, and
 *  the invocation is a `session/prompt` whose text is the command. Rigma's
 *  composer cannot be the path: `parseSlash` above knows only Rigma's own
 *  roster, so an advertised name that collides with one of them (`/compact` is
 *  the real case — mcode advertises its own) would be intercepted and Rigma's
 *  compaction would run instead of the backend's command. The caller therefore
 *  hands this string straight to `send()`, which bypasses the parser by design.
 *
 *  Pure, so the one decision here — the exact text sent — is testable without a
 *  renderer or a store. The name is used verbatim: it came from the server, and
 *  a second vocabulary invented here is a second thing to keep in step. */
export function commandPrompt(name: string): string {
  return `/${name}`;
}

/** `{name, args}` for a line that IS a command, else null.
 *
 *  Deliberately strict, and the strictness is the feature. DSH's own regex is
 *  `/^\/([a-z][a-z0-9_-]*)(?=$|[\t\n\r ])/` — the lookahead means `/compact` is
 *  a command but `/compaction is slow` is NOT, and neither is a path like
 *  `/usr/bin/python`. Getting that wrong would eat a legitimate message, which
 *  is the worst thing a command box can do: silently swallowing a prompt is
 *  indistinguishable from the app being broken.
 *
 *  So: the name must be known, and the character after it must be whitespace or
 *  end-of-line. A leading slash on anything else is just text.
 */
export function parseSlash(line: string): { name: string; args: string } | null {
  const m = /^\/([a-z][a-z0-9_-]*)(?=$|[\t\n\r ])/u.exec(line);
  if (m === null) return null;
  const name = m[1];
  if (!COMMANDS.some((c) => c.name === name)) return null;
  return { name, args: line.slice(m[0].length).trim() };
}

/** Is the box holding a command NAME and nothing else?
 *
 *  True while the user is typing the name AND when it is exactly complete —
 *  `/`, `/com`, `/compact` are all menu states. The menu closes only when a
 *  SPACE is typed, which is the moment the user has committed to the command and
 *  moved on to its argument. Closing at `/compact` instead would take the menu
 *  away one keystroke before Enter is pressed, which is exactly when the user is
 *  looking at it.
 */
export function commandQuery(line: string): string | null {
  const m = /^\/([a-z][a-z0-9_-]*)?$/u.exec(line);
  return m === null ? null : (m[1] ?? "");
}

/** The commands a partially typed name matches, in roster order. */
export function matching(query: string): SlashCommand[] {
  const q = query.toLowerCase();
  return COMMANDS.filter((c) => c.name.startsWith(q));
}

/** What to do about a parsed command. Pure: the caller performs the effect, so
 *  the DECISION is testable and the side effect is one small switch. */
export type CommandPlan =
  | { kind: "run"; name: string; args: string }
  | { kind: "usage"; name: string; message: string };

/** Resolve a parsed command into a plan, refusing one that needs args and has
 *  none — with a message that says which args, rather than failing silently. */
export function planFor(name: string, args: string): CommandPlan {
  const cmd = COMMANDS.find((c) => c.name === name);
  if (!cmd) return { kind: "usage", name, message: `no such command: /${name}` };
  if (cmd.takesArgs && !args) {
    return {
      kind: "usage",
      name,
      // Naming the acceptable values is the whole value of the message: "usage:
      // /permission" leaves the user guessing what to type next.
      message: `/${name} needs an argument — ${cmd.argHint ?? "see /help"}`,
    };
  }
  return { kind: "run", name, args };
}

/** The `/help` body. Built from the roster so it cannot drift from the menu. */
export function helpText(): string {
  return COMMANDS.map(
    (c) => `/${c.name}${c.takesArgs ? ` <${c.argHint ?? "args"}>` : ""} — ${c.summary}`,
  ).join("\n");
}

/** The permission modes `/permission` accepts. Kept here rather than imported
 *  from the store so the parser can refuse a bad value before any request is
 *  made — the server validates too, and refusing early is the better error. */
export const PERMISSION_MODES = ["off", "smart", "full"] as const;

/** R6-ACP-REACH: is this chat configured into a combination that CANNOT work?
 *
 *  `smart` means "ask before risky actions". On mcode's `exec` wire there is nobody to
 *  ask: mcode raises the request, the turn fails, and its own guard then refuses to
 *  start ANY later session in that chat — so the chat is blocked PERMANENTLY, and
 *  retrying makes it fail again in seconds. That is not a risk to warn about, it is a
 *  broken pairing, and it lived only in a tooltip until now.
 *
 *  Returns the sentence to show, or "" for every combination that works. A STRING
 *  rather than a boolean so the reason travels with the decision and cannot drift from
 *  it — and so this is testable without a renderer, which is how this project covers
 *  its UI.
 *
 *  Only mcode: `exec`/`acp` are mcode's two wires, so the same pairing on another
 *  backend would be a warning about nothing.
 */
export function permissionTrap(
  harness: string, permission: string, transport: string,
): string {
  if (harness !== "mcode") return "";
  if (permission !== "smart") return "";
  if (transport !== "exec") return "";
  return "smart asks, and exec has nobody to answer \u2014 so the first time it asks, "
    + "this chat is blocked permanently and every later turn fails immediately. "
    + "Switch transport to acp to make it answerable, or choose full.";
}

export function isPermissionMode(v: string): boolean {
  return (PERMISSION_MODES as readonly string[]).includes(v);
}
