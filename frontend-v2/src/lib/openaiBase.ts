// The OpenAI-compatible base URL, from the SERVER.
//
// Settings hardcoded `http://127.0.0.1:11499/v1` and called it "Rigma's OpenAI
// API". 11499 is llama-server (`serve._public_port(11499) = 11500`), so that
// URL talks to the engine directly and bypasses the session, the tool-call
// repair and the idle-unload bookkeeping — and it is simply dead on any
// non-default launch. `/api/server` already publishes the right value as
// `openai_base` (serve.py sets it from the public port); read it, and fall back
// to the origin that served this page, never to a literal port.
export function openaiBase(
  info: { openai_base?: unknown } | null | undefined,
  origin: string,
): string {
  const b = info?.openai_base;
  if (typeof b === "string" && b.trim()) return b.trim();
  return origin.replace(/\/+$/, "") + "/v1";
}
