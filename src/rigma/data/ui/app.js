"use strict";
/* app.js — chat core. Server-authoritative: state here is a thin cache;
   every mutation goes through the API (store.js) and re-renders from truth. */
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("in"), form = $("f"), send = $("send"),
      tps = $("tps");
let current = null;        // full session object, server-authoritative
let defaultPrompt = "";
let presetList = [];
let modelCaps = [];        // capabilities of the running model
let streaming = false;
let turn = null;           // {abort} handle for the in-flight stream
let lastMeta = null;

/* ---------- header status ---------- */
async function loadStatus() {
  try {
    const s = await api("GET", "/api/status");
    $("model").textContent = s.model + " (" + s.quant + ")";
    defaultPrompt = s.default_system_prompt || "";
    modelCaps = s.capabilities || [];
    $("effort-toggle").hidden = !modelCaps.includes("thinking");
    if (s.ctx) lastMeta = Object.assign({}, lastMeta, {ctx: s.ctx});
  } catch { $("model").textContent = "server not running"; }
  renderSysBar();
}

/* ---------- presets ---------- */
async function loadPresets() {
  try { presetList = await api("GET", "/api/presets"); } catch { presetList = []; }
  const pick = $("preset-pick");
  pick.innerHTML = '<option value="">no preset</option>';
  for (const p of presetList) {
    const o = document.createElement("option");
    o.value = p.id;
    o.textContent = p.name;
    pick.appendChild(o);
  }
  if (current) pick.value = current.preset_id || "";
}
$("preset-pick").addEventListener("change", async (e) => {
  if (!current) { e.target.value = ""; return; }
  try {
    current = await api("POST", "/api/sessions/" + current.id,
                        {preset_id: e.target.value});
    renderSysBar();
  } catch {
    e.target.value = (current && current.preset_id) || "";
  }
});

/* ---------- session rail ---------- */
async function renderRail() {
  const list = await api("GET", "/api/sessions");
  const nav = $("chat-list");
  nav.innerHTML = "";
  if (!list.length) {
    const d = document.createElement("div");
    d.className = "rail-empty";
    d.textContent = "No chats yet.";
    nav.appendChild(d);
    return;
  }
  for (const s of list) {
    const item = document.createElement("div");
    item.className = "chat-item" + (current && s.id === current.id ? " active" : "");
    item.setAttribute("role", "button");
    item.tabIndex = 0;
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = s.title || "(untitled)";
    title.title = "double-click to rename — id " + s.id;
    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "✕";
    del.setAttribute("aria-label", "Delete chat");
    item.append(title, del);
    item.onclick = () => openSession(s.id);
    item.onkeydown = (e) => { if (e.key === "Enter") openSession(s.id); };
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm('Delete "' + (s.title || s.id) + '"?')) return;
      await api("DELETE", "/api/sessions/" + s.id);
      if (current && current.id === s.id) { current = null; renderAll(); }
      renderRail();
    };
    title.ondblclick = (e) => {
      e.stopPropagation();
      title.contentEditable = "true";
      title.focus();
      document.getSelection().selectAllChildren(title);
      let cancelled = false;
      const commit = async () => {
        title.contentEditable = "false";
        if (cancelled) { title.textContent = s.title || "(untitled)"; return; }
        const t = title.textContent.trim() || "(untitled)";
        await api("POST", "/api/sessions/" + s.id, {title: t});
        renderRail();
      };
      title.onblur = commit;
      title.onkeydown = (ev) => {
        if (ev.key === "Enter") { ev.preventDefault(); title.blur(); }
        if (ev.key === "Escape") { cancelled = true; title.blur(); }
      };
    };
    nav.appendChild(item);
  }
}

async function newChat() {
  current = await api("POST", "/api/sessions", {});
  lastMeta = lastMeta ? {ctx: lastMeta.ctx} : null;
  $("ctx-bar").classList.remove("live");
  renderAll();
  if (typeof refreshDrawer === "function") refreshDrawer();
  input.focus();
}

async function openSession(id) {
  current = await api("GET", "/api/sessions/" + id);
  document.body.classList.remove("rail-open");
  lastMeta = lastMeta ? {ctx: lastMeta.ctx} : null;
  $("ctx-bar").classList.remove("live");
  renderAll();
  if (typeof refreshDrawer === "function") refreshDrawer();
  input.focus();
}

async function refreshSession() {
  if (!current) return;
  try { current = await api("GET", "/api/sessions/" + current.id); }
  catch {}
}

/* ---------- transcript ---------- */
function decorateCode(body) {
  for (const pre of body.querySelectorAll("pre")) {
    if (pre.querySelector(".code-copy")) continue;
    const b = document.createElement("button");
    b.className = "code-copy";
    b.type = "button";
    b.textContent = "copy";
    b.onclick = async () => {
      const code = pre.querySelector("code");
      await navigator.clipboard.writeText(code ? code.textContent : "");
      b.textContent = "copied";
      setTimeout(() => { b.textContent = "copy"; }, 1200);
    };
    pre.appendChild(b);
  }
}

function textOf(content) {
  if (typeof content === "string") return content;
  return (content || []).filter((p) => p && p.type === "text")
    .map((p) => p.text || "").join("\n");
}

function renderUserContent(el, content) {
  if (typeof content === "string") { el.textContent = content; return; }
  for (const p of content || []) {
    if (p.type === "text") {
      const t = document.createElement("div");
      t.textContent = p.text || "";
      el.appendChild(t);
    } else if (p.type === "image_url") {
      const im = document.createElement("img");
      im.className = "msg-img";
      im.src = (p.image_url || {}).url || "";
      im.alt = "attached image";
      el.appendChild(im);
    }
  }
}

function addMsg(cls, content, thinking) {
  const e = $("empty");
  if (e) e.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  if (cls === "user") { renderUserContent(d, content); }
  else {
    if (thinking) d.appendChild(makeThinkBlock(thinking, true));
    const b = document.createElement("div");
    b.className = "body";
    b.innerHTML = renderMarkdown(content);
    decorateCode(b);
    d.appendChild(b);
  }
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

function makeThinkBlock(text, collapsed) {
  const wrap = document.createElement("div");
  wrap.className = "think" + (collapsed ? " closed" : "");
  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = "thinking " + (collapsed ? "▸" : "▾");
  const body = document.createElement("div");
  body.className = "tbody";
  body.textContent = text;
  btn.onclick = () => {
    wrap.classList.toggle("closed");
    btn.textContent = "thinking " +
      (wrap.classList.contains("closed") ? "▸" : "▾");
  };
  wrap.append(btn, body);
  return wrap;
}

/* ---------- tool-call trace (agentic tools) ---------- */
function makeToolBox() {
  const box = document.createElement("div");
  box.className = "tool-trace";
  return box;
}
function addToolChip(box, name, args) {
  const row = document.createElement("details");
  row.className = "tool-chip running";
  const sum = document.createElement("summary");
  const a = args && Object.keys(args).length
    ? " " + Object.values(args).map((v) => String(v)).join(", ").slice(0, 60)
    : "";
  sum.innerHTML = '<span class="tc-spin">⟳</span> <b>' + name + "</b>"
    + '<span class="tc-arg">' + escapeHtml(a) + "</span>";
  const out = document.createElement("div");
  out.className = "tc-out";
  row.append(sum, out);
  box.appendChild(row);
  return row;
}
function setToolResult(row, result) {
  row.classList.remove("running");
  row.classList.add("done");
  row.querySelector(".tc-spin").textContent = "✓";
  row.querySelector(".tc-out").textContent = result || "(no output)";
}
function renderToolTrace(trace) {
  const box = makeToolBox();
  for (const t of trace) {
    const row = addToolChip(box, t.name, t.args);
    setToolResult(row, t.result);
  }
  return box;
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function actionBtn(label, fn, cls) {
  const b = document.createElement("button");
  b.type = "button";
  if (cls) b.className = cls;
  b.textContent = label;
  b.onclick = fn;
  return b;
}

function addActions(el, m, idx, isLast) {
  const row = document.createElement("div");
  row.className = "actions" + (m.role === "user" ? " for-user" : "");
  row.appendChild(actionBtn("copy", async (e) => {
    await navigator.clipboard.writeText(textOf(m.content) || "");
    e.target.textContent = "copied";
    setTimeout(() => { e.target.textContent = "copy"; }, 1200);
  }));
  row.appendChild(actionBtn("edit", () => editMessage(el, m, idx)));
  row.appendChild(actionBtn("delete", () => deleteMessage(idx)));
  row.appendChild(actionBtn("branch", () => branchFrom(idx)));
  if (m.role === "assistant" && Array.isArray(m.variants) && m.variants.length) {
    row.appendChild(actionBtn("◀", () => flipVariant(idx, -1), "flip"));
    const n = document.createElement("span");
    n.className = "flip";
    n.style.font = "11.5px var(--mono)";
    n.textContent = (m.variants.length + 1) + " takes";
    row.appendChild(n);
    row.appendChild(actionBtn("▶", () => flipVariant(idx, 1), "flip"));
  }
  if (m.role === "assistant" && isLast) {
    row.appendChild(actionBtn("regenerate", regenerate));
    if (!(current && current.use_rag))
      row.appendChild(actionBtn("continue", continueTurn));
  }
  log.appendChild(row);
}

function addCitations(cites) {
  const box = document.createElement("div");
  box.className = "cites";
  const btn = actionBtn("▸ " + cites.length + " citation" +
                        (cites.length > 1 ? "s" : ""),
                        () => box.classList.toggle("open"));
  const ul = document.createElement("ul");
  for (const c of cites) {
    const li = document.createElement("li");
    li.textContent = typeof c === "string" ? c : (c.source || JSON.stringify(c));
    ul.appendChild(li);
  }
  box.append(btn, ul);
  log.appendChild(box);
}

function renderMessages() {
  log.innerHTML = "";
  if (!current || !current.messages.length) {
    const d = document.createElement("div");
    d.className = "empty";
    d.id = "empty";
    d.innerHTML = "<b>Talk to your model.</b> Runs on your GPU. " +
                  "Nothing leaves this machine.";
    log.appendChild(d);
    return;
  }
  const msgs = current.messages;
  msgs.forEach((m, idx) => {
    const el = addMsg(m.role === "user" ? "user" : "bot", m.content,
                      m.thinking);
    if (m.tool_trace && m.tool_trace.length)
      el.insertBefore(renderToolTrace(m.tool_trace), el.querySelector(".body"));
    addActions(el, m, idx, idx === msgs.length - 1);
    if (m.citations && m.citations.length) addCitations(m.citations);
  });
  log.scrollTop = log.scrollHeight;
}

/* ---------- message-level operations ---------- */
async function saveMessages(msgs) {
  current = await api("POST", "/api/sessions/" + current.id, {messages: msgs});
}

function editMessage(el, m, idx) {
  if (streaming) return;
  const ta = document.createElement("textarea");
  ta.className = "edit";
  ta.value = textOf(m.content) || "";
  el.replaceChildren(ta);
  ta.focus();
  ta.onkeydown = async (ev) => {
    if (ev.key === "Escape") { renderMessages(); return; }
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      const v = ta.value.trim();
      if (!v) { renderMessages(); return; }
      const msgs = current.messages.slice();
      const imgs = Array.isArray(m.content)
        ? m.content.filter((p) => p && p.type === "image_url") : [];
      msgs[idx] = Object.assign({}, msgs[idx], {content:
        imgs.length ? [{type: "text", text: v}].concat(imgs) : v});
      if (m.role === "user") {
        await saveMessages(msgs.slice(0, idx + 1));  // edits invalidate downstream
        renderMessages();
        chatTurn(null);
      } else {
        await saveMessages(msgs);
        renderMessages();
      }
    }
  };
  ta.onblur = () => { if (document.body.contains(ta)) renderMessages(); };
}

async function deleteMessage(idx) {
  if (streaming || !current) return;
  const msgs = current.messages.slice();
  msgs.splice(idx, 1);
  await saveMessages(msgs);
  renderMessages();
}

async function branchFrom(idx) {
  // fork the chat at this message: duplicate, truncate the copy, open it
  if (streaming || !current) return;
  let d = null;
  try { d = await api("POST", "/api/sessions/" + current.id + "/duplicate"); }
  catch { return; }
  try {
    await api("POST", "/api/sessions/" + d.id,
              {messages: d.messages.slice(0, idx + 1),
               title: (current.title || "chat") + " (branch)"});
  } catch {
    await api("DELETE", "/api/sessions/" + d.id).catch(() => {});
    return;
  }
  await openSession(d.id);
}

let flipping = false;
async function flipVariant(idx, dir) {
  if (streaming || !current || flipping) return;   // debounce rapid clicks
  flipping = true;
  try { await _flipVariant(idx, dir); } finally { flipping = false; }
}
async function _flipVariant(idx, dir) {
  const msgs = current.messages.slice();
  const m = Object.assign({}, msgs[idx]);
  const pool = [m.content].concat(m.variants || []);  // all takes, current first
  const rot = dir > 0 ? pool.slice(1).concat([pool[0]])
                      : [pool[pool.length - 1]].concat(pool.slice(0, -1));
  m.content = rot[0];
  m.variants = rot.slice(1);
  msgs[idx] = m;
  await saveMessages(msgs);
  renderMessages();
}

/* ---------- context meter + compact ---------- */
function renderCtxBar() {
  const bar = $("ctx-bar"), fill = $("ctx-fill");
  if (!lastMeta || !lastMeta.ctx || !lastMeta.prompt_tokens) {
    $("ctx-compact").hidden = true;
    return;
  }
  const frac = Math.min(1, lastMeta.prompt_tokens / lastMeta.ctx);
  bar.classList.add("live");
  bar.title = lastMeta.prompt_tokens.toLocaleString() + " / " +
              lastMeta.ctx.toLocaleString() + " tokens";
  fill.style.width = (frac * 100).toFixed(1) + "%";
  fill.className = frac > 0.9 ? "hot" : frac > 0.75 ? "warn" : "";
  $("ctx-compact").hidden =
    !(frac > 0.75 && current && current.messages.length > 8);
}

async function compactChat(keep) {
  if (!current || streaming) return;
  const overlay = $("switching");
  overlay.firstElementChild.textContent = "compacting…";
  overlay.hidden = false;
  try {
    const out = await api("POST", "/api/sessions/" + current.id + "/compact",
                          {keep: keep || 6});
    current = out.session;
    if (lastMeta) { lastMeta = {ctx: lastMeta.ctx}; }   // meter now stale
    $("ctx-bar").classList.remove("live");
    $("ctx-compact").hidden = true;
    renderMessages();
    renderRail();
  } catch (err) {
    alert("Compact failed: " + err.message);
  } finally {
    overlay.hidden = true;
    overlay.firstElementChild.textContent = "switching model…";
  }
}
$("ctx-compact").onclick = () => compactChat(6);

/* ---------- chat turn ---------- */
function setStreaming(on) {
  streaming = on;
  send.textContent = on ? "Stop" : "Send";
  send.classList.toggle("stop", on);
}

function chatTurn(message, opts) {
  if (!current || streaming) return;
  setStreaming(true);
  if (message !== null) addMsg("user", message);
  const bot = addMsg("bot", "");
  bot.classList.add("streaming");
  const body = bot.querySelector(".body");
  const t0 = performance.now();
  let ntok = 0, text = "", cites = null, errored = false;
  body.innerHTML = '<span class="pending">thinking…</span>';
  log.style.scrollBehavior = "auto";
  const pendingTimer = setInterval(() => {
    const p = body.querySelector(".pending");
    if (p) p.textContent =
      "thinking… " + Math.round((performance.now() - t0) / 1000) + "s";
  }, 1000);
  let thinkEl = null, thinkText = "";
  let toolBox = null, lastTool = null;
  const payload = Object.assign({message}, opts || {});
  turn = streamTurn(current.id, payload, {
    tool(d) {
      if (!toolBox) { toolBox = makeToolBox(); bot.insertBefore(toolBox, body); }
      const p = body.querySelector(".pending"); if (p) p.remove();
      lastTool = addToolChip(toolBox, d.name, d.args);
      log.scrollTop = log.scrollHeight;
    },
    toolResult(d) {
      if (lastTool) setToolResult(lastTool, d.result);
      log.scrollTop = log.scrollHeight;
    },
    think(d) {
      thinkText += d;
      if (!thinkEl) {
        thinkEl = makeThinkBlock("", false);
        bot.insertBefore(thinkEl, body);
      }
      thinkEl.querySelector(".tbody").textContent = thinkText;
      log.scrollTop = log.scrollHeight;
    },
    delta(d) {
      if (thinkEl && !thinkEl.classList.contains("closed") && text === "")
        thinkEl.querySelector("button").click();   // collapse once reply starts
      text += d;
      ntok++;
      body.innerHTML = renderMarkdown(text);
      decorateCode(body);
      log.scrollTop = log.scrollHeight;
      tps.textContent =
        (ntok / ((performance.now() - t0) / 1000)).toFixed(1) + " tok/s";
    },
    error(d) {
      errored = true;
      bot.classList.add("error");
      // keep any partial reply; append the error under it, don't overwrite
      const prior = text ? renderMarkdown(text) : "";
      const err = document.createElement("div");
      err.className = "stream-err";
      err.textContent = d.message;
      body.innerHTML = prior;
      body.appendChild(err);
      if (/exceeds the available context size/i.test(d.message || ""))
        showFitAdvisor();   // actionable remedy, right where it hurt
    },
    citations(c) { cites = c; },
    meta(d) {
      lastMeta = d;
      if (d.predicted_per_second)
        tps.textContent = d.predicted_per_second.toFixed(1) + " tok/s";
      renderCtxBar();
    },
    async done(aborted) {
      clearInterval(pendingTimer);
      bot.classList.remove("streaming");
      turn = null;
      const pv = pendingVariant;      // consume exactly once, every path
      pendingVariant = null;
      const restorePv = async () => {
        try {
          await saveMessages(current.messages.slice().concat([
            {role: "assistant", content: pv.content, variants: pv.variants || []},
          ]));
        } catch {}
      };
      if (aborted && text) {
        // user stopped mid-generation: keep what they got
        const msgs = current.messages.slice();
        const cont = payload["continue"];
        const last = msgs[msgs.length - 1];
        if (cont && last && last.role === "assistant") {
          msgs[msgs.length - 1] =
            Object.assign({}, last, {content: (last.content || "") + text});
        } else {
          if (message !== null)
            msgs.push({role: "user", content: message});
          const asst = {role: "assistant", content: text};
          if (pv)
            asst.variants = (pv.variants || []).concat([pv.content]).filter(Boolean);
          msgs.push(asst);
        }
        try { await saveMessages(msgs); } catch {}
      } else if (aborted && !text && pv) {
        // user stopped a regenerate before any token arrived: restore the old reply
        await restorePv();
      } else if (!aborted && errored && pv) {
        // generation errored after a regenerate: restore the replaced reply
        await restorePv();
      }
      setStreaming(false);
      // persist citations onto the saved reply so they survive a re-render
      if (!errored && cites && cites.length) {
        try {
          const cur = await api("GET", "/api/sessions/" + current.id);
          const last = cur.messages[cur.messages.length - 1];
          if (last && last.role === "assistant") {
            last.citations = cites;
            await saveMessages(cur.messages);
          }
        } catch {}
      }
      await refreshSession();
      log.style.scrollBehavior = "";
      // a restored pending-variant (abort/error after regenerate) must show —
      // otherwise the reply vanishes until the next full render
      if (!errored || pv) {
        renderMessages();
        if (!aborted && !errored && pv) await mergeVariants(pv);
        if (aborted) {
          const mark = document.createElement("div");
          mark.className = "cites";
          mark.textContent = "⏹ stopped";
          log.appendChild(mark);
        }
        if (errored) {   // restored the old reply, but note the failure
          const mark = document.createElement("div");
          mark.className = "cites stream-err";
          mark.textContent = "regenerate failed — previous reply kept";
          log.appendChild(mark);
        }
      }
      // else: errored with no pv — keep the error bubble already in the DOM
      renderRail();
      input.focus();
    },
  });
}

function stopTurn() {
  if (turn) turn.abort();
}

function continueTurn() {
  chatTurn(null, {"continue": true});
}

/* ---------- regenerate with variants ---------- */
let pendingVariant = null;   // {content, variants} of the replaced reply

async function regenerate() {
  if (!current || streaming) return;
  setStreaming(true);
  const msgs = current.messages.slice();
  let old = null;
  if (msgs.length && msgs[msgs.length - 1].role === "assistant")
    old = msgs.pop();   // exactly ONE reply — popping more would lose them
  if (!msgs.length) { setStreaming(false); return; }
  try {
    await saveMessages(msgs);
    renderMessages();
    if (old) pendingVariant = {content: old.content,
                               variants: old.variants || []};
  } finally {
    setStreaming(false);        // released synchronously before chatTurn re-locks
  }
  chatTurn(null);
}

async function mergeVariants(old) {
  // after a regenerate completes, fold the replaced reply into variants
  const msgs = current.messages.slice();
  const last = msgs[msgs.length - 1];
  if (!last || last.role !== "assistant") return;
  msgs[msgs.length - 1] = Object.assign({}, last, {
    variants: (last.variants || []).concat(old.variants || [], [old.content])
      .filter(Boolean),
  });
  try {
    await saveMessages(msgs);
    renderMessages();
  } catch {}
}

/* ---------- system prompt + notes bars ---------- */
function renderSysBar() {
  const active = current && current.system_prompt;
  const preset = current && current.preset_id
    ? presetList.find((p) => p.id === current.preset_id) : null;
  $("sys-preview").textContent = active ? current.system_prompt
    : preset ? "preset: " + preset.name
    : defaultPrompt ? "default: " + defaultPrompt : "no system prompt";
  $("sys-edit").value = active ? current.system_prompt : "";
  $("sys-edit").placeholder = preset || defaultPrompt
    ? "Blank = preset/default (shown above)" : "System prompt for this chat";
  $("preset-pick").value = (current && current.preset_id) || "";
  $("effort-toggle").textContent = effortLabel((current && current.effort) || "");
  $("notes-edit").value = (current && current.notes) || "";
  $("notes-toggle").style.color =
    current && current.notes ? "var(--moss)" : "";
}
function wireToggle(btnId, areaId, label) {
  $(btnId).onclick = () => {
    const ed = $(areaId);
    ed.hidden = !ed.hidden;
    $(btnId).textContent = label + (ed.hidden ? " ▸" : " ▾");
    $(btnId).setAttribute("aria-expanded", String(!ed.hidden));
    if (!ed.hidden) ed.focus();
  };
}
wireToggle("sys-toggle", "sys-edit", "sys");
wireToggle("notes-toggle", "notes-edit", "notes");
const EFFORTS = ["", "off", "on"];
function effortLabel(v) { return "effort ▸ " + (v || "auto"); }
$("effort-toggle").onclick = async () => {
  if (!current) return;
  const cur = current.effort || "";
  const next = EFFORTS[(EFFORTS.indexOf(cur) + 1) % EFFORTS.length];
  try {
    current = await api("POST", "/api/sessions/" + current.id, {effort: next});
    $("effort-toggle").textContent = effortLabel(current.effort);
  } catch {}
};
let fieldSaving = null;   // in-flight sys/notes save, awaited before a turn
$("sys-edit").addEventListener("blur", () => {
  if (!current) return;
  const v = $("sys-edit").value.trim();
  if (v === (current.system_prompt || "")) return;
  fieldSaving = api("POST", "/api/sessions/" + current.id,
                    {system_prompt: v}).then((s) => { current = s;
                                                      renderSysBar(); });
});
$("notes-edit").addEventListener("blur", () => {
  if (!current) return;
  const v = $("notes-edit").value.trim();
  if (v === (current.notes || "")) return;
  fieldSaving = api("POST", "/api/sessions/" + current.id,
                    {notes: v}).then((s) => { current = s; renderSysBar(); });
});
for (const id of ["sys-edit", "notes-edit"]) {
  $(id).addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.preventDefault(); $(id).blur(); }
  });
}

/* ---------- RAG toggle + docs panel (rail) ---------- */
$("use-rag").addEventListener("change", async (e) => {
  if (!current) { e.target.checked = false; return; }
  current = await api("POST", "/api/sessions/" + current.id,
                      {use_rag: e.target.checked});
  document.body.classList.toggle("rag-on", current.use_rag);
  refreshDocs();
});

let docsTimer = null;
async function refreshDocs() {
  let s;
  try { s = await api("GET", "/api/rag/status"); } catch { return; }
  $("rag-dot").classList.toggle("on", s.running);
  const ul = $("source-list");
  ul.innerHTML = "";
  for (const src of s.sources) {
    const li = document.createElement("li");
    li.textContent = src;
    li.title = src;
    ul.appendChild(li);
  }
  $("docs-status").textContent = s.indexing ? "indexing…"
    : s.error ? "index failed: " + s.error
    : s.running ? "sidecar running"
    : s.sources.length ? "sidecar starts on first grounded message" : "";
  clearTimeout(docsTimer);
  if (s.indexing) docsTimer = setTimeout(refreshDocs, 2000);
}
$("docs-toggle").onclick = () => {
  const b = $("docs-body");
  b.hidden = !b.hidden;
  $("docs-toggle").setAttribute("aria-expanded", String(!b.hidden));
  if (!b.hidden) refreshDocs();
};
$("add-source").onsubmit = async (e) => {
  e.preventDefault();
  const path = $("source-path").value.trim();
  if (!path) return;
  try {
    await api("POST", "/api/rag/sources", {path});
    $("source-path").value = "";
    refreshDocs();
  } catch (err) { $("docs-status").textContent = err.message; }
};

/* ---------- engine room chip + fit advisor + switch ---------- */
let engineInfo = null;
async function pollEngine() {
  try { engineInfo = await api("GET", "/api/server"); }
  catch { engineInfo = null; }
  const dot = $("engine-dot"), label = $("engine-label");
  if (!engineInfo) { dot.className = "dot"; label.textContent = "engine"; return; }
  if (engineInfo.calibrating) {
    // first-load hardware tune in progress — poll faster so the step updates
    dot.className = "dot warn";
    const step = engineInfo.calibrating.step || "";
    label.textContent = "optimizing" + (step && step !== "starting"
      ? " (" + step + ")" : "…");
    $("model").textContent = "optimizing " + engineInfo.calibrating.model +
      " for your hardware…";
    clearTimeout(pollEngine._t);
    pollEngine._t = setTimeout(pollEngine, 2000);
    return;
  }
  if (engineInfo.unloaded) {
    dot.className = "dot warn";
    label.textContent = "unloaded";
    return;
  }
  const lowRam = engineInfo.ram_free_mb < 1536;
  if (engineInfo.verdict === "degraded") {
    dot.className = "dot bad";
    label.textContent = "decode degraded";
  } else if (lowRam) {
    dot.className = "dot warn";
    label.textContent = "low RAM";
  } else if (engineInfo.verdict === "healthy") {
    dot.className = "dot on";
    label.textContent = "engine ok";
  } else {
    dot.className = "dot";
    label.textContent = "engine";
  }
}
$("engine-chip").onclick = () => toggleDrawer("server");
$("model").onclick = () => toggleDrawer("server");   // model name = switcher door
$("model").title = "Model & engine — click to switch (Server tab)";
setInterval(pollEngine, 15000);

async function showFitAdvisor() {
  let opts = [];
  try { opts = await api("GET", "/api/server/switch-options"); } catch {}
  const cur = (engineInfo && engineInfo.ctx) || (lastMeta && lastMeta.ctx) || 0;
  if (cur) opts = opts.filter((o) => o.ctx > cur);
  const box = document.createElement("div");
  box.className = "advisor";
  const label = document.createElement("span");
  label.textContent = "fit advisor:";
  box.appendChild(label);
  if (!opts.length) {
    const t = document.createElement("span");
    t.className = "dim";
    t.textContent = " no downloaded model fits a bigger context right now — " +
      "trim/delete old messages, branch from an earlier point, start a new " +
      "chat, or free RAM and retry (max_tokens only caps one reply — it " +
      "can't grow the window)";
    box.appendChild(t);
  }
  for (const o of opts.slice(0, 2)) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = "switch to " + o.model + " (" + o.reason + ")";
    b.onclick = () => doSwitch(o.model);
    box.appendChild(b);
  }
  log.appendChild(box);
  log.scrollTop = log.scrollHeight;
}

async function doSwitch(model) {
  if (streaming) return;
  if (!confirm("Switch model to " + model +
               "? The current model will be stopped.")) return;
  $("switching").hidden = false;
  try {
    await api("POST", "/api/server/switch", {model});
    await loadStatus();
    await pollEngine();
  } catch (err) {
    alert("Switch failed: " + err.message);
  } finally {
    $("switching").hidden = true;
  }
}

/* ---------- rail toggle (mobile) ---------- */
$("rail-toggle").onclick = () => document.body.classList.toggle("rail-open");
$("rail-scrim").onclick = () => document.body.classList.remove("rail-open");

/* ---------- composer + images ---------- */
let pendingImages = [];

function downscaleImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, 1344 / Math.max(img.width, img.height));
      const cv = document.createElement("canvas");
      cv.width = Math.round(img.width * scale);
      cv.height = Math.round(img.height * scale);
      cv.getContext("2d").drawImage(img, 0, 0, cv.width, cv.height);
      URL.revokeObjectURL(img.src);
      resolve(cv.toDataURL("image/jpeg", 0.85));
    };
    img.onerror = reject;
    img.src = URL.createObjectURL(file);
  });
}

function renderImgChips() {
  const box = $("img-chips");
  box.innerHTML = "";
  box.hidden = !pendingImages.length;
  pendingImages.forEach((url, i) => {
    const chip = document.createElement("span");
    chip.className = "img-chip";
    const im = document.createElement("img");
    im.src = url;
    const x = document.createElement("button");
    x.type = "button";
    x.textContent = "✕";
    x.onclick = () => { pendingImages.splice(i, 1); renderImgChips(); };
    chip.append(im, x);
    box.appendChild(chip);
  });
}

let imagesProcessing = null;   // resolves when in-flight downscales finish
async function addImageFiles(files) {
  const job = (async () => {
    for (const f of files) {
      if (!f.type.startsWith("image/")) continue;
      try { pendingImages.push(await downscaleImage(f)); } catch {}
    }
    renderImgChips();
  })();
  imagesProcessing = job;
  await job;
  if (imagesProcessing === job) imagesProcessing = null;
}
$("attach").onclick = () => $("img-file").click();
$("img-file").addEventListener("change", (e) => {
  addImageFiles([...e.target.files]);
  e.target.value = "";
});
input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData || {}).files || []];
  if (files.length) { e.preventDefault(); addImageFiles(files); }
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
form.onsubmit = async (e) => {
  e.preventDefault();
  if (streaming) { stopTurn(); return; }   // Send morphs into Stop mid-turn
  // a pasted image may still be downscaling, and a focused sys/notes edit may
  // not have saved — flush both before the turn so nothing is lost or stale
  if (imagesProcessing) { try { await imagesProcessing; } catch {} }
  const ae = document.activeElement;
  if (ae && (ae.id === "sys-edit" || ae.id === "notes-edit")) ae.blur();
  if (fieldSaving) { try { await fieldSaving; } catch {} fieldSaving = null; }
  const q = input.value.trim();
  if (!q && !pendingImages.length) return;
  const message = pendingImages.length
    ? [{type: "text", text: q || "Describe this image."}].concat(
        pendingImages.map((u) => ({type: "image_url", image_url: {url: u}})))
    : q;
  input.value = "";
  pendingImages = [];
  renderImgChips();
  if (!current) {
    setStreaming(true);
    try { current = await api("POST", "/api/sessions", {}); }
    catch (err) { input.value = q; return; }
    finally { setStreaming(false); }
  }
  chatTurn(message);
};
$("new-chat").onclick = newChat;

/* ---------- render all + boot ---------- */
function renderAll() {
  renderMessages();
  renderRail();
  renderSysBar();
  renderCtxBar();
  $("use-rag").checked = !!(current && current.use_rag);
  document.body.classList.toggle("rag-on", !!(current && current.use_rag));
}
(async function boot() {
  await loadStatus();
  await loadPresets();
  const list = await api("GET", "/api/sessions").catch(() => []);
  if (list.length) { await openSession(list[0].id); }
  else { renderAll(); }
  refreshDocs();
  pollEngine();
})();
