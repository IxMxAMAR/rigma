"use strict";
/* app.js — chat core. Server-authoritative: state here is a thin cache;
   every mutation goes through the API (store.js) and re-renders from truth. */
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("in"), form = $("f"), send = $("send"),
      tps = $("tps");
let current = null;        // full session object, server-authoritative
let defaultPrompt = "";
let presetList = [];
let streaming = false;
let turn = null;           // {abort} handle for the in-flight stream
let lastMeta = null;

/* ---------- header status ---------- */
async function loadStatus() {
  try {
    const s = await api("GET", "/api/status");
    $("model").textContent = s.model + " (" + s.quant + ")";
    defaultPrompt = s.default_system_prompt || "";
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
      if (current && current.id === s.id) { current = null; renderMessages(); }
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
  input.focus();
}

async function openSession(id) {
  current = await api("GET", "/api/sessions/" + id);
  document.body.classList.remove("rail-open");
  lastMeta = lastMeta ? {ctx: lastMeta.ctx} : null;
  $("ctx-bar").classList.remove("live");
  renderAll();
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

function addMsg(cls, content) {
  const e = $("empty");
  if (e) e.remove();
  const d = document.createElement("div");
  d.className = "msg " + cls;
  if (cls === "user") { d.textContent = content; }
  else {
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
    await navigator.clipboard.writeText(m.content || "");
    e.target.textContent = "copied";
    setTimeout(() => { e.target.textContent = "copy"; }, 1200);
  }));
  row.appendChild(actionBtn("edit", () => editMessage(el, m, idx)));
  row.appendChild(actionBtn("delete", () => deleteMessage(idx)));
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
    const el = addMsg(m.role === "user" ? "user" : "bot", m.content);
    addActions(el, m, idx, idx === msgs.length - 1);
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
  ta.value = m.content || "";
  el.replaceChildren(ta);
  ta.focus();
  ta.onkeydown = async (ev) => {
    if (ev.key === "Escape") { renderMessages(); return; }
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      const v = ta.value.trim();
      if (!v) { renderMessages(); return; }
      const msgs = current.messages.slice();
      msgs[idx] = Object.assign({}, msgs[idx], {content: v});
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

async function flipVariant(idx, dir) {
  if (streaming || !current) return;
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

/* ---------- context meter ---------- */
function renderCtxBar() {
  const bar = $("ctx-bar"), fill = $("ctx-fill");
  if (!lastMeta || !lastMeta.ctx || !lastMeta.prompt_tokens) return;
  const frac = Math.min(1, lastMeta.prompt_tokens / lastMeta.ctx);
  bar.classList.add("live");
  bar.title = lastMeta.prompt_tokens.toLocaleString() + " / " +
              lastMeta.ctx.toLocaleString() + " tokens";
  fill.style.width = (frac * 100).toFixed(1) + "%";
  fill.className = frac > 0.9 ? "hot" : frac > 0.75 ? "warn" : "";
}

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
  const payload = Object.assign({message}, opts || {});
  turn = streamTurn(current.id, payload, {
    delta(d) {
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
      body.textContent = d.message;
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
      await refreshSession();
      log.style.scrollBehavior = "";
      // errors aren't persisted server-side — a re-render would erase the
      // error bubble, so keep the DOM as-is and only refresh the rail
      if (!errored) {
        renderMessages();
        if (cites && cites.length) addCitations(cites);
        if (!aborted && pv) await mergeVariants(pv);
        if (aborted) {
          const mark = document.createElement("div");
          mark.className = "cites";
          mark.textContent = "⏹ stopped";
          log.appendChild(mark);
        }
      }
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
  while (msgs.length && msgs[msgs.length - 1].role === "assistant")
    old = msgs.pop();
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
$("sys-edit").addEventListener("blur", async () => {
  if (!current) return;
  const v = $("sys-edit").value.trim();
  if (v === (current.system_prompt || "")) return;
  current = await api("POST", "/api/sessions/" + current.id, {system_prompt: v});
  renderSysBar();
});
$("notes-edit").addEventListener("blur", async () => {
  if (!current) return;
  const v = $("notes-edit").value.trim();
  if (v === (current.notes || "")) return;
  current = await api("POST", "/api/sessions/" + current.id, {notes: v});
  renderSysBar();
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

/* ---------- rail toggle (mobile) ---------- */
$("rail-toggle").onclick = () => document.body.classList.toggle("rail-open");
$("rail-scrim").onclick = () => document.body.classList.remove("rail-open");

/* ---------- composer ---------- */
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
form.onsubmit = async (e) => {
  e.preventDefault();
  if (streaming) { stopTurn(); return; }   // Send morphs into Stop mid-turn
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  if (!current) {
    setStreaming(true);
    try { current = await api("POST", "/api/sessions", {}); }
    catch (err) { input.value = q; return; }
    finally { setStreaming(false); }
  }
  chatTurn(q);
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
})();
