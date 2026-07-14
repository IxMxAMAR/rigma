"use strict";
const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("in"), form = $("f"), send = $("send"),
      tps = $("tps"), railEl = $("rail");
let current = null;        // full session object, server-authoritative
let defaultPrompt = "";
let streaming = false;

async function api(method, path, body) {
  const r = await fetch(path, {
    method,
    headers: body !== undefined ? {"content-type": "application/json"} : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.error || ("server replied " + r.status));
  }
  return r.json();
}

/* ---------- header status ---------- */
async function loadStatus() {
  try {
    const s = await api("GET", "/api/status");
    $("model").textContent = s.model + " (" + s.quant + ")";
    defaultPrompt = s.default_system_prompt || "";
  } catch { $("model").textContent = "server not running"; }
  renderSysBar();
}

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
  renderAll();
  input.focus();
}

async function openSession(id) {
  current = await api("GET", "/api/sessions/" + id);
  document.body.classList.remove("rail-open");
  renderAll();
  input.focus();
}

async function refreshSession() {
  if (!current) return;
  try { current = await api("GET", "/api/sessions/" + current.id); }
  catch { current = null; }
}

/* ---------- transcript ---------- */
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
    d.appendChild(b);
  }
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

function addActions(el, opts) {   // opts: {copy, regen, edit}
  const row = document.createElement("div");
  row.className = "actions";
  if (opts.copy) {
    const copy = document.createElement("button");
    copy.textContent = "copy";
    copy.onclick = async () => {
      await navigator.clipboard.writeText(el.dataset.raw || "");
      copy.textContent = "copied";
      setTimeout(() => { copy.textContent = "copy"; }, 1200);
    };
    row.appendChild(copy);
  }
  if (opts.regen) {
    const regen = document.createElement("button");
    regen.textContent = "regenerate";
    regen.onclick = regenerate;
    row.appendChild(regen);
  }
  if (opts.edit) {
    const edit = document.createElement("button");
    edit.textContent = "edit";
    edit.onclick = editLast;
    row.appendChild(edit);
  }
  if (row.children.length) log.appendChild(row);
}

function addCitations(cites) {
  const box = document.createElement("div");
  box.className = "cites";
  const btn = document.createElement("button");
  btn.textContent = "▸ " + cites.length + " citation" + (cites.length > 1 ? "s" : "");
  const ul = document.createElement("ul");
  for (const c of cites) {
    const li = document.createElement("li");
    li.textContent = typeof c === "string" ? c : (c.source || JSON.stringify(c));
    ul.appendChild(li);
  }
  btn.onclick = () => box.classList.toggle("open");
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
    el.dataset.raw = m.content;
    const last = idx === msgs.length - 1;
    if (m.role === "assistant")
      addActions(el, {copy: true, regen: last});
    else if (last)               // trailing user msg (e.g. after an error)
      addActions(el, {edit: true});
  });
  log.scrollTop = log.scrollHeight;
}

/* ---------- chat turn ---------- */
async function chatTurn(message) {
  if (!current || streaming) return;
  streaming = true;
  send.disabled = true;
  if (message !== null) addMsg("user", message);
  const bot = addMsg("bot", "");
  bot.classList.add("streaming");
  const body = bot.querySelector(".body");
  const t0 = performance.now();
  let ntok = 0, text = "", cites = null, errored = false;
  try {
    const r = await fetch("/api/sessions/" + current.id + "/chat", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({message}),
    });
    if (!r.ok) throw new Error("server replied " + r.status);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "", event = "";
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const ln of lines) {
        if (ln.startsWith("event: ")) { event = ln.slice(7).trim(); continue; }
        if (!ln.startsWith("data: ")) continue;
        const payload = ln.slice(6).trim();
        if (payload === "[DONE]") { event = ""; continue; }
        let d;
        try { d = JSON.parse(payload); } catch { event = ""; continue; }
        if (event === "error") {
          errored = true;
          bot.classList.add("error");
          body.textContent = d.message +
            " — check `rigma status` in a terminal.";
        } else if (event === "citations") {
          cites = d.citations;
        } else if (d.delta) {
          text += d.delta;
          ntok++;
          body.innerHTML = renderMarkdown(text);
          log.scrollTop = log.scrollHeight;
          tps.textContent =
            (ntok / ((performance.now() - t0) / 1000)).toFixed(1) + " tok/s";
        }
        event = "";
      }
    }
  } catch (err) {
    errored = true;
    bot.classList.add("error");
    body.textContent = "Couldn't reach the model: " + err.message +
      " — check `rigma status` in a terminal.";
  } finally {
    bot.classList.remove("streaming");
    streaming = false;
    send.disabled = false;
    await refreshSession();   // server truth: title, persisted messages
    // errors aren't persisted server-side — a re-render would erase the
    // error bubble, so keep the DOM as-is and only refresh the rail
    if (!errored) {
      renderMessages();
      if (cites && cites.length) addCitations(cites);  // re-render drops non-message DOM
    }
    renderRail();
    input.focus();
  }
}

async function regenerate() {
  if (!current || streaming) return;
  streaming = true;
  send.disabled = true;
  const msgs = current.messages.slice();
  while (msgs.length && msgs[msgs.length - 1].role === "assistant") msgs.pop();
  if (!msgs.length) { streaming = false; send.disabled = false; return; }
  try {
    current = await api("POST", "/api/sessions/" + current.id, {messages: msgs});
    renderMessages();
  } finally {
    streaming = false;          // released synchronously before chatTurn re-locks
    send.disabled = false;
  }
  chatTurn(null);
}

async function editLast() {
  if (!current || streaming) return;
  streaming = true;
  send.disabled = true;
  try {
    const msgs = current.messages.slice();
    while (msgs.length && msgs[msgs.length - 1].role === "assistant") msgs.pop();
    const last = msgs.pop();
    if (!last) return;
    current = await api("POST", "/api/sessions/" + current.id, {messages: msgs});
    renderMessages();
    input.value = last.content;
  } finally {
    streaming = false;
    send.disabled = false;
    input.focus();
  }
}

/* ---------- system prompt bar ---------- */
function renderSysBar() {
  const active = current && current.system_prompt;
  $("sys-preview").textContent = active ? current.system_prompt
    : (defaultPrompt ? "default: " + defaultPrompt : "no system prompt");
  $("sys-edit").value = active ? current.system_prompt : "";
  $("sys-edit").placeholder = defaultPrompt
    ? "Blank = use-case default (shown above)" : "System prompt for this chat";
}
$("sys-toggle").onclick = () => {
  const ed = $("sys-edit");
  ed.hidden = !ed.hidden;
  $("sys-toggle").textContent = ed.hidden ? "sys ▸" : "sys ▾";
  $("sys-toggle").setAttribute("aria-expanded", String(!ed.hidden));
  if (!ed.hidden) ed.focus();
};
$("sys-edit").addEventListener("blur", async () => {
  if (!current) return;
  const v = $("sys-edit").value.trim();
  if (v === (current.system_prompt || "")) return;
  current = await api("POST", "/api/sessions/" + current.id, {system_prompt: v});
  renderSysBar();
});
$("sys-edit").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.preventDefault(); $("sys-edit").blur(); $("sys-toggle").click(); }
});

/* ---------- RAG toggle + docs panel ---------- */
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
  const q = input.value.trim();
  if (!q || streaming) return;
  input.value = "";
  if (!current) {
    streaming = true;
    send.disabled = true;
    try { current = await api("POST", "/api/sessions", {}); }
    catch (err) { input.value = q; return; }
    finally { streaming = false; send.disabled = false; }
  }
  chatTurn(q);
};
$("new-chat").onclick = newChat;

/* ---------- render all + boot ---------- */
function renderAll() {
  renderMessages();
  renderRail();
  renderSysBar();
  $("use-rag").checked = !!(current && current.use_rag);
  document.body.classList.toggle("rag-on", !!(current && current.use_rag));
}
(async function boot() {
  await loadStatus();
  const list = await api("GET", "/api/sessions").catch(() => []);
  if (list.length) { await openSession(list[0].id); }
  else { renderAll(); }
  refreshDocs();
})();
