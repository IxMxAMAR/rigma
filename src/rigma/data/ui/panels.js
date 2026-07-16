"use strict";
/* panels.js — settings drawer (chat params / presets manager). Loads after
   app.js; talks to the same server-authoritative state via store.js. */

const PARAM_DEFS = [
  ["temperature", 0, 4, 0.05],
  ["top_p", 0, 1, 0.01],
  ["min_p", 0, 1, 0.01],
  ["repeat_penalty", 0.5, 2, 0.01],
  ["max_tokens", 1, 32768, 1],
];
// modern anti-repetition samplers — collapsed under "Advanced sampling"
const PARAM_DEFS_ADV = [
  ["dry_multiplier", 0, 2, 0.05],
  ["dry_base", 1, 4, 0.05],
  ["dry_allowed_length", 1, 10, 1],
  ["xtc_probability", 0, 1, 0.05],
  ["xtc_threshold", 0, 0.5, 0.01],
  ["top_n_sigma", -1, 5, 0.1],
];
const INT_PARAMS = ["max_tokens", "dry_allowed_length"];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/* ---------- drawer shell ---------- */
function toggleDrawer(tab) {
  const d = $("drawer");
  if (!d.hidden && !tab) { d.hidden = true; return; }
  d.hidden = false;
  openTab(tab || "chat");
}
let activeTab = "chat";
function openTab(name) {
  activeTab = name;
  for (const b of document.querySelectorAll("#drawer-tabs button"))
    b.classList.toggle("active", b.dataset.tab === name);
  if (name === "chat") renderChatTab();
  else if (name === "presets") renderPresetsTab();
  else if (name === "server") renderServerTab();
}
function refreshDrawer() {
  if (!$("drawer").hidden) openTab(activeTab);
}
$("drawer-close").onclick = () => { $("drawer").hidden = true; };
$("gear").onclick = () => toggleDrawer();
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("drawer").hidden) $("drawer").hidden = true;
});
for (const b of document.querySelectorAll("#drawer-tabs button"))
  b.onclick = () => openTab(b.dataset.tab);

/* ---------- chat tab: sampler params + export/duplicate ---------- */
let paramTimer = null;
function renderChatTab() {
  const box = $("drawer-body");
  box.innerHTML = "";
  if (!current) { box.appendChild(el("p", "dim", "Open a chat first.")); return; }
  const sid = current.id;
  box.appendChild(el("h3", "", "Sampling — this chat"));
  const hint = el("p", "dim",
    "Blank = engine default (or the preset's value). Applied per request.");
  box.appendChild(hint);
  const params = Object.assign({}, current.params || {});
  const addParamRow = (key, lo, hiDef, step) => {
    let hi = hiDef;
    const row = el("div", "param-row");
    const lbl = el("label", "", key);
    if (key === "max_tokens") {
      // per-reply cap, NOT the context window — and it can never exceed it
      const ctx = (typeof engineInfo === "object" && engineInfo && engineInfo.ctx)
        || (lastMeta && lastMeta.ctx) || 0;
      if (ctx) hi = ctx;
      lbl.title = "Cap on ONE reply's length. The context window (" +
        (ctx ? ctx.toLocaleString() + " tokens" : "engine-fixed") +
        ") is set at engine launch and includes the whole conversation.";
    }
    row.appendChild(lbl);
    const range = el("input");
    range.type = "range";
    range.min = lo; range.max = hi; range.step = step;
    const num = el("input", "val");
    num.type = "number";
    num.min = lo; num.max = hi; num.step = step;
    num.placeholder = "—";
    if (params[key] !== undefined) { range.value = params[key]; num.value = params[key]; }
    else range.value = key === "repeat_penalty" ? 1 : lo;
    const clear = el("button", "clear", "✕");
    clear.title = "Clear (use default)";
    const push = () => {
      clearTimeout(paramTimer);
      paramTimer = setTimeout(async () => {
        if (!current || current.id !== sid) return;   // stale editor: never cross-write
        try {
          current = await api("POST", "/api/sessions/" + sid, {params});
        } catch (err) { hint.textContent = err.message; }
      }, 350);
    };
    range.oninput = () => { num.value = range.value;
      params[key] = INT_PARAMS.includes(key) ? parseInt(range.value, 10)
                                         : parseFloat(range.value); push(); };
    num.oninput = () => {
      if (num.value === "") { delete params[key]; push(); return; }
      range.value = num.value;
      params[key] = INT_PARAMS.includes(key) ? parseInt(num.value, 10)
                                         : parseFloat(num.value); push();
    };
    clear.onclick = () => { num.value = ""; delete params[key]; push(); };
    row.append(range, num, clear);
    box.appendChild(row);
  };
  for (const [key, lo, hi, step] of PARAM_DEFS) addParamRow(key, lo, hi, step);
  box.appendChild(el("h3", "", "Advanced sampling (anti-repetition)"));
  for (const [key, lo, hi, step] of PARAM_DEFS_ADV) addParamRow(key, lo, hi, step);
  box.appendChild(el("h3", "", "Memory"));
  const dg = el("textarea");
  dg.rows = 4;
  dg.placeholder = "Compacted digest of earlier turns (empty = none)";
  dg.value = current.digest || "";
  dg.onblur = async () => {
    if (!current || current.id !== sid) return;
    if (dg.value.trim() === (current.digest || "")) return;
    try {
      current = await api("POST", "/api/sessions/" + sid,
                          {digest: dg.value.trim()});
    } catch (err) { hint.textContent = err.message; }
  };
  box.appendChild(dg);
  const macts = el("div", "drawer-acts");
  const compactBtn = el("button", "act", "Compact now (keep last 6)");
  compactBtn.onclick = () => { $("drawer").hidden = true; compactChat(6); };
  macts.appendChild(compactBtn);
  box.appendChild(macts);

  box.appendChild(el("h3", "", "This chat"));
  const acts = el("div", "drawer-acts");
  const exMd = el("a", "act", "Export markdown");
  exMd.href = "/api/sessions/" + sid + "/export?fmt=md";
  const exJs = el("a", "act", "Export JSON");
  exJs.href = "/api/sessions/" + sid + "/export?fmt=json";
  const dup = el("button", "act", "Duplicate chat");
  dup.onclick = async () => {
    const d = await api("POST", "/api/sessions/" + sid + "/duplicate");
    $("drawer").hidden = true;
    await openSession(d.id);
  };
  acts.append(exMd, exJs, dup);
  box.appendChild(acts);
}

/* ---------- presets tab: manager ---------- */
function renderPresetsTab() {
  const box = $("drawer-body");
  box.innerHTML = "";
  box.appendChild(el("h3", "", "Presets"));
  const list = el("div", "preset-list");
  for (const p of presetList) {
    const row = el("button", "preset-row" + (p.builtin ? " builtin" : ""));
    row.type = "button";
    row.textContent = p.name;
    if (p.builtin) row.appendChild(el("span", "badge", "built-in"));
    row.onclick = () => renderPresetForm(p);
    list.appendChild(row);
  }
  box.appendChild(list);
  const acts = el("div", "drawer-acts");
  const mk = el("button", "act", "New preset");
  mk.onclick = () => renderPresetForm(null);
  acts.appendChild(mk);
  if (current) {
    const fromChat = el("button", "act", "Save this chat's prompt as preset");
    fromChat.onclick = () => renderPresetForm({
      name: (current.title || "chat") + " prompt",
      system_prompt: current.system_prompt ||
        (($("sys-preview").textContent || "").replace(/^default: /, "")),
      greeting: "", params: current.params || {}, _new: true,
    });
    acts.appendChild(fromChat);
  }
  box.appendChild(acts);
}

function renderPresetForm(p) {
  const box = $("drawer-body");
  box.innerHTML = "";
  const isNew = !p || p._new;
  const ro = p && p.builtin;
  box.appendChild(el("h3", "", isNew ? "New preset"
    : (ro ? p.name + " (read-only)" : "Edit preset")));
  const name = el("input");
  name.placeholder = "Name";
  name.value = (p && p.name) || "";
  const sys = el("textarea");
  sys.rows = 6;
  sys.placeholder = "System prompt";
  sys.value = (p && p.system_prompt) || "";
  const greet = el("textarea");
  greet.rows = 3;
  greet.placeholder = "Greeting (optional first assistant message)";
  greet.value = (p && p.greeting) || "";
  for (const f of [name, sys, greet]) { f.disabled = !!ro; box.appendChild(f); }
  const acts = el("div", "drawer-acts");
  if (!ro) {
    const save = el("button", "act", isNew ? "Create" : "Save");
    save.onclick = async () => {
      const body = {name: name.value.trim() || "Preset",
                    system_prompt: sys.value, greeting: greet.value};
      if (isNew) await api("POST", "/api/presets",
                           Object.assign({params: (p && p.params) || {}}, body));
      else await api("POST", "/api/presets/" + p.id, body);
      await loadPresets();
      renderPresetsTab();
    };
    acts.appendChild(save);
    if (!isNew) {
      const del = el("button", "act danger", "Delete");
      del.onclick = async () => {
        if (!confirm('Delete preset "' + p.name + '"?')) return;
        await api("DELETE", "/api/presets/" + p.id);
        await loadPresets();
        renderPresetsTab();
      };
      acts.appendChild(del);
    }
  }
  const back = el("button", "act", "Back");
  back.onclick = renderPresetsTab;
  acts.appendChild(back);
  box.appendChild(acts);
}

/* ---------- server tab: engine room ---------- */
async function renderServerTab() {
  const box = $("drawer-body");
  box.innerHTML = "";
  box.appendChild(el("h3", "", "Engine"));
  let info = null;
  try { info = await api("GET", "/api/server"); } catch {}
  if (!info) {
    box.appendChild(el("p", "dim", "Server not running."));
    return;
  }
  const rows = [
    ["model", info.model + " (" + info.quant + ")"],
    ["backend", info.backend + "  ·  llama.cpp " + (info.engine_version || "?")],
    ["use case", info.use_case || "general"],
    ["context", (info.ctx || 0).toLocaleString() + " tokens"],
    ["uptime", Math.round((Date.now() / 1000 - info.started_at) / 60) + " min"],
    ["RAM free", (info.ram_free_mb / 1024).toFixed(1) + " / " +
                 (info.ram_total_mb / 1024).toFixed(1) + " GB"],
    ["decode", (info.last_tg ? info.last_tg.toFixed(1) + " tok/s" : "—") +
               (info.expected_tg ? "  (expected ~" +
                info.expected_tg.toFixed(1) + ")" : "")],
    ["verdict", info.verdict],
    ["agents", "point aider/Cline/Continue at " + (info.openai_base || "—") +
               " (see docs/agents.md)"],
  ];
  const tbl = el("div", "srv-rows");
  for (const [k, v] of rows) {
    const r = el("div", "srv-row");
    r.appendChild(el("span", "k", k));
    r.appendChild(el("span", "v" + (k === "verdict" ? " " + info.verdict : ""),
                     String(v)));
    tbl.appendChild(r);
  }
  box.appendChild(tbl);

  box.appendChild(el("h3", "", "Switch model (downloaded only)"));
  let opts = [];
  try { opts = await api("GET", "/api/server/switch-options"); } catch {}
  if (!opts.length) {
    box.appendChild(el("p", "dim",
      "No alternative models on disk. Download via: rigma up --model <slug>"));
  }
  const acts = el("div", "drawer-acts");
  for (const o of opts) {
    const b = el("button", "act", o.model + " — " + o.reason);
    b.onclick = () => doSwitch(o.model);
    acts.appendChild(b);
  }
  box.appendChild(acts);

  box.appendChild(el("h3", "", "Engine log"));
  const pre = el("pre", "srv-log", "loading…");
  const load = async () => {
    try {
      const r = await fetch("/api/server/log?lines=200");
      pre.textContent = (await r.text()) || "(empty)";
      pre.scrollTop = pre.scrollHeight;
    } catch { pre.textContent = "(log unavailable)"; }
  };
  const refresh = el("button", "act", "Refresh log");
  refresh.onclick = load;
  box.appendChild(pre);
  const acts2 = el("div", "drawer-acts");
  acts2.appendChild(refresh);
  box.appendChild(acts2);
  load();
}

/* ---------- rail search ---------- */
let searchTimer = null;
$("rail-search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = $("rail-search").value.trim();
    if (!q) { renderRail(); return; }
    let hits = [];
    try { hits = await api("GET", "/api/sessions/search?q="
                                  + encodeURIComponent(q)); } catch {}
    const nav = $("chat-list");
    nav.innerHTML = "";
    if (!hits.length) {
      nav.appendChild(el("div", "rail-empty", "No matches."));
      return;
    }
    for (const h of hits) {
      const item = el("div", "chat-item");
      item.setAttribute("role", "button");
      item.tabIndex = 0;
      const wrap = el("span", "title");
      wrap.appendChild(el("div", "", h.title || "(untitled)"));
      wrap.appendChild(el("div", "snippet", h.snippet));
      item.appendChild(wrap);
      item.onclick = () => { $("rail-search").value = ""; openSession(h.id); };
      item.onkeydown = (e) => { if (e.key === "Enter") item.onclick(); };
      nav.appendChild(item);
    }
  }, 250);
});
