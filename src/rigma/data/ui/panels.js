"use strict";
/* panels.js — settings drawer (chat params / presets manager). Loads after
   app.js; talks to the same server-authoritative state via store.js. */

const PARAM_DEFS = [
  ["temperature", 0, 4, 0.05],
  ["top_p", 0, 1, 0.01],
  ["top_k", 0, 200, 1],
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
const INT_PARAMS = ["max_tokens", "dry_allowed_length", "top_k", "seed"];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/* ---------- drawer shell ---------- */
function toggleDrawer(tab) {
  const d = $("drawer");
  if (!d.hidden && !tab) { d.hidden = true; $("drawer-scrim").hidden = true;
                           return; }
  d.hidden = false;
  $("drawer-scrim").hidden = false;
  openTab(tab || "chat");
}
let activeTab = "chat";
function openTab(name) {
  activeTab = name;
  for (const b of document.querySelectorAll("#drawer-tabs button")) {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  }
  if (name === "chat") renderChatTab();
  else if (name === "presets") renderPresetsTab();
  else if (name === "server") renderServerTab();
}
function refreshDrawer() {
  if (!$("drawer").hidden) openTab(activeTab);
}
function closeDrawer() { $("drawer").hidden = true;
                        $("drawer-scrim").hidden = true; }
$("drawer-close").onclick = closeDrawer;
$("drawer-scrim").onclick = closeDrawer;
$("gear").onclick = () => toggleDrawer();
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("drawer").hidden) closeDrawer();
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
      const val = INT_PARAMS.includes(key) ? parseInt(num.value, 10)
                                           : parseFloat(num.value);
      if (Number.isNaN(val)) return;   // "-" / "." mid-type: don't send NaN
      range.value = num.value;
      params[key] = val; push();
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
  compactBtn.onclick = () => { closeDrawer(); compactChat(6); };
  macts.appendChild(compactBtn);
  box.appendChild(macts);

  box.appendChild(el("h3", "", "Tools — workspace & code"));
  box.appendChild(el("p", "dim",
    "Tools are on for new chats (toggle in the sys bar). Search, fetch, math, " +
    "date, and your documents work everywhere. File and code tools use the " +
    "workspace folder below — defaults to your home folder."));
  const wsRow = el("div", "path-row");
  const ws = el("input");
  ws.placeholder = "Workspace folder (lets the model read/list files there)";
  ws.value = current.workspace || "";
  const wsSave = el("button", "act", "Set");
  wsSave.onclick = async () => {
    if (!current || current.id !== sid) return;
    try { current = await api("POST", "/api/sessions/" + sid,
                              {workspace: ws.value.trim()}); }
    catch (err) { hint.textContent = err.message; }
  };
  ws.onkeydown = (e) => { if (e.key === "Enter") wsSave.onclick(); };
  wsRow.append(ws, wsSave);
  box.appendChild(wsRow);
  const codeRow = el("label", "cap-row");
  const codeCb = el("input");
  codeCb.type = "checkbox";
  codeCb.checked = !!current.allow_code;
  codeCb.onchange = async () => {
    if (!current || current.id !== sid) return;
    try { current = await api("POST", "/api/sessions/" + sid,
                              {allow_code: codeCb.checked}); }
    catch (err) { hint.textContent = err.message; codeCb.checked = !codeCb.checked; }
  };
  codeRow.append(codeCb, document.createTextNode(
    " Allow running Python & shell on this machine (starts in the workspace "
    + "folder, but code can reach anywhere you can)"));
  box.appendChild(codeRow);
  const roundsRow = el("label", "cap-row");
  const roundsIn = el("input");
  roundsIn.type = "number"; roundsIn.min = "1"; roundsIn.max = "100";
  roundsIn.style.width = "64px";
  roundsIn.value = current.max_tool_rounds || 25;
  roundsIn.title = "How many tool-call rounds the model may take in one reply "
    + "before it must stop (raise for long autonomous jobs)";
  roundsIn.onchange = async () => {
    if (!current || current.id !== sid) return;
    const v = Math.max(1, Math.min(100, parseInt(roundsIn.value, 10) || 25));
    roundsIn.value = v;
    try { current = await api("POST", "/api/sessions/" + sid,
                              {max_tool_rounds: v}); }
    catch (err) { hint.textContent = err.message; }
  };
  roundsRow.append(document.createTextNode("Max tool calls per turn: "),
                   roundsIn);
  box.appendChild(roundsRow);

  box.appendChild(el("h3", "", "This chat"));
  const acts = el("div", "drawer-acts");
  const exMd = el("a", "act", "Export markdown");
  exMd.href = "/api/sessions/" + sid + "/export?fmt=md";
  const exJs = el("a", "act", "Export JSON");
  exJs.href = "/api/sessions/" + sid + "/export?fmt=json";
  const dup = el("button", "act", "Duplicate chat");
  dup.onclick = async () => {
    const d = await api("POST", "/api/sessions/" + sid + "/duplicate");
    closeDrawer();
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

  try {
    const s = await api("GET", "/api/server/stats");
    if (s.total_turns) {
      box.appendChild(el("h3", "", "Lifetime"));
      const od = el("div", "odometer");
      const stat = (n, k) => {
        const c = el("div");
        c.appendChild(el("span", "n", n.toLocaleString()));
        c.appendChild(el("span", "k", k));
        od.appendChild(c);
      };
      stat(s.total_tokens || 0, "tokens generated");
      stat(s.total_turns || 0, "replies");
      const top = Object.entries(s.by_model || {})
        .sort((a, b) => b[1] - a[1])[0];
      if (top) {
        const c = el("div");
        c.appendChild(el("span", "n", top[0]));
        c.appendChild(el("span", "k", "most used"));
        od.appendChild(c);
      }
      box.appendChild(od);
    }
  } catch {}

  box.appendChild(el("h3", "", "Context size"));
  box.appendChild(el("p", "dim",
    "Relaunches the engine at the new size — bigger context costs " +
    "VRAM/RAM, and the fit math keeps it honest."));
  const ctxRow = el("div", "ctx-presets");
  const applyCtx = async (want) => {
    const ov = $("switching");
    ov.firstElementChild.textContent = "resizing context to "
      + want.toLocaleString() + "…";
    ov.hidden = false;
    try { await api("POST", "/api/server/ctx", {ctx: want}); }
    catch (e) { alert(e.message); }
    ov.hidden = true;
    pollEngine();
    renderServerTab();
  };
  for (const k of [8192, 16384, 32768, 65536, 131072, 262144]) {
    const b = el("button", "act mini"
      + (info.ctx === k ? " current" : ""), (k / 1024) + "K");
    if (info.ctx === k) b.disabled = true;
    b.onclick = () => applyCtx(k);
    ctxRow.appendChild(b);
  }
  const ctxIn = el("input");
  ctxIn.type = "number";
  ctxIn.placeholder = "custom";
  ctxIn.min = 2048;
  ctxIn.step = 1024;
  const ctxGo = el("button", "act mini", "Apply");
  ctxGo.onclick = () => {
    const v = parseInt(ctxIn.value, 10);
    if (v >= 2048) applyCtx(v);
  };
  ctxIn.onkeydown = (e) => { if (e.key === "Enter") ctxGo.onclick(); };
  ctxRow.append(ctxIn, ctxGo);
  box.appendChild(ctxRow);

  box.appendChild(el("h3", "", "Engine memory"));
  const memActs = el("div", "drawer-acts");
  if (info.unloaded && !info.model) {
    box.appendChild(el("p", "dim",
      "No model loaded yet. Pick one from the Models tab or the list below — " +
      "it downloads, tunes, and loads on demand."));
    const pick = el("button", "act", "Open Models");
    pick.onclick = openModelsView;
    memActs.appendChild(pick);
  } else if (info.unloaded) {
    box.appendChild(el("p", "dim",
      "Engine unloaded — VRAM/RAM are free. Chats will error until a " +
      "model is loaded."));
    const loadBtn = el("button", "act", "Load " + info.model + " again");
    loadBtn.onclick = async () => {
      const ov = $("switching");
      ov.firstElementChild.textContent = "loading " + info.model + "…";
      ov.hidden = false;
      try { await api("POST", "/api/server/load"); }
      catch (e) { alert(e.message); }
      ov.hidden = true;
      pollEngine();
      renderServerTab();
    };
    memActs.appendChild(loadBtn);
  } else {
    const unloadBtn = el("button", "act", "Unload engine — free VRAM/RAM");
    unloadBtn.title = "Stops llama-server but keeps this UI running; " +
      "reload here or run any model from the Models tab";
    unloadBtn.onclick = async () => {
      try { await api("POST", "/api/server/unload"); }
      catch (e) { alert(e.message); }
      pollEngine();
      renderServerTab();
    };
    memActs.appendChild(unloadBtn);
  }
  box.appendChild(memActs);

  if (info.model && !info.unloaded) {
  box.appendChild(el("h3", "", "Hardware tuning"));
  box.appendChild(el("p", "dim",
    "Rigma auto-tunes each model on its first load. If a tune landed on a slow " +
    "config, re-optimize to measure again — baseline is always tested, so it " +
    "can never end up slower than plain defaults."));
  const tuneActs = el("div", "drawer-acts");
  const reBtn = el("button", "act", "Re-optimize " + info.model);
  reBtn.title = "Unloads the model, runs a quick benchmark sweep, and " +
    "relaunches on the fastest config (a few minutes)";
  reBtn.onclick = async () => {
    if (!confirm("Re-optimize " + info.model + "?\n\nThis unloads the model, " +
      "benchmarks a few configs, and relaunches on the fastest. Takes a few " +
      "minutes.")) return;
    const ov = $("switching");
    ov.firstElementChild.textContent = "optimizing " + info.model +
      " for your hardware…";
    ov.hidden = false;
    try { await api("POST", "/api/server/recalibrate"); }
    catch (e) { alert(e.message); }
    ov.hidden = true;
    pollEngine();
    renderServerTab();
  };
  tuneActs.appendChild(reBtn);
  box.appendChild(tuneActs);
  }

  box.appendChild(el("h3", "", info.model ? "Switch model (downloaded only)"
                                          : "Load a downloaded model"));
  let opts = [];
  try { opts = await api("GET", "/api/server/switch-options"); } catch {}
  if (!opts.length) {
    box.appendChild(el("p", "dim",
      "No models on disk yet. Use the Models tab to download one."));
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

/* ---------- models tab: the hangar ---------- */
function fmtGB(bytes) { return (bytes / 2 ** 30).toFixed(1) + " GB"; }
function fmtEta(sec) {
  if (sec < 90) return Math.round(sec) + "s";
  if (sec < 3600) return Math.round(sec / 60) + " min";
  return (sec / 3600).toFixed(1) + " h";
}

// fit verdict badge, speed-aware: a bigger quant that only "fits" via heavy
// RAM offload is slower, not better — say so plainly.
function fitBadge(fit) {
  if (!fit || !fit.ok) return el("span", "fit no", "too big for this machine");
  const ctx = "~" + Math.round(fit.ctx / 1024) + "K ctx";
  const tiers = {
    gpu: ["ok", "on GPU · " + ctx, "Runs fully on the GPU — fastest"],
    light: ["ok", "mostly GPU · " + ctx,
            "Mostly on the GPU, a little in RAM — still quick"],
    offload: ["warn", "RAM offload · " + ctx,
              "Part of the model runs in RAM every token — this will be slow"],
  };
  const t = tiers[fit.speed] || ["ok", ctx, ""];
  const b = el("span", "fit " + t[0], "fits — " + t[1]);
  if (t[2]) b.title = t[2];
  return b;
}

// collapsible "what do these mean?" glossary for gguf quant names
function quantLegend() {
  const box = el("details", "quant-legend");
  const sum = el("summary", "", "What do Q4, IQ4_XS, K vs I mean?");
  box.appendChild(sum);
  const rows = [
    ["The number (Q2…Q8)", "bits per weight. Higher = better quality + " +
      "bigger file. Q4 is the usual sweet spot; below Q3 quality drops off."],
    ["Q_K (K-quant)", "classic block quantization — well-tested, maximally " +
      "compatible, a touch faster raw."],
    ["IQ (i-quant)", "uses an importance matrix to spend bits where they " +
      "matter — smaller file for the same quality, tiny extra GPU compute."],
    ["_S / _M / _L / _XS", "size within a level: XS/S small, M balanced, L " +
      "largest/best. (Some uploaders add their own suffix like _P.)"],
    ["UD- (Unsloth Dynamic)", "per-tensor bit allocation for better quality " +
      "at the same size."],
    ["F16 / BF16", "16-bit, full quality, largest — overkill for chat."],
    ["★", "the best-quality quant that still fits your machine."],
  ];
  for (const [k, v] of rows) {
    const r = el("div", "gl-row");
    r.appendChild(el("span", "gl-k", k));
    r.appendChild(el("span", "gl-v", v));
    box.appendChild(r);
  }
  return box;
}

function uploadGguf(file, attachTo, onProg) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", "/api/models/upload?filename="
      + encodeURIComponent(file.name)
      + (attachTo ? "&attach_to=" + encodeURIComponent(attachTo) : ""));
    x.upload.onprogress = (e) => {
      if (e.lengthComputable && e.total > 0 && onProg)
        onProg(e.loaded / e.total);   // e.total==0 -> Infinity, guard it
    };
    x.onload = () => {
      let j = {};
      try { j = JSON.parse(x.responseText); } catch {}
      if (x.status === 200) resolve(j);
      else reject(new Error(j.error || ("upload failed (HTTP " + x.status + ")")));
    };
    x.onerror = () => reject(new Error("upload failed — server unreachable"));
    x.send(file);
  });
}

/* ---------- Models: full main-area view (the hangar) ---------- */
let modelsPollTimer = null;
let modelsRenderGen = 0;   // only the newest in-flight render may touch the DOM

function openModelsView() {
  closeDrawer();
  if (typeof closeAutoView === "function") closeAutoView();
  $("models-view").hidden = false;
  document.body.classList.add("models-open");
  renderModelsView();
}
function closeModelsView() {
  clearTimeout(modelsPollTimer);
  $("models-view").hidden = true;
  document.body.classList.remove("models-open");
}
$("open-models").onclick = openModelsView;
$("mv-close").onclick = closeModelsView;
$("mv-refresh").onclick = () => renderModelsView();

// one quant row for the collapsible per-model list (download / delete / progress)
function quantRow(m, q, reload) {
  const row = el("div", "quant-row");
  row.appendChild(el("span", "dot" + (q.on_disk ? " on" : "")));
  const ql = el("span", "q", q.quant);
  ql.title = quantHelp(q.quant);
  row.appendChild(ql);
  row.appendChild(el("span", "sz", fmtGB(q.bytes)));
  if (q.pull && q.pull.status === "downloading") {
    const done = q.pull.done || 0, tot = q.pull.total || q.bytes || 1;
    const pct = Math.round(100 * done / tot);
    const bar = el("span", "pull-bar");
    bar.appendChild(el("span", "fill")).style.width = pct + "%";
    row.appendChild(bar);
    let stat;
    if (done <= 0) {                       // no bytes yet — say so, don't lie
      stat = el("span", "sz pull-stat", "connecting…");
    } else {
      const parts = [pct + "%"];
      if (q.pull.bps) parts.push((q.pull.bps / 2 ** 20).toFixed(1) + " MB/s");
      stat = el("span", "sz pull-stat", parts.join(" · "));
      stat.title = fmtGB(done) + " of " + fmtGB(tot)
        + (q.pull.eta ? " · ~" + fmtEta(q.pull.eta) + " left" : "");
    }
    row.appendChild(stat);
  } else if (q.pull && q.pull.status === "error") {
    row.appendChild(el("span", "err", q.pull.error || "download failed"));
  } else if (!q.on_disk && q.pullable) {
    const dl = el("button", "act mini", "Download");
    dl.onclick = async () => {
      dl.disabled = true;
      try { await api("POST", "/api/models/" + m.slug + "/pull",
                      {file: q.file}); }
      catch (e) { alert(e.message); }
      reload();
    };
    row.appendChild(dl);
  } else if (!q.on_disk && !q.pullable) {
    row.appendChild(el("span", "err", "local-only — can't re-download"));
  } else if (q.on_disk) {
    const del = el("button", "clear", "✕");
    del.title = "Delete " + q.file + " from disk";
    del.setAttribute("aria-label", del.title);
    del.onclick = async () => {
      if (!confirm("Delete " + q.file + " (" + fmtGB(q.bytes)
                   + ") from disk?")) return;
      try { await api("DELETE", "/api/models/" + m.slug + "/files/"
                      + encodeURIComponent(q.file)); }
      catch (e) { alert(e.message); }
      reload();
    };
    row.appendChild(del);
  }
  return row;
}

// a library model as a grid card: leads with the on-disk quant + Run, the
// full quant list tucked behind a disclosure so the card stays scannable
function libraryCard(m, reload) {
  const card = el("div", "model-card" + (m.running ? " running" : ""));
  const head = el("div", "mc-head");
  head.appendChild(el("span", "mc-name", m.slug));
  if (m.running) head.appendChild(el("span", "badge live", "RUNNING"));
  if (m.custom) head.appendChild(el("span", "badge", "custom"));
  for (const c of m.capabilities || [])
    head.appendChild(el("span", "cap " + c, c === "thinking" ? "think" : c));
  card.appendChild(head);
  card.appendChild(el("div", "mc-sub", m.family + " · " + m.kind + " · ctx "
    + (m.native_ctx || 0).toLocaleString()
    + (m.mmproj ? " · mmproj " + (m.mmproj.on_disk ? "on disk"
        : "not downloaded") : "")));

  const onDisk = m.quants.filter((q) => q.on_disk);
  const pulling = m.quants.find((q) => q.pull && q.pull.status === "downloading");
  if (pulling) {
    const p = el("div", "mc-primary");
    p.appendChild(quantRow(m, pulling, reload));
    card.appendChild(p);
  } else if (onDisk.length) {
    const q = onDisk[0], p = el("div", "mc-primary");
    const ql = el("span", "q", q.quant);
    ql.title = quantHelp(q.quant);
    p.appendChild(el("span", "dot on"));
    p.append(ql, el("span", "sz", fmtGB(q.bytes) + " · ready"));
    card.appendChild(p);
  }

  // the rest of the quants (not the one already shown in the primary line):
  // lead with downloads expanded when nothing's on disk, else tuck away
  const shown = pulling || onDisk[0];
  const rest = m.quants.filter((q) => q !== shown);
  if (rest.length) {
    const others = rest.filter((q) => !q.on_disk);
    const det = el("details", "mc-quants");
    if (!onDisk.length && !pulling) det.open = true;
    const canPull = others.some((q) => q.pullable);
    det.appendChild(el("summary", "",
      (onDisk.length || pulling) ? rest.length + " more quant"
        + (rest.length > 1 ? "s" : "") + " — manage"
        : (canPull ? "Download a quant ↓" : "not on disk")));
    for (const q of rest) det.appendChild(quantRow(m, q, reload));
    card.appendChild(det);
  }

  // vision projector: a separate file that turns on image understanding.
  // surface it as its own downloadable row when it isn't on disk yet.
  if (m.mmproj && (!m.mmproj.on_disk || m.mmproj.pull)) {
    const mm = m.mmproj, row = el("div", "quant-row mmproj-row");
    row.appendChild(el("span", "dot" + (mm.on_disk ? " on" : "")));
    const lbl = el("span", "q", "vision projector");
    lbl.title = "Enables image understanding — this model needs it to see images";
    row.appendChild(lbl);
    row.appendChild(el("span", "sz", fmtGB(mm.bytes)));
    if (mm.pull && mm.pull.status === "downloading") {
      const done = mm.pull.done || 0, tot = mm.pull.total || mm.bytes || 1;
      const bar = el("span", "pull-bar");
      bar.appendChild(el("span", "fill")).style.width =
        Math.round(100 * done / tot) + "%";
      row.appendChild(bar);
      row.appendChild(el("span", "sz pull-stat", done <= 0 ? "connecting…"
        : Math.round(100 * done / tot) + "%"
          + (mm.pull.bps ? " · " + (mm.pull.bps / 2 ** 20).toFixed(1)
             + " MB/s" : "")));
    } else if (!mm.on_disk && mm.pullable) {
      const dl = el("button", "act mini", "Download");
      dl.onclick = async () => {
        dl.disabled = true;
        try { await api("POST", "/api/models/" + m.slug + "/pull",
                        {file: mm.file}); }
        catch (e) { alert(e.message); }
        reload();
      };
      row.appendChild(dl);
    } else if (!mm.on_disk) {
      row.appendChild(el("span", "err", "local-only"));
    }
    card.appendChild(row);
  }

  const acts = el("div", "mc-acts");
  if (!m.running && onDisk.length) {
    const run = el("button", "act primary", "Run");
    run.onclick = () => { closeModelsView(); doSwitch(m.slug); };
    acts.appendChild(run);
  }
  if (m.custom) {
    const caps = el("button", "act", "Edit capabilities");
    caps.onclick = () => renderCapsEditor(m);
    acts.appendChild(caps);
    const rm = el("button", "act danger", "Remove");
    rm.onclick = async () => {
      if (!confirm("Remove " + m.slug + " and delete its files?")) return;
      try { await api("DELETE", "/api/models/" + m.slug); }
      catch (e) { alert(e.message); }
      reload();
    };
    acts.appendChild(rm);
  }
  if (acts.childNodes.length) card.appendChild(acts);
  return card;
}

async function renderModelsView() {
  clearTimeout(modelsPollTimer);
  const myGen = ++modelsRenderGen;
  const reload = refreshModelsGrid;   // card actions refresh grid only
  const body = $("mv-body");
  let data = null;
  try { data = await api("GET", "/api/models"); } catch (e) {
    if (myGen === modelsRenderGen) {
      body.innerHTML = "";
      body.appendChild(el("p", "dim", e.message));
    }
    return;
  }
  if (myGen !== modelsRenderGen || $("models-view").hidden) return;
  body.innerHTML = "";
  $("mv-disk").textContent = data.disk.models_gb + " GB on disk · "
    + data.disk.free_gb + " GB free";
  $("mv-disk").title = data.disk.dir;

  // toolbar: find-on-HF + install-your-own, side by side on wide screens
  const tools = el("div", "mv-tools");

  const find = el("div", "mv-panel");
  find.appendChild(el("h3", "", "Find a model on Hugging Face"));
  const hfRow = el("div", "path-row");
  const hfIn = el("input");
  hfIn.placeholder = "Search any gguf model…";
  const hfBtn = el("button", "act", "Search");
  hfRow.append(hfIn, hfBtn);
  find.appendChild(hfRow);
  const hfBox = el("div", "hf-results");
  find.appendChild(hfBox);
  let hfGen = 0;
  const hfDetail = async (repo) => {
    const g = ++hfGen;
    hfBox.innerHTML = "";
    hfBox.appendChild(el("p", "dim",
      "reading " + repo + "'s header remotely (a few MB, not the model)…"));
    let d = null;
    try { d = await api("GET", "/api/hf/repo?id=" + encodeURIComponent(repo)); }
    catch (e) {
      if (g !== hfGen) return;
      hfBox.innerHTML = "";
      hfBox.appendChild(el("p", "drop-status err", e.message));
      return;
    }
    if (g !== hfGen) return;
    hfBox.innerHTML = "";
    const card = el("div", "model-card");
    const chead = el("div", "mc-head");
    chead.appendChild(el("span", "mc-name", d.name));
    for (const c of d.capabilities || [])
      chead.appendChild(el("span", "cap " + c, c === "thinking" ? "think" : c));
    card.appendChild(chead);
    card.appendChild(el("div", "mc-sub", repo + " · " + d.kind + " · ctx "
      + (d.native_ctx || 0).toLocaleString()
      + (d.mmproj ? " · mmproj included" : "")
      + (d.split_skipped ? " · " + d.split_skipped + " split files skipped"
          : "")));
    const rec = d.recommended || recommendedQuant(d.ggufs);
    for (const q of d.ggufs) {
      const row = el("div", "quant-row" + (q.quant === rec ? " rec-row" : ""));
      const ql = el("span", "q", q.quant);
      ql.title = quantHelp(q.quant);
      row.appendChild(ql);
      if (q.quant === rec) {
        const tag = el("span", "rec-tag", "★ Recommended");
        tag.title = "Best quality that still runs at GPU speed on your machine";
        row.appendChild(tag);
      }
      row.appendChild(el("span", "sz", fmtGB(q.bytes)));
      row.appendChild(fitBadge(q.fit));
      card.appendChild(row);
    }
    card.appendChild(quantLegend());
    const acts = el("div", "mc-acts");
    if (d.already) {
      acts.appendChild(el("span", "dim", "already in your library"));
    } else {
      const add = el("button", "act primary", "Add to library");
      add.onclick = async () => {
        add.disabled = true;
        try {
          await api("POST", "/api/hf/add", {repo});
          add.textContent = "Added ✓";
          refreshModelsGrid();   // show it in Your Models now, keep this panel
        } catch (e) { add.disabled = false; alert(e.message); }
      };
      acts.appendChild(add);
    }
    const back = el("button", "act", "Back to results");
    back.onclick = () => doSearch();
    acts.appendChild(back);
    card.appendChild(acts);
    hfBox.appendChild(card);
  };
  const doSearch = async () => {
    const g = ++hfGen;
    const q = hfIn.value.trim();
    if (!q) { hfBox.innerHTML = ""; return; }
    hfBox.innerHTML = "";
    hfBox.appendChild(el("p", "dim", "searching…"));
    let rows = [];
    try { rows = await api("GET", "/api/hf/search?q=" + encodeURIComponent(q)); }
    catch (e) {
      if (g !== hfGen) return;
      hfBox.innerHTML = "";
      hfBox.appendChild(el("p", "drop-status err", e.message));
      return;
    }
    if (g !== hfGen) return;
    hfBox.innerHTML = "";
    if (!rows.length) {
      hfBox.appendChild(el("p", "dim", "no gguf models found"));
      return;
    }
    for (const r of rows) {
      const b = el("button", "hf-row");
      b.type = "button";
      b.appendChild(el("span", "repo", r.repo));
      b.appendChild(el("span", "meta", "↓ " + (r.downloads || 0).toLocaleString()
        + (r.likes ? "  ♥ " + r.likes : "")));
      b.onclick = () => hfDetail(r.repo);
      hfBox.appendChild(b);
    }
  };
  hfBtn.onclick = doSearch;
  hfIn.onkeydown = (e) => { if (e.key === "Enter") doSearch(); };

  const inst = el("div", "mv-panel");
  inst.appendChild(el("h3", "", "Install your own"));
  const zone = el("div", "drop-zone");
  zone.appendChild(el("div", "big", "Drop a .gguf here"));
  zone.appendChild(el("div", "dim",
    "Fine-tunes welcome — Rigma reads the file's own header to size it. " +
    "For a vision projector (mmproj-*.gguf), install its model first."));
  const zStatus = el("div", "drop-status");
  const doInstall = async (run) => {
    try {
      const j = await run();
      zStatus.textContent = "installed: " + j.slug;
      zStatus.className = "drop-status ok";
      reload();
    } catch (e) {
      zStatus.textContent = e.message;
      zStatus.className = "drop-status err";
    }
  };
  zone.ondragover = (e) => { e.preventDefault(); zone.classList.add("over"); };
  zone.ondragleave = () => zone.classList.remove("over");
  zone.ondrop = (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    let attach = "";
    if (/mmproj/i.test(f.name)) {
      const customs = data.models.filter((m) => m.custom).map((m) => m.slug);
      if (!customs.length) {
        zStatus.textContent = "that's a vision projector — install its model first";
        zStatus.className = "drop-status err";
        return;
      }
      attach = prompt("Attach this projector to which custom model?\n"
                      + customs.join("\n"), customs[0]) || "";
      if (!attach) return;
    }
    doInstall(() => uploadGguf(f, attach, (p) => {
      zStatus.textContent = "uploading " + f.name + " — "
        + Math.round(p * 100) + "%";
      zStatus.className = "drop-status";
    }));
  };
  inst.appendChild(zone);
  const pathRow = el("div", "path-row");
  const pathIn = el("input");
  pathIn.placeholder = "…or paste a file path (moved into Rigma's folder)";
  const pathBtn = el("button", "act", "Install");
  pathBtn.onclick = () => {
    const p = pathIn.value.trim();
    if (p) doInstall(() => api("POST", "/api/models/install", {path: p}));
  };
  pathIn.onkeydown = (e) => { if (e.key === "Enter") pathBtn.onclick(); };
  pathRow.append(pathIn, pathBtn);
  inst.appendChild(pathRow);
  inst.appendChild(zStatus);

  tools.append(find, inst);
  body.appendChild(tools);

  // your models — the grid. Rendered into its own container so it can be
  // refreshed in place (e.g. after "Add to library") without rebuilding the
  // HF browse panel and losing the user's search/detail.
  body.appendChild(el("h3", "mv-section", "Your models"));
  body.appendChild(el("div", "model-grid", "")).id = "mv-grid";
  await refreshModelsGrid();
}

async function refreshModelsGrid() {
  const grid = document.getElementById("mv-grid");
  if (!grid || $("models-view").hidden) return;
  clearTimeout(modelsPollTimer);
  const myGen = ++modelsRenderGen;
  let data;
  try { data = await api("GET", "/api/models"); } catch { return; }
  if (myGen !== modelsRenderGen || !document.getElementById("mv-grid")
      || $("models-view").hidden) return;
  $("mv-disk").textContent = data.disk.models_gb + " GB on disk · "
    + data.disk.free_gb + " GB free";
  grid.innerHTML = "";
  for (const m of data.models)
    grid.appendChild(libraryCard(m, refreshModelsGrid));
  const busy = (m) => m.quants.some((q) =>
      q.pull && q.pull.status === "downloading")
    || (m.mmproj && m.mmproj.pull && m.mmproj.pull.status === "downloading");
  if (data.models.some(busy))
    modelsPollTimer = setTimeout(refreshModelsGrid, 1500);
}

function renderCapsEditor(m) {
  const box = $("mv-body");
  box.innerHTML = "";
  box.appendChild(el("h3", "mv-section", m.slug + " — capabilities"));
  box.appendChild(el("p", "dim",
    "Header-derived guesses; correct them if the model card says otherwise. " +
    "Vision needs an attached mmproj."));
  const picked = new Set(m.capabilities || []);
  for (const c of ["tools", "thinking", "vision"]) {
    const row = el("label", "cap-row");
    const cb = el("input");
    cb.type = "checkbox";
    cb.checked = picked.has(c);
    cb.onchange = () => { cb.checked ? picked.add(c) : picked.delete(c); };
    row.append(cb, document.createTextNode(" " + c));
    box.appendChild(row);
  }
  const acts = el("div", "drawer-acts");
  const save = el("button", "act", "Save");
  save.onclick = async () => {
    try {
      await api("PATCH", "/api/models/" + m.slug,
                {capabilities: [...picked]});
      renderModelsView();
    } catch (e) { alert(e.message); }
  };
  const back = el("button", "act", "Back");
  back.onclick = renderModelsView;
  acts.append(save, back);
  box.appendChild(acts);
}

/* ---------- rail search ---------- */
let searchTimer = null, railSearchGen = 0;
$("rail-search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const g = ++railSearchGen;   // out-of-order responses must not clobber
    const q = $("rail-search").value.trim();
    if (!q) { renderRail(); return; }
    let hits = [];
    try { hits = await api("GET", "/api/sessions/search?q="
                                  + encodeURIComponent(q)); } catch {}
    if (g !== railSearchGen) return;
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

/* ---------- Autonomous: full main-area view ---------- */
let autoPollTimer = null;

function openAutoView() {
  closeDrawer();
  if (typeof closeModelsView === "function") closeModelsView();
  $("auto-view").hidden = false;
  document.body.classList.add("auto-open");
  renderAutoView();
}
function closeAutoView() {
  clearInterval(autoPollTimer); autoPollTimer = null;
  $("auto-view").hidden = true;
  document.body.classList.remove("auto-open");
}
$("open-auto").onclick = openAutoView;
$("auto-close").onclick = closeAutoView;
$("auto-refresh").onclick = () => renderAutoView();

let autoRunId = null;   // the run we're showing (persists after it ends)

async function renderAutoView() {
  const box = $("auto-body"); box.innerHTML = "";
  clearInterval(autoPollTimer); autoPollTimer = null;
  let run = {};
  try { run = await api("GET", "/api/runs/active"); } catch {}
  if (run && run.id) {          // a live run
    autoRunId = run.id;
    renderAutoDash(box, run);
    autoPollTimer = setInterval(refreshAuto, 3000);
    return;
  }
  if (autoRunId) {              // no live run, but show the last one's outcome
    try {
      const last = await api("GET", "/api/runs/" + autoRunId);
      if (last && last.id) { renderAutoDash(box, last); return; }
    } catch {}
  }
  renderAutoStart(box);
}

function _input(type, ph, val) {
  const i = document.createElement(type === "textarea" ? "textarea" : "input");
  if (type !== "textarea") i.type = type;
  if (ph) i.placeholder = ph; if (val != null) i.value = val;
  return i;
}

function renderAutoStart(box) {
  box.appendChild(el("p", "dim",
    "Give the model a mission and it works on its own — planning, acting, "
    + "logging progress, and compacting its own context — until it's done or a "
    + "safety cap trips. It keeps going even if you close this tab."));
  box.appendChild(el("label", "", "Mission"));
  const mission = _input("textarea",
    "Be specific. e.g. \"Go through the images in D:\\Art, write a taste "
    + "directive, then 100 new prompts into prompts.txt\"");
  box.appendChild(mission);
  const row = el("div", "row");
  const hoursL = el("label", "", "Time budget (hours)");
  const hours = _input("text", "", "8"); hoursL.appendChild(hours);
  const profL = el("label", "", "Safety profile");
  const prof = document.createElement("select");
  for (const p of ["all", "no-network", "no-delete", "confined"]) {
    const o = document.createElement("option"); o.value = p; o.textContent = p;
    prof.appendChild(o);
  }
  profL.appendChild(prof);
  const effL = el("label", "", "Reasoning");
  const eff = document.createElement("select");
  for (const [v, t] of [["on", "on — plan each step (best quality)"],
                        ["auto", "auto — model default"],
                        ["off", "off — act fast, no thinking"]]) {
    const o = document.createElement("option"); o.value = v; o.textContent = t;
    eff.appendChild(o);
  }
  effL.appendChild(eff);
  row.append(hoursL, profL, effL); box.appendChild(row);
  box.appendChild(el("label", "", "Workspace folder (where it works / writes)"));
  const ws = _input("text", "D:\\project  (optional — defaults to home)");
  box.appendChild(ws);
  const acts = el("div", "acts");
  const start = el("button", "act", "▶ Start autonomous run");
  start.onclick = async () => {
    if (!mission.value.trim()) { alert("Enter a mission first."); return; }
    start.disabled = true;
    try {
      const run = await api("POST", "/api/runs", {
        mission: mission.value.trim(), budget_hours: parseFloat(hours.value) || 8,
        profile: prof.value, effort: eff.value, workspace: ws.value.trim()});
      autoRunId = run.id;                                   // track it
      if (typeof renderRail === "function") renderRail();   // show its chat
      // render the dashboard immediately from the created run, then poll —
      // don't re-fetch /active (a fast-stalling run would flip back to the form)
      const b = $("auto-body"); b.innerHTML = "";
      renderAutoDash(b, run);
      clearInterval(autoPollTimer);
      autoPollTimer = setInterval(refreshAuto, 3000);
    } catch (e) { alert(e.message); start.disabled = false; }
  };
  acts.appendChild(start); box.appendChild(acts);
}

function _renderPlan(ul, plan) {
  ul.innerHTML = "";
  for (const t of (plan || [])) {
    const li = document.createElement("li");
    li.className = t.status === "done" ? "done" : "";
    li.textContent = t.text; ul.appendChild(li);
  }
  if (!(plan || []).length) ul.appendChild(el("li", "dim", "(no plan yet)"));
}

function _dur(s) {
  s = Math.round(s || 0);
  return s < 60 ? s + "s" : Math.floor(s / 60) + "m " + (s % 60) + "s";
}

function renderAutoDash(box, run) {
  const head = el("div", "");
  head.appendChild(el("span", "auto-badge " + run.status, run.status));
  head.appendChild(document.createTextNode(
    "  iter " + (run.iteration || 0) + " · errors " + (run.error_streak || 0)));
  // live heartbeat: a slow turn must never look like a frozen screen
  if (run.status === "running" && (run.waiting_secs || 0) > 0) {
    head.appendChild(el("span", "auto-wait", "  ⏳ "
      + (run.waiting_kind === "starting"
         ? "model is starting (loading your context)"
         : "generating")
      + " — " + _dur(run.waiting_secs) + " so far"));
  }
  box.appendChild(head);
  box.appendChild(el("h3", "", "Mission"));
  box.appendChild(el("p", "", run.mission || ""));
  box.appendChild(el("h3", "", "Plan"));
  const ul = document.createElement("ul"); ul.id = "auto-plan";
  _renderPlan(ul, run.plan); box.appendChild(ul);
  box.appendChild(el("h3", "", "Progress log"));
  const log = el("div", ""); log.id = "auto-log";
  log.textContent = run.log_tail
    || ((run.waiting_secs || 0) > 0
        ? "working… nothing logged yet (" + _dur(run.waiting_secs) + ")"
        : "(waiting…)");
  box.appendChild(log);
  const live = ["running", "paused", "waiting_approval"].includes(run.status);
  if (live) {
    box.appendChild(el("h3", "", "Steer — course-correct without stopping"));
    const steer = _input("text", "e.g. \"stop using regex, use BeautifulSoup\"");
    box.appendChild(steer);
    const acts = el("div", "acts");
    const send = el("button", "act", "Send guidance");
    send.onclick = async () => {
      if (!steer.value.trim()) return;
      await api("POST", "/api/runs/" + run.id + "/inject",
                {message: steer.value.trim()});
      steer.value = ""; send.textContent = "queued ✓";
      setTimeout(() => { send.textContent = "Send guidance"; }, 1500);
    };
    acts.appendChild(send);
    const pb = el("button", "act", run.paused ? "Resume" : "Pause");
    pb.onclick = async () => {
      await api("POST", "/api/runs/" + run.id
                + "/" + (run.paused ? "resume" : "pause"), {});
      renderAutoView();
    };
    acts.appendChild(pb);
    const stop = el("button", "act", "Stop run");
    stop.onclick = async () => {
      if (!confirm("Stop this run?")) return;
      await api("POST", "/api/runs/" + run.id + "/stop", {});
      renderAutoView();
    };
    acts.appendChild(stop); box.appendChild(acts);
  } else {
    box.appendChild(el("p", "dim",
      "Run ended: " + (run.summary || run.halt_reason || run.status)));
    const acts = el("div", "acts");
    const nb = el("button", "act", "Start a new run");
    nb.onclick = () => { autoRunId = null; renderAutoView(); };
    acts.appendChild(nb); box.appendChild(acts);
  }
}

async function refreshAuto() {
  if ($("auto-view").hidden) { clearInterval(autoPollTimer); return; }
  let run = {};
  try {   // poll the tracked run (persists after it ends), not /active
    run = await api("GET", "/api/runs/" + (autoRunId || "_"));
  } catch { return; }
  const badge = $("auto-body").querySelector(".auto-badge");
  const logEl = $("auto-log");
  if (!run || !run.id || !badge || !logEl
      || !["running", "paused", "waiting_approval"].includes(run.status)) {
    renderAutoView(); return;   // terminal or shape changed: full re-render
  }
  badge.className = "auto-badge " + run.status; badge.textContent = run.status;
  const head = badge.parentElement;
  head.childNodes[1].nodeValue =
    "  iter " + (run.iteration || 0) + " · errors " + (run.error_streak || 0);
  logEl.textContent = run.log_tail || "(waiting…)";
  logEl.scrollTop = logEl.scrollHeight;
  const ul = $("auto-plan"); if (ul) _renderPlan(ul, run.plan);
}
