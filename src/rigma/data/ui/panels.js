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
  else if (name === "models") renderModelsTab();
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
  if (info.unloaded) {
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

/* ---------- models tab: the hangar ---------- */
function fmtGB(bytes) { return (bytes / 2 ** 30).toFixed(1) + " GB"; }

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

let modelsPollTimer = null;
let modelsRenderGen = 0;   // only the newest in-flight render may touch the DOM
async function renderModelsTab() {
  clearTimeout(modelsPollTimer);
  const myGen = ++modelsRenderGen;
  const box = $("drawer-body");
  let data = null;
  try { data = await api("GET", "/api/models"); } catch (e) {
    if (myGen !== modelsRenderGen) return;
    box.innerHTML = "";
    box.appendChild(el("p", "dim", e.message));
    return;
  }
  // stale render: tab changed, drawer closed, or a newer render superseded us
  if (myGen !== modelsRenderGen || activeTab !== "models"
      || $("drawer").hidden) return;
  box.innerHTML = "";

  // disk gauge
  const gauge = el("div", "disk-gauge");
  gauge.appendChild(el("span", "k", "models on disk"));
  gauge.appendChild(el("span", "v", data.disk.models_gb + " GB"));
  gauge.appendChild(el("span", "k", "disk free"));
  gauge.appendChild(el("span", "v", data.disk.free_gb + " GB"));
  gauge.title = data.disk.dir;
  box.appendChild(gauge);

  // install zone: drop a .gguf, or install from a local path (moved, not copied)
  box.appendChild(el("h3", "", "Install a model"));
  const zone = el("div", "drop-zone");
  zone.appendChild(el("div", "big", "Drop a .gguf here"));
  zone.appendChild(el("div", "dim",
    "Fine-tunes welcome — Rigma reads the file's own header to size it. " +
    "For a vision projector (mmproj-*.gguf), install its model first."));
  const zStatus = el("div", "drop-status");   // shared by drop + path installs
  const doInstall = async (run) => {
    try {
      const j = await run();
      zStatus.textContent = "installed: " + j.slug;
      zStatus.className = "drop-status ok";
      renderModelsTab();
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
  box.appendChild(zone);
  const pathRow = el("div", "path-row");
  const pathIn = el("input");
  pathIn.placeholder = "…or paste a file path (file is MOVED into Rigma's folder)";
  const pathBtn = el("button", "act", "Install");
  pathBtn.onclick = () => {
    const p = pathIn.value.trim();
    if (p) doInstall(() => api("POST", "/api/models/install", {path: p}));
  };
  pathIn.onkeydown = (e) => { if (e.key === "Enter") pathBtn.onclick(); };
  pathRow.append(pathIn, pathBtn);
  box.appendChild(pathRow);
  box.appendChild(zStatus);

  // Hugging Face browser: search anything, fit-check before download
  box.appendChild(el("h3", "", "Find models on Hugging Face"));
  const hfRow = el("div", "path-row");
  const hfIn = el("input");
  hfIn.placeholder = "Search all gguf models on Hugging Face…";
  const hfBtn = el("button", "act", "Search");
  hfRow.append(hfIn, hfBtn);
  box.appendChild(hfRow);
  const hfBox = el("div", "hf-results");
  box.appendChild(hfBox);
  let hfGen = 0;   // out-of-order search/detail responses must not clobber
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
    const head = el("div", "mc-head");
    head.appendChild(el("span", "mc-name", d.name));
    for (const c of d.capabilities || [])
      head.appendChild(el("span", "cap " + c, c === "thinking" ? "think" : c));
    card.appendChild(head);
    card.appendChild(el("div", "mc-sub", repo + " · " + d.kind + " · ctx "
      + (d.native_ctx || 0).toLocaleString()
      + (d.mmproj ? " · mmproj included" : "")
      + (d.split_skipped ? " · " + d.split_skipped + " split files skipped" : "")));
    for (const q of d.ggufs) {
      const row = el("div", "quant-row");
      row.appendChild(el("span", "q", q.quant));
      row.appendChild(el("span", "sz", fmtGB(q.bytes)));
      const fit = q.fit && q.fit.ok
        ? el("span", "fit ok", "fits — ~" + Math.round(q.fit.ctx / 1024) + "K ctx")
        : el("span", "fit no", "too big for this machine");
      row.appendChild(fit);
      card.appendChild(row);
    }
    const acts = el("div", "drawer-acts");
    if (d.already) {
      acts.appendChild(el("span", "dim", "already in your library"));
    } else {
      const add = el("button", "act", "Add to library");
      add.onclick = async () => {
        add.disabled = true;
        try {
          await api("POST", "/api/hf/add", {repo});
          // keep the detail card in view (don't nuke the user's browse state);
          // just mark it added — the model now appears in the list below too
          add.textContent = "Added ✓";
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
    if (!rows.length) { hfBox.appendChild(el("p", "dim", "no gguf models found")); return; }
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

  // model cards
  box.appendChild(el("h3", "", "Models"));
  let anyPulling = false;
  for (const m of data.models) {
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
      + (m.mmproj ? " · mmproj " + (m.mmproj.on_disk ? "on disk" : "not downloaded") : "")));
    const onDisk = m.quants.some((q) => q.on_disk);
    for (const q of m.quants) {
      const row = el("div", "quant-row");
      row.appendChild(el("span", "dot" + (q.on_disk ? " on" : "")));
      row.appendChild(el("span", "q", q.quant));
      row.appendChild(el("span", "sz", fmtGB(q.bytes)));
      if (q.pull && q.pull.status === "downloading") {
        anyPulling = true;
        const pct = q.pull.total ? Math.round(100 * (q.pull.done || 0) / q.pull.total) : 0;
        const bar = el("span", "pull-bar");
        bar.appendChild(el("span", "fill")).style.width = pct + "%";
        row.appendChild(bar);
        row.appendChild(el("span", "sz", pct + "%"));
      } else if (q.pull && q.pull.status === "error") {
        const err = el("span", "err", q.pull.error || "download failed");
        row.appendChild(err);
      } else if (!q.on_disk && !m.custom) {
        const dl = el("button", "act mini", "Download");
        dl.onclick = async () => {
          dl.disabled = true;
          try { await api("POST", "/api/models/" + m.slug + "/pull", {file: q.file}); }
          catch (e) { alert(e.message); }
          renderModelsTab();
        };
        row.appendChild(dl);
      } else if (q.on_disk) {
        const del = el("button", "clear", "✕");
        del.title = "Delete " + q.file + " from disk";
        del.setAttribute("aria-label", del.title);
        del.onclick = async () => {
          if (!confirm("Delete " + q.file + " (" + fmtGB(q.bytes) + ") from disk?")) return;
          try {
            await api("DELETE", "/api/models/" + m.slug + "/files/"
                      + encodeURIComponent(q.file));
          } catch (e) { alert(e.message); }
          renderModelsTab();
        };
        row.appendChild(del);
      }
      card.appendChild(row);
    }
    const acts = el("div", "drawer-acts");
    if (!m.running && onDisk) {
      const run = el("button", "act", "Run this model");
      run.onclick = () => doSwitch(m.slug);
      acts.appendChild(run);
    }
    if (m.custom) {
      const caps = el("button", "act", "Edit capabilities");
      caps.onclick = () => renderCapsEditor(m);
      acts.appendChild(caps);
      const rm = el("button", "act danger", "Remove");
      rm.onclick = async () => {
        if (!confirm("Remove " + m.slug + " and delete its files from disk?")) return;
        try { await api("DELETE", "/api/models/" + m.slug); }
        catch (e) { alert(e.message); }
        renderModelsTab();
      };
      acts.appendChild(rm);
    }
    if (acts.childNodes.length) card.appendChild(acts);
    box.appendChild(card);
  }
  if (anyPulling)
    modelsPollTimer = setTimeout(() => {
      if (activeTab === "models" && !$("drawer").hidden) renderModelsTab();
    }, 1500);
}

function renderCapsEditor(m) {
  const box = $("drawer-body");
  box.innerHTML = "";
  box.appendChild(el("h3", "", m.slug + " — capabilities"));
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
      renderModelsTab();
    } catch (e) { alert(e.message); }
  };
  const back = el("button", "act", "Back");
  back.onclick = renderModelsTab;
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
