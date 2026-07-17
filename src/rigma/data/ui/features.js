"use strict";
/* features.js — loads last. Author's note / prefill wiring, per-message
   telemetry chips, command palette (Ctrl+K), keyboard shortcuts, and
   SillyTavern-style character-card (PNG) import. Reuses app.js globals. */

/* ---------- author's note (depth-targeted) ---------- */
$("an-toggle").addEventListener("click", () => {
  const box = $("an-edit");
  box.hidden = !box.hidden;
  $("an-toggle").setAttribute("aria-expanded", String(!box.hidden));
  if (!box.hidden && current) {
    $("an-text").value = current.authors_note || "";
    $("an-depth").value = current.authors_note_depth ?? 3;
    $("an-text").focus();
  }
});
let anSaving = null;
function saveAuthorsNote() {
  if (!current) return;
  const an = $("an-text").value;
  let depth = parseInt($("an-depth").value, 10);
  if (Number.isNaN(depth)) depth = 3;
  if (an === (current.authors_note || "") &&
      depth === (current.authors_note_depth ?? 3)) return;
  anSaving = api("POST", "/api/sessions/" + current.id,
                 {authors_note: an, authors_note_depth: depth})
    .then((s) => { current = s; });
}
$("an-text").addEventListener("blur", saveAuthorsNote);
$("an-depth").addEventListener("change", saveAuthorsNote);

/* ---------- reply prefill ---------- */
$("prefill-toggle").addEventListener("click", () => {
  const row = $("prefill-row");
  row.hidden = !row.hidden;
  $("prefill-toggle").classList.toggle("on", !row.hidden);
  if (!row.hidden) {
    if (current) $("prefill-text").value = current.prefill || "";
    $("prefill-text").focus();
  }
});
$("prefill-clear").addEventListener("click", async () => {
  $("prefill-text").value = "";
  if (current) current = await api("POST", "/api/sessions/" + current.id,
                                   {prefill: ""});
  $("prefill-row").hidden = true;
  $("prefill-toggle").classList.remove("on");
});
$("prefill-text").addEventListener("blur", async () => {
  if (!current) return;
  const v = $("prefill-text").value;
  if (v === (current.prefill || "")) return;
  current = await api("POST", "/api/sessions/" + current.id, {prefill: v});
});

/* ---------- per-message telemetry chips ---------- */
// hook renderMessages: after it runs, decorate assistant messages that carry
// saved stats with a small info chip. We wrap the existing function.
const _origRenderMessages = window.renderMessages;
window.renderMessages = function () {
  _origRenderMessages();
  if (!current) return;
  // attach a telemetry chip to each assistant bubble that carries saved stats
  const stats = current.messages.filter((m) => m.role === "assistant");
  const bubbles = document.querySelectorAll("#log .bot");
  bubbles.forEach((b, i) => {
    const m = stats[i];
    if (!m || !m.stats) return;
    const chip = document.createElement("button");
    chip.className = "stat-chip";
    chip.type = "button";
    chip.textContent = "ⓘ " + (m.stats.tps || "?") + " tok/s";
    chip.title = [
      m.stats.tps ? m.stats.tps + " tok/s decode" : "",
      m.stats.tokens ? m.stats.tokens + " tokens generated" : "",
      m.stats.prompt_tokens ? m.stats.prompt_tokens + " prompt tokens" : "",
      m.stats.model ? "model: " + m.stats.model : "",
    ].filter(Boolean).join("\n");
    b.appendChild(chip);
  });
};

/* ---------- repo packer: dump a code folder into the prompt ---------- */
async function attachCodeFolder() {
  const folder = prompt("Path to a code folder — Rigma packs its text files "
                        + "(respecting .gitignore-style skips) into the prompt:");
  if (!folder || !folder.trim()) return;
  const note = $("in");
  const was = note.value;
  note.value = "packing " + folder.trim() + "…";
  note.disabled = true;
  try {
    const r = await api("POST", "/api/workspace/pack", {folder: folder.trim()});
    note.value = r.content + "\n\n" + was;
    const warn = r.truncated ? " (truncated to fit)" : "";
    $("in").placeholder = r.file_count + " files, " +
      Math.round(r.chars / 1000) + "K chars packed" + warn;
  } catch (e) {
    note.value = was;
    alert("Couldn't pack folder: " + e.message);
  } finally {
    note.disabled = false;
    note.focus();
  }
}

/* ---------- command palette (Ctrl/Cmd+K) ---------- */
let palItems = [], palIdx = 0;
function paletteActions() {
  const acts = [
    {label: "New chat", run: () => newChat()},
    {label: "Settings — Chat params", run: () => toggleDrawer("chat")},
    {label: "Models — manage & download", run: () => openModelsView()},
    {label: "Server — engine room", run: () => toggleDrawer("server")},
    {label: "Compact this chat", run: () => current && compactChat(6)},
    {label: "Export chat as markdown",
     run: () => current && window.open("/api/sessions/" + current.id +
                                       "/export?fmt=md")},
    {label: "Toggle documents (RAG) for this chat", run: async () => {
      if (!current) return;
      current = await api("POST", "/api/sessions/" + current.id,
                          {use_rag: !current.use_rag});
      document.body.classList.toggle("rag-on", current.use_rag);
      $("use-rag").checked = current.use_rag; refreshDocs();
    }},
    {label: "Attach a code folder to the prompt", run: attachCodeFolder},
  ];
  return acts;
}
async function openPalette() {
  $("palette").hidden = false;
  const input = $("palette-input");
  input.value = "";
  input.focus();
  await renderPalette("");
}
function closePalette() { $("palette").hidden = true; }
async function renderPalette(q) {
  const ql = q.trim().toLowerCase();
  const acts = paletteActions().filter((a) => a.label.toLowerCase().includes(ql));
  let chats = [];
  if (ql.length >= 2) {
    try {
      const hits = await api("GET", "/api/sessions/search?q=" +
                             encodeURIComponent(q));
      chats = hits.slice(0, 6).map((h) => ({
        label: "Chat: " + (h.title || "(untitled)"),
        run: () => openSession(h.id)}));
    } catch {}
  }
  palItems = acts.concat(chats);
  palIdx = 0;
  const list = $("palette-list");
  list.innerHTML = "";
  palItems.forEach((it, i) => {
    const row = document.createElement("div");
    row.className = "palette-row" + (i === 0 ? " sel" : "");
    row.textContent = it.label;
    row.onclick = () => { closePalette(); it.run(); };
    list.appendChild(row);
  });
}
function movePalette(d) {
  const rows = $("palette-list").children;
  if (!rows.length) return;
  rows[palIdx] && rows[palIdx].classList.remove("sel");
  palIdx = (palIdx + d + rows.length) % rows.length;
  rows[palIdx].classList.add("sel");
  rows[palIdx].scrollIntoView({block: "nearest"});
}
$("palette-input").addEventListener("input", (e) => renderPalette(e.target.value));
$("palette-input").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { e.preventDefault(); movePalette(1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); movePalette(-1); }
  else if (e.key === "Enter") {
    e.preventDefault();
    const it = palItems[palIdx];
    if (it) { closePalette(); it.run(); }
  } else if (e.key === "Escape") { closePalette(); }
});
$("palette").addEventListener("click", (e) => {
  if (e.target.id === "palette") closePalette();
});

/* ---------- global keyboard shortcuts ---------- */
document.addEventListener("keydown", (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); }
  else if (mod && e.key.toLowerCase() === "n") { e.preventDefault(); newChat(); }
  else if (mod && e.key === "/") { e.preventDefault(); $("in").focus(); }
});

/* ---------- character-card (PNG) import -> preset ---------- */
// SillyTavern V2/V3 cards embed a base64 JSON in a PNG 'chara' tEXt chunk.
function readPngText(buf) {
  const dv = new DataView(buf);
  if (dv.byteLength < 8 || dv.getUint32(0) !== 0x89504e47) return null;
  let off = 8;
  const out = {};
  while (off + 12 <= dv.byteLength) {   // need length(4)+type(4)+crc(4)
    const len = dv.getUint32(off);
    if (len < 0 || off + 12 + len > dv.byteLength) break;   // truncated chunk
    const type = String.fromCharCode(dv.getUint8(off + 4), dv.getUint8(off + 5),
                                     dv.getUint8(off + 6), dv.getUint8(off + 7));
    if (type === "tEXt") {
      const bytes = new Uint8Array(buf, off + 8, len);
      let sep = bytes.indexOf(0);
      if (sep === -1) sep = bytes.length;
      const key = new TextDecoder().decode(bytes.slice(0, sep));
      const val = new TextDecoder().decode(bytes.slice(sep + 1));
      out[key] = val;
    }
    if (type === "IEND") break;
    off += 12 + len;
  }
  return out;
}
function decodeCharacterCard(buf) {
  let chunks;
  try { chunks = readPngText(buf); } catch { return null; }
  if (!chunks) return null;
  const raw = chunks.chara || chunks.ccv3;
  if (!raw) return null;
  let json;
  try { json = JSON.parse(decodeURIComponent(escape(atob(raw)))); }
  catch { try { json = JSON.parse(atob(raw)); } catch { return null; } }
  const d = json.data || json;   // V2 wraps under .data
  if (!d || !(d.name || d.first_mes || d.description)) return null;
  const persona = [d.description, d.personality, d.scenario]
    .filter(Boolean).join("\n\n");
  return {name: d.name || "Character",
          system_prompt: persona || ("You are " + (d.name || "a character") + "."),
          greeting: d.first_mes || ""};
}
async function importCharacterCard(file) {
  let card = null;
  try { card = decodeCharacterCard(await file.arrayBuffer()); }
  catch { card = null; }
  if (!card) { alert("That PNG isn't a character card (no embedded persona)."); return; }
  try {
    await api("POST", "/api/presets",
              {name: card.name, system_prompt: card.system_prompt,
               greeting: card.greeting, params: {}});
    if (typeof loadPresets === "function") await loadPresets();
    alert('Imported "' + card.name + '" as a preset — pick it from the sys bar.');
  } catch (e) { alert("Import failed: " + e.message); }
}
// global drop: a PNG anywhere (outside the model drop-zone) imports a card
document.addEventListener("dragover", (e) => {
  if (e.dataTransfer && [...e.dataTransfer.types].includes("Files"))
    e.preventDefault();
});
document.addEventListener("drop", (e) => {
  if (e.target.closest && e.target.closest(".drop-zone")) return;  // model install
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && /\.png$/i.test(f.name)) { e.preventDefault(); importCharacterCard(f); }
});
