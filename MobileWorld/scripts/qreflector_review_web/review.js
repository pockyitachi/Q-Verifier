"use strict";

const data = window.REVIEW_DATA;
const cases = data.cases;
const byId = new Map(cases.map(c => [c.id, c]));
const fields = ["goal", "action_type", "action_description", "action_arguments", "precondition", "postcondition", "reason", "answer", "notes"];
const englishFields = ["goal", "action_description", "precondition", "postcondition", "reason"];
const storageKey = `qreflector-review:${data.dataset_id}`;
const $ = id => document.getElementById(id);
let reviews = {}, current = byId.has(location.hash.slice(1)) ? location.hash.slice(1) : cases[0].id;
let legacyBackup = null;
let serverAvailable = false, localAvailable = true, draining = false, saveTimer, toastTimer;
const pending = new Map();

function toast(message) {
  $("toast").textContent = message; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4000);
}
function defaultFields(c) {
  return {...c.original, action_arguments: JSON.stringify(c.original.action_arguments, null, 2)};
}
function draft(c) { return reviews[c.id] || {status: "draft", fields: defaultFields(c)}; }
function status(id) { return reviews[id]?.status || "draft"; }
function statusText(value) { return ({draft: "待复核", done: "已确认", uncertain: "待补证据"})[value]; }
function recordIsValid(value) {
  return value && ["draft", "done", "uncertain"].includes(value.status) && value.fields &&
    Object.keys(value.fields).length === fields.length && fields.every(k => typeof value.fields[k] === "string");
}
function timestamp(record) { return Date.parse(record?.updated_at || "") || 0; }
function convertBundle(value) {
  if (value?.schema_version !== 1 || !value.reviews || typeof value.reviews !== "object" || Array.isArray(value.reviews)) throw new Error("无效的复核文件。");
  if (value.dataset_id === data.dataset_id) return {reviews: value.reviews, legacy: false};
  if (!data.legacy_import || value.dataset_id !== data.legacy_import.dataset_id) throw new Error("文件不属于当前 dataset 或其旧版本，无法导入。");
  const migrated = {};
  for (const [id, record] of Object.entries(value.reviews)) {
    if (!data.legacy_import.case_ids.includes(id) || !record || !["draft", "done", "uncertain"].includes(record.status) || !record.fields ||
        Object.keys(record.fields).length !== fields.length + 1 || ![...fields, "subgoal"].every(k => typeof record.fields[k] === "string")) throw new Error(`无效的旧版记录：${id}`);
    if (!byId.has(id)) continue; // Archived train-2 remains in the legacy backup.
    migrated[id] = {status: record.status === "done" ? "draft" : record.status,
      fields: Object.fromEntries(fields.map(k => [k, record.fields[k]])), updated_at: record.updated_at};
  }
  return {reviews: migrated, legacy: true};
}
function rememberLegacy(value) {
  if (!value || value.dataset_id !== data.legacy_import?.dataset_id) return;
  legacyBackup ||= {schema_version: 1, dataset_id: value.dataset_id, reviews: {}};
  for (const [id, record] of Object.entries(value.reviews || {})) if (!legacyBackup.reviews[id] || timestamp(record) >= timestamp(legacyBackup.reviews[id])) legacyBackup.reviews[id] = record;
}
function writeLocal() {
  try {
    localStorage.setItem(storageKey, JSON.stringify({schema_version: 1, dataset_id: data.dataset_id, reviews, legacy_backup: legacyBackup}));
    localAvailable = true;
  } catch (_) { localAvailable = false; }
}
function saveMessage() {
  const count = pending.size;
  if (draining || count) $("save-indicator").textContent = serverAvailable ? "正在保存…" : localAvailable ? "已保存到此浏览器 · 服务器未同步，可导出备份" : "尚未保存，请立即导出备份";
  else $("save-indicator").textContent = serverAvailable ? "自动保存到复核文件" : localAvailable ? "自动保存到此浏览器 · 可导出 JSON 备份" : "浏览器无法保存，请导出 JSON 备份";
  const time = reviews[current]?.updated_at;
  $("saved-time").textContent = time ? new Date(time).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "";
}
async function drain() {
  if (draining || !serverAvailable || !pending.size) { saveMessage(); return; }
  draining = true; saveMessage();
  while (pending.size) {
    const [id, snapshot] = pending.entries().next().value;
    pending.delete(id);
    try {
      const response = await fetch(`/api/reviews/${id}`, {method: "PUT", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({dataset_id: data.dataset_id, review: snapshot}), signal: AbortSignal.timeout(5000)});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "保存失败");
      if (reviews[id]?.updated_at === snapshot.updated_at) {
        reviews[id] = result.review; writeLocal();
      }
    } catch (error) {
      if (!pending.has(id)) pending.set(id, snapshot);
      serverAvailable = false;
      toast(`服务器保存未完成，草稿仍在浏览器中。${error.message}`);
      break;
    }
  }
  draining = false; saveMessage();
}
function saveCurrent(nextStatus = "draft") {
  const value = Object.fromEntries(fields.map(key => [key, $(`field-${key}`).value]));
  const record = {status: nextStatus, fields: value, updated_at: new Date().toISOString()};
  reviews[current] = record; writeLocal(); pending.set(current, structuredClone(record));
  clearTimeout(saveTimer); saveTimer = setTimeout(drain, 400);
  renderList(); renderStatus(); saveMessage();
}
function renderStatus() {
  const s = status(current); $("case-status").textContent = statusText(s);
  $("case-status").className = `badge ${s}`;
  const done = cases.filter(c => status(c.id) === "done").length;
  const unsure = cases.filter(c => status(c.id) === "uncertain").length;
  $("progress-text").textContent = `已确认 ${done} / ${cases.length}${unsure ? ` · 待补证据 ${unsure}` : ""}`;
}
function filteredCases() {
  const filter = $("status-filter").value;
  return cases.filter(c => filter === "all" || (["train", "test"].includes(filter) ? c.split === filter : filter === "pending" ? status(c.id) !== "done" : status(c.id) === filter));
}
function renderList() {
  const container = $("case-list"); container.replaceChildren();
  let group;
  for (const c of filteredCases()) {
    if (group !== c.split) {
      const title = document.createElement("div"); title.className = "list-group";
      title.textContent = c.split === "train" ? "训练集 · 优先复核" : "测试集";
      container.append(title); group = c.split;
    }
    const button = document.createElement("button"); button.className = `case-link${c.id === current ? " active" : ""}`;
    button.dataset.caseId = c.id; button.setAttribute("aria-label", `${c.number}. ${c.id} ${statusText(status(c.id))}`);
    if (c.id === current) button.setAttribute("aria-current", "true");
    const ordinal = document.createElement("span"); ordinal.className = "case-number"; ordinal.textContent = String(c.number).padStart(2, "0");
    const content = document.createElement("span"); const strong = document.createElement("strong"); strong.textContent = `${c.split} · ${c.line}`;
    const small = document.createElement("small"); small.textContent = c.missing_action ? "Action 缺失" : c.title.slice(0, 11);
    content.append(strong, small); const dot = document.createElement("span"); dot.className = `status-dot ${status(c.id)}`;
    button.append(ordinal, content, dot); button.addEventListener("click", () => openCase(c.id)); container.append(button);
  }
  if (!container.children.length) { const p = document.createElement("p"); p.className = "small muted"; p.textContent = "此筛选下暂无记录"; container.append(p); }
}
function renderHistory(c) {
  $("history-count").textContent = `${c.history.length} 步`;
  const container = $("history-list"); container.replaceChildren();
  if (!c.history.length) { const p = document.createElement("p"); p.className = "small muted"; p.textContent = "这条样本没有保存历史片段。"; container.append(p); return; }
  for (const h of c.history) {
    const section = document.createElement("article"); section.className = "history-step";
    const heading = document.createElement("strong"); heading.textContent = `Step ${h.step} · ${h.action.type}`;
    const description = document.createElement("p"); description.lang = "en"; description.textContent = h.action.description;
    const args = document.createElement("pre"); args.textContent = JSON.stringify(h.action.arguments, null, 2);
    const details = document.createElement("details"), summary = document.createElement("summary"), raw = document.createElement("pre");
    summary.textContent = "查看这一步的完整历史记录（旧标注）"; raw.textContent = JSON.stringify(h.raw, null, 2);
    details.append(summary, raw); section.append(heading, description, args, details); container.append(section);
  }
}
function openCase(id, scroll = true) {
  if (!byId.has(id)) return;
  clearTimeout(saveTimer); void drain();
  current = id; const c = byId.get(id), d = draft(c);
  history.replaceState(null, "", `#${id}`);
  $("case-kicker").textContent = `CASE ${String(c.number).padStart(2, "0")} / ${cases.length}   ·   ${c.split} ${c.line}${c.split === "train" ? " · 训练集优先" : ""}`;
  $("case-title").textContent = c.title; $("case-question").textContent = c.question;
  $("goal-text").textContent = c.original.goal;
  $("action-description").textContent = c.original.action_description;
  $("action-code").textContent = `${c.original.action_type} ${JSON.stringify(c.original.action_arguments)}`;
  $("missing-action-badge").hidden = !c.missing_action;
  for (const key of fields) $(`field-${key}`).value = d.fields[key];
  for (const badge of document.querySelectorAll("[data-attention]")) badge.hidden = !c.attention_fields.includes(badge.dataset.attention);
  const needsContext = c.attention_fields.some(k => ["goal", "action_type", "action_arguments", "action_description"].includes(k));
  $("context-editor").open = needsContext; $("context-attention").hidden = !needsContext;
  $("form-errors").hidden = true; $("image-error").hidden = true;
  $("before-image").src = c.images.before; $("after-image").src = c.images.after;
  $("source-record").textContent = JSON.stringify({case_id: c.id, task: c.task_key, source_issues: c.source_issues,
    current_v4_annotation: c.original, legacy_transition: c.legacy_transition}, null, 2);
  renderHistory(c); renderList(); renderStatus(); saveMessage();
  $("previous-button").disabled = c.number === 1; $("next-button").disabled = c.number === cases.length;
  if (scroll) window.scrollTo({top: 0, behavior: "instant"});
}
function showErrors(errors) {
  const box = $("form-errors"); box.replaceChildren();
  const list = document.createElement("ul"); for (const error of errors) { const li = document.createElement("li"); li.textContent = error; list.append(li); }
  box.append(list); box.hidden = false; box.scrollIntoView({block: "center", behavior: "smooth"});
}
function validate(record, c) {
  const errors = [], f = record.fields;
  for (const name of englishFields) {
    if (!f[name].trim()) errors.push(`${name} 不能为空。`);
    if (/[\u3400-\u9fff]/u.test(f[name]) && f[name] !== c.original[name]) errors.push(`${name} 请用英文填写；中文说明可写在备注中。`);
  }
  if (!["Yes", "No"].includes(f.answer)) errors.push("请选择 Yes 或 No。无法判断时请使用“暂无法判断”。");
  if (["", "unknown", "none"].includes(f.action_type.trim().toLowerCase()) || f.action_description.trim().toLowerCase() === "none") errors.push("当前 action 仍缺失，请补齐类型和英文描述，或标记暂无法判断。");
  let args;
  try { args = JSON.parse(f.action_arguments); if (args !== null && (typeof args !== "object" || Array.isArray(args))) errors.push("Action arguments 应为 JSON 对象或 null。"); }
  catch (_) { errors.push("Action arguments 不是合法 JSON。"); }
  if (c.missing_action && args !== undefined) {
    const numeric = ({Tap:["x","y"], Long_Press:["x","y"], Swipe:["x1","y1","x2","y2"], Tap_and_Type:["x","y"], Tap_Type_and_Enter:["x","y"]})[f.action_type.trim()] || [];
    for (const key of numeric) if (typeof args?.[key] !== "number" || !Number.isFinite(args[key])) errors.push(`补充的 ${f.action_type} 需要数值参数 ${key}。`);
    if (["Type", "Tap_and_Type", "Tap_Type_and_Enter"].includes(f.action_type.trim()) && typeof args?.text !== "string") errors.push("补充的输入动作需要 text 参数。");
  }
  return errors;
}
function advance() {
  const filter = $("status-filter").value;
  const scope = ["train", "test"].includes(filter) ? cases.filter(c => c.split === filter) : cases;
  const index = scope.findIndex(c => c.id === current);
  const next = [...scope.slice(index+1), ...scope.slice(0,index)].find(c => status(c.id) !== "done");
  if (next) openCase(next.id); else toast(scope.length === cases.length ? "所有 case 均已确认。可以导出填写结果。" : "当前分组已确认，可以切换到其他数据继续复核。");
}
function zoom(phase) {
  $("dialog-title").textContent = `${current} · ${phase === "before" ? "Before" : "After"}`;
  $("large-image").src = byId.get(current).images[phase]; $("image-scroll").classList.remove("original-size");
  $("zoom-button").textContent = "原始尺寸"; $("image-dialog").showModal();
}
async function exportReviews() {
  await drain();
  const result = {schema_version: 1, dataset_id: data.dataset_id, dataset_version: data.dataset_version,
    source_sha256: data.source_sha256, exported_at: new Date().toISOString(), reviews, legacy_backup: legacyBackup,
    archived_cases: data.archived_cases,
    progress: {total: cases.length, confirmed: cases.filter(c => status(c.id) === "done").length, uncertain: cases.filter(c => status(c.id) === "uncertain").length}};
  const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type: "application/json;charset=utf-8"}));
  const a = document.createElement("a"); a.href = url; a.download = `qreflector-review-${new Date().toISOString().slice(0,10)}.json`;
  a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast("已导出当前填写结果，可随时导入继续复核。");
}
async function importReviews(file) {
  try {
    const value = JSON.parse(await file.text());
    const bundle = convertBundle(value);
    const entries = Object.entries(bundle.reviews);
    for (const [id, review] of entries) {
      if (!byId.has(id) || !recordIsValid(review)) throw new Error(`无效的 case 记录：${id}`);
      if (review.status === "done" && validate(review, byId.get(id)).length) throw new Error(`${id} 标记已确认，但字段校验未通过。`);
    }
    let count = 0;
    for (const [id, review] of entries) {
      if (!reviews[id] || timestamp(review) >= timestamp(reviews[id])) {
        reviews[id] = {...review, updated_at: new Date().toISOString()}; pending.set(id, structuredClone(reviews[id])); count++;
      }
    }
    rememberLegacy(bundle.legacy ? value : value.legacy_backup);
    writeLocal(); openCase(current, false); toast(bundle.legacy ? `已恢复旧版 ${count} 条草稿，请按 Task 重新确认；旧记录已备份。` : `已导入 ${count} 条记录；保留了本地较新的修改。`); void drain();
  } catch (error) { toast(error.message || "导入失败，请检查文件格式。"); }
}
async function initialize() {
  let migratedCount = 0;
  try {
    const cached = JSON.parse(localStorage.getItem(storageKey) || "null");
    if (cached?.dataset_id === data.dataset_id) for (const [id, r] of Object.entries(cached.reviews || {})) if (byId.has(id) && recordIsValid(r)) reviews[id] = r;
    rememberLegacy(cached?.legacy_backup);
    if (data.legacy_import) {
      const old = JSON.parse(localStorage.getItem(`qreflector-review:${data.legacy_import.dataset_id}`) || "null");
      if (old) {
        const bundle = convertBundle(old); rememberLegacy(old);
        for (const [id, r] of Object.entries(bundle.reviews)) if (!reviews[id] || timestamp(r) > timestamp(reviews[id])) { reviews[id] = r; migratedCount++; }
      }
    }
  } catch (_) { localAvailable = false; }
  if (["http:", "https:"].includes(location.protocol)) {
    try {
      const response = await fetch("/api/reviews", {signal: AbortSignal.timeout(3500)}), saved = await response.json();
      if (!response.ok || saved.dataset_id !== data.dataset_id) throw new Error("Unrecognized review server");
      serverAvailable = true;
      rememberLegacy(saved.legacy_backup);
      for (const [id, r] of Object.entries(saved.reviews || {})) if (byId.has(id) && recordIsValid(r) && (!reviews[id] || timestamp(r) >= timestamp(reviews[id]))) reviews[id] = r;
      for (const [id, r] of Object.entries(reviews)) if (!saved.reviews?.[id] || timestamp(r) > timestamp(saved.reviews[id])) pending.set(id, structuredClone(r));
    } catch (_) { serverAvailable = false; }
  }
  $("review-form").addEventListener("input", () => { $("form-errors").hidden = true; saveCurrent("draft"); });
  $("review-form").addEventListener("submit", event => {
    event.preventDefault(); saveCurrent("draft"); const errors = validate(reviews[current], byId.get(current));
    if (errors.length) { showErrors(errors); return; }
    saveCurrent("done"); toast(`${current} 已确认`); advance(); void drain();
  });
  $("uncertain-button").addEventListener("click", () => { saveCurrent("uncertain"); toast(`${current} 已标记待补证据`); advance(); void drain(); });
  $("previous-button").addEventListener("click", () => { const i = cases.findIndex(c => c.id === current); if (i > 0) openCase(cases[i-1].id); });
  $("next-button").addEventListener("click", () => { const i = cases.findIndex(c => c.id === current); if (i < cases.length-1) openCase(cases[i+1].id); });
  $("status-filter").addEventListener("change", () => { const list = filteredCases(); if (list.length && !list.some(c => c.id === current)) openCase(list[0].id); else renderList(); });
  $("before-button").addEventListener("click", () => zoom("before")); $("after-button").addEventListener("click", () => zoom("after"));
  for (const phase of ["before", "after"]) $(`${phase}-image`).addEventListener("error", () => { $("image-error").hidden = false; });
  $("close-image").addEventListener("click", () => $("image-dialog").close());
  $("zoom-button").addEventListener("click", () => { const original = $("image-scroll").classList.toggle("original-size"); $("zoom-button").textContent = original ? "适应窗口" : "原始尺寸"; });
  $("export-button").addEventListener("click", exportReviews);
  $("import-button").addEventListener("click", () => $("import-file").click());
  $("import-file").addEventListener("change", event => { const file = event.target.files[0]; if (file) void importReviews(file); event.target.value = ""; });
  window.addEventListener("hashchange", () => { const id = location.hash.slice(1); if (byId.has(id)) openCase(id); });
  window.addEventListener("beforeunload", () => { writeLocal(); });
  document.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); saveCurrent(status(current)); void drain(); toast("已保存当前草稿"); } });
  writeLocal(); openCase(current, false); void drain();
  if (migratedCount) toast(`已恢复旧版 ${migratedCount} 条草稿，请按 Task 重新确认。`);
}
void initialize();
