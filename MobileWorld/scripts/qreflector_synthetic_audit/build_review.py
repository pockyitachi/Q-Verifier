"""Make an isolated review page for 13 quarantined and 5 relabelled examples."""
from __future__ import annotations

import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from convert_qreflector_condition_generation import parse_source, normalize_action
from repair_qreflector_condition_semantics import context_of
from build import ROOT, SOURCE, OUTPUT, read, dump, sha

DEST = ROOT / '../review/condition_generation_v5_synthetic_audit'

QUESTIONS = {
    679: "画面仍在 Cantor Arts Center，但照片从 3/319 变为 5/319。需要判断是自动轮播、误触，还是有助于挑选博物馆的浏览。缺少依据可继续隔离。",
    698: "点击后打开 Cantor Arts Center 相册。动作描述说返回，但浏览博物馆照片可能有助于行程规划，不能只因描述不一致就判 No。",
    707: "点击后显示博物馆评分与免费接驳车信息，可能有助于规划。需要核对实际点击意图与轨迹，区分有效浏览和误触。",
    713: "打开的是 6:00 PM 编辑器，6:00 AM 仍未启用。编辑 PM 闹钟也可能用于改成 AM；需要判断是否构成有效中间步骤。",
    748: "Swipe 让应用抽屉略微滚动，Chrome 仍可见。描述写打开 Chrome；需要判断是有效定位还是无效尝试。",
    797: "没有清空查询，而是选中了 My store & nearby stores。任务没有门店偏好，不能仅凭描述不符确定这个搜索范围调整无用。",
    834: "没有提交酒店搜索，而是打开房间／人数选择。它可能是参数核对，也可能是误触；当前截图不足以确认任务进展。",
    844: "Search this area 后结果没变。成功刷新也可能得到相同结果，需原始请求、加载状态或执行日志；不要仅凭图像相同判 No。",
    853: "打开的是房间／人数选择，未提交搜索。酒店搜索参数相关，但是否需要调整不明确；无额外依据可继续隔离。",
    860: "打开蛋糕店菜单而非电话号码。菜单可帮助检查 birthday cakes 条件，未显示号码不等于这一步完全无用。",
    876: "前后图片字节相同，但刷新成功后也可能不变。需执行或请求完成证据，否则继续隔离。",
    877: "键盘已打开，但看不到输入框、查询文字或结果。需要确认焦点是否进入 Google 搜索，区分部分进展和失败。",
    881: "打开 Select stores 并启用门店过滤，没有打开查询编辑器。需要判断是有用的门店设置还是不必要的限制。",
}

def main():
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite {DEST}")
    audit = read(OUTPUT / "synthetic_audit.jsonl")
    selected = sorted([r for r in audit if r["status"] != "negative"],
                      key=lambda r: (r["status"] != "quarantine", r["source_line"]))
    rows, refs = read(SOURCE / "train.jsonl"), read(SOURCE / "train.references.jsonl")
    DEST.mkdir()
    (DEST / "case_images").mkdir()
    cases, saved = [], {}
    for a in selected:
        line, sid = a["source_line"], a["sample_id"]
        row, ref = rows[line - 1], refs[line - 1]
        ctx = context_of(row)
        _, _, transition, history, _, _ = parse_source(ref["original_sample"])
        is_quarantined = a["status"] == "quarantine"
        t = a["replacement_target"] or {**a["original_target"], "reason": "", "answer": ""}
        original = {"goal": ctx["goal"], "action_type": ctx["action"]["type"],
                    "action_description": ctx["action"]["description"],
                    "action_arguments": ctx["action"]["arguments"], **t,
                    "notes": a["decision_evidence"]}
        urls = {}
        for phase, p in zip(("before", "after"), ref["converted_images"]):
            path = (SOURCE / p).resolve(strict=True)
            link = DEST / "case_images" / f"{sid}-{phase}{path.suffix}"
            link.symlink_to(os.path.relpath(path, link.parent))
            urls[phase] = "case_images/" + link.name
        cases.append({"id": sid, "split": "train", "line": line, "number": len(cases) + 1,
            "title": ("已隔离 · 需要补证据" if is_quarantined else "已修正 · No → Yes"),
            "question": (QUESTIONS[line] + " 下方旧 pre/post 仅供复核，标签已留空，不进入训练。" if is_quarantined else
                "这条已按有效中间进展改为 Yes，写入 v5。这里展示新标注供查看；无需为了使用 v5 再次确认。"),
            "attention_fields": ["precondition", "postcondition", "answer", "reason"] if is_quarantined else [],
            "missing_action": False, "original": original, "images": urls,
            "task_key": Path(ref["source_images"][0]).stem,
            "source_images": ref["source_images"], "source_issues": [a["decision_evidence"]],
            "legacy_transition": transition, "audit": a,
            "history": [{"step": h["step"], "action": normalize_action(h["action"]), "raw": h}
                        for h in history],
            "provenance": {"dataset_directory": str(OUTPUT), "source_line": line,
                           "output_train_line": a["output_train_line"]}})
        if is_quarantined:
            saved[sid] = {"status": "uncertain", "fields": {
                **original, "action_arguments": json.dumps(original["action_arguments"], indent=2)},
                "updated_at": json.loads((OUTPUT / "metadata.json").read_text())["created_at_utc"]}
    dataset_id = "qreflector-v5-audit-" + sha(OUTPUT / "synthetic_audit.jsonl")[:16]
    manifest = {"schema_version": 1, "dataset_id": dataset_id,
        "dataset_version": OUTPUT.name, "source_directory": str(OUTPUT), "cases": cases,
        "review_scope": "13 quarantined cases plus 5 relabelled cases; original v4 review sidecar remains separate."}
    dump(DEST / "cases.json", manifest)
    (DEST / "cases.js").write_text("window.REVIEW_DATA = " + json.dumps(manifest, ensure_ascii=False) + ";\n")
    dump(DEST / "answers.json", {"schema_version": 1, "dataset_id": dataset_id, "reviews": saved})
    assets = ROOT / "scripts/qreflector_review_web"
    for name in ("review.css", "review.js"):
        shutil.copyfile(assets / name, DEST / name)
    html = (assets / "index.html").read_text()
    replacements = {
        "v4 · 29 条数据复核": "v5 · 合成样本审计",
        "已确认 0 / 29": "13 条隔离 · 5 条已修正",
        "全部 29 条": "全部 18 条",
        '<option value="train">先看训练集 · 3 条</option><option value="test">测试集 · 26 条</option>': "",
        "原 01（train-2）已归档。": "前 13 条隔离，后 5 条展示修正。",
        "需要你确认": "审计结论",
        "依据 Task 和当前一步复核，无需填写 Subgoal。前三条为训练数据；原始数据保持不变。":
            "235 条已审计：217 条保留 No、5 条改 Yes、13 条隔离。隔离样本未进入 v5 的 869 条训练数据；无新证据可保持隔离。这里保存的修改不会自动覆盖数据集。",
    }
    for old, new in replacements.items():
        assert old in html
        html = html.replace(old, new)
    (DEST / "index.html").write_text(html)
    # Only a display label differs from the existing, tested review client.
    js = (DEST / "review.js").read_text().replace("current_v4_annotation:", "current_audit_annotation:")
    (DEST / "review.js").write_text(js)
    stats = Counter(k for r in audit for k in r["changes"])
    report = f"""Q-Reflector synthetic training audit — v5

Scope: all 235 synthetic training rows (stable v4 IDs train-648 to train-882).
Reviewed both screenshots, task, recorded action, target conditions and reasons.
Reviewer: assistant visual audit; this is not an independent expert certification.

Outcome
  217 retain No, with evidence-grounded target repairs.
  5 change No to Yes: train-678, train-743, train-781, train-813, train-845.
  13 are quarantined with no active target and excluded from train.jsonl.
  Changed target fields among retained synthetic rows: {dict(stats)}.

New training data: 869 rows = 647 original + 222 retained synthetic.
Labels: 522 Yes, 347 No.
The user's previously confirmed train-138 correction (Yes -> No) is also applied.
The confirmed train-155 review is preserved; it did not change the existing target.
No assistant draft was imported as a user-confirmed decision.

Why five labels changed
  train-678: Chrome opened to an existing tab; a fresh page is unnecessary.
  train-743: Settings search and keyboard opened, enabling the Data Saver search.
  train-781 and train-845: Switch_App opened Overview; app selection is a later step.
  train-813: the blocking store-selection sheet closed, enabling subsequent query editing.

What was repaired
  Ground preconditions in named controls and visible before states.
  Keep postconditions local to the current action; do not require downstream work.
  Replace unobserved causes (lag, missed registration, touch pressure) with visible facts.
  Correct claims of unchanged screens where a different app, panel, or switch actually changed.
  Preserve recorded actions, including erroneous parameters and descriptions, as model input.
  Keep uncertain intent/progress cases and unobservable refresh outcomes out of training.

Provenance and checks
  All 235 map to training_data_corruption_1 by both image hashes, task and action.
  No goal+action match was found in the compared generic image-swap generator JSONL outputs.
  No mutation log was found; the exact corruption operation is NOT reconstructed.
  All 235 source screenshots are 1440 x 3120.
  50 actions contain coordinate value 2399; 49 contain 1079. This is a diagnostic pattern,
  not proof of a particular generator, scaling defect or image-stitching operation.
  13 pairs have byte-identical before/after images; identity alone was not used to assign No.
  No duplicate complete image pairs in the synthetic subset.
  No exact image-byte overlap with original training rows, validation, or test images.
  These checks do not rule out near duplicates or related trajectories.

Preservation
  Original v4 and earlier datasets are preserved.
  Validation 71 and test 1082: data, inputs, references byte-identical to v4.
  Screenshots linked through directory symlinks; zero image copies or source pixel changes.
  The v4 review answers were snapshotted; user-confirmed answers remain separate.
  Stable sample IDs refer to v4 rows; see train.line_map.jsonl for new physical line numbers.

Remaining limits
  These examples cover 27 tasks and many repeated app-launch/gesture failures. The audit
  repairs labels and explanations; it does not establish hard-negative diversity.
  The other 647 training examples were not exhaustively visually reaudited here.
  26 inherited test-condition/action ambiguities remain; test files were not changed.
  No training, offline evaluation, MobileWorld experiment, or GPU operation was started.

Review page: http://localhost:8766/
Original review page: http://localhost:8765/

Quarantined cases
"""
    for a in selected:
        if a["status"] == "quarantine":
            report += f"  {a['sample_id']}: {a['decision_evidence']}\n"
    (OUTPUT / "AUDIT_REPORT.txt").write_text(report)
    # Register the additional report without changing prior data fingerprints.
    metadata = json.loads((OUTPUT / "metadata.json").read_text())
    metadata["output_sha256"]["AUDIT_REPORT.txt"] = sha(OUTPUT / "AUDIT_REPORT.txt")
    metadata["review_directory"] = str(DEST)
    dump(OUTPUT / "metadata.json", metadata)
    print(json.dumps({"review_directory": str(DEST), "cases": len(cases),
                      "quarantined_blank_labels": len(saved), "images_copied": 0,
                      "dataset_id": dataset_id}, indent=2))

if __name__ == "__main__":
    main()
