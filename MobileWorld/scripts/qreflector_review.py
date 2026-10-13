"""Build and serve a small, local Q-Reflector annotation review page."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from convert_qreflector_condition_generation import parse_source, normalize_action
from repair_qreflector_condition_semantics import context_of

ASSETS = Path(__file__).with_name("qreflector_review_web")
GUIDANCE = {
    "train-2": ("搜索动作与旧 subgoal 不一致", "图中这一步是在输入搜索词，旧 subgoal 却写返回桌面。请确认应保留哪个 subgoal，并核对当前步的判断。", ["subgoal", "answer", "reason"]),
    "train-138": ("加载中的航班页面：排序入口是否存在", "before 仍在加载，动作指向系统状态栏。请看 after 是否真的打开应用内排序选项，再确认标签和解释。", ["answer", "reason"]),
    "train-155": ("切换 Clock 与计时器时长", "当前步是切换应用。请结合历史确认 pre/post 只覆盖这一步；旧注释的 3 分钟与任务的 3 小时不应变成这一步必须完成的结果。", ["precondition", "postcondition", "answer"]),
    "train-743": ("Location 与 Network & Internet", "动作描述与目标入口不一致。请确认这一步的局部意图和实际变化，补齐 pre/post 与判断依据。此条是合成负例，不要为了使动作成功而修正原错误参数。", ["precondition", "postcondition", "answer", "reason"]),
    "test-154": ("铅笔图标：编辑入口还是 OCR", "请识别铅笔图标实际打开的功能。post 只写这次点击的局部预期，不要求同时完成 OCR 或复制文字。", ["precondition", "postcondition", "answer", "reason"]),
    "test-163": ("Order by Date：排序设置", "请判断点这行文字会打开排序选项、切换排序，还是没有效果；不要要求本次点击已经完成重复费用整理。", ["precondition", "postcondition", "answer", "reason"]),
    "test-166": ("圆形箭头图标的实际用途", "动作对 OCR 功能只是猜测。请结合前后图确认图标用途和这次点击的结果，再补 pre/post。", ["precondition", "postcondition", "answer", "reason"]),
    "test-187": ("日期 / DESC 文字的点击效果", "请确认此处是日期范围、排序入口还是不可点击文字。只判断这一 tap 的局部结果。", ["precondition", "postcondition", "answer", "reason"]),
    "test-252": ("扩大搜索半径", "请核对 INCREASE SEARCH RADIUS 的局部作用。扩大搜索范围或重新搜索不等于已经添加地图标记。", ["precondition", "postcondition", "answer", "reason"]),
    "test-264": ("地图定位图标", "请识别左侧定位图标的作用。明确它应定位哪里，及 after 是否呈现了相应变化。", ["precondition", "postcondition", "answer", "reason"]),
    "test-285": ("地图十字准星按钮", "请确认准星按钮是定位、居中还是添加标记入口，不要用完成整项地点任务作为这个 tap 的 post。", ["precondition", "postcondition", "answer", "reason"]),
    "test-365": ("收据图像上的未知工具图标", "请识别被点击的工具和实际变化。动作描述猜测有 OCR，但需要按本次控件的真实作用补条件。", ["precondition", "postcondition", "answer", "reason"]),
    "test-491": ("地图右上角标记图标", "请确认右上角图标打开什么界面，以及是否形成了本步预期的有效进展。", ["precondition", "postcondition", "answer", "reason"]),
    "test-508": ("图片编辑入口", "请确认铅笔图标打开的编辑功能。开启编辑界面与完成文字提取需要分别描述。", ["precondition", "postcondition", "answer", "reason"]),
    "test-539": ("地图蓝色定位按钮", "请根据前后图确定蓝色按钮的用途。当前动作属于探索，不能预设它必然新增航点。", ["precondition", "postcondition", "answer", "reason"]),
    "test-577": ("路线界面的加号", "请识别加号属于地图缩放、航点操作还是其他菜单，并写清本次点击的局部效果。", ["precondition", "postcondition", "answer", "reason"]),
    "test-610": ("Sort by 与创建轨迹", "请确认 Sort by 打开的实际选项。排序入口不能仅因大目标是创建轨迹，就要求它创建轨迹。", ["precondition", "postcondition", "answer", "reason"]),
    "test-640": ("点击 Company 字段", "请确认这一步是否只获得 Company 输入焦点，还是确实露出了 Phone 字段。按当前点击补条件和判断。", ["precondition", "postcondition", "answer", "reason"]),
    "test-686": ("计时器数字 2 与当前输入状态", "任务要求 16 分 35 秒。请看当前键盘状态，判断单次按 2 实际做了什么；不要要求一步输入完整时间。", ["precondition", "postcondition", "answer", "reason"]),
    "test-714": ("图片信息图标与 OCR", "请确认 i 图标是否只是图片信息入口，以及它与本步预期的关系。不能把图片信息页直接当作 OCR 完成。", ["precondition", "postcondition", "answer", "reason"]),
    "test-930": ("0 tracks 区域的点击效果", "请确认此区域能打开的功能，或是否无变化。post 应对应这一步，而非完整建轨任务。", ["precondition", "postcondition", "answer", "reason"]),
    "test-1053": ("All recipes 分类与字母排序", "选择 All recipes 不一定改变排序。请核对实际变化，以及旧 reason / 标签是否误把分类切换当作排序完成。", ["postcondition", "answer", "reason"]),
}
MISSING = {"test-102", "test-220", "test-267", "test-329", "test-693", "test-801", "test-918", "test-1055"}
ENGLISH_FIELDS = ("goal", "action_description", "precondition", "postcondition", "reason")
REVIEW_FIELDS = ENGLISH_FIELDS + ("action_type", "action_arguments", "answer", "notes")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def build(source, output, previous_review=None):
    source, output = source.resolve(), output.resolve()
    metadata = json.loads((source / "metadata.json").read_text())
    assert metadata["dataset_version"] == "condition_generation_v4_no_subgoal"
    pending = {}
    for r in map(json.loads, (source / "review_required.jsonl").read_text().splitlines()):
        pending.setdefault((r["split"], r["line"]), []).append(r["issue"])
    assert len(pending) == 30
    # This case's sole flagged problem was the supplied subgoal. The user
    # reviewed its other fields in chat; v4 no longer supplies that field.
    pending.pop(("train", 2))
    hashes = {f"{s}.jsonl": sha(source / f"{s}.jsonl") for s in ("train", "val", "test")}
    hashes["review_required.jsonl"] = sha(source / "review_required.jsonl")
    dataset_id = "qreflector-v4-" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    output.mkdir(parents=True, exist_ok=True)
    old_manifest = output / "cases.json"
    if old_manifest.exists():
        assert json.loads(old_manifest.read_text())["dataset_id"] == dataset_id, "Use a new review folder for a different dataset"
    images = output / "case_images"
    images.mkdir(exist_ok=True)
    cases = []
    for split in ("train", "val", "test"):
        rows = list(map(json.loads, (source / f"{split}.jsonl").read_text().splitlines()))
        refs = list(map(json.loads, (source / f"{split}.references.jsonl").read_text().splitlines()))
        for line, (row, ref) in enumerate(zip(rows, refs), 1):
            if (split, line) not in pending:
                continue
            case_id = f"{split}-{line}"
            context = context_of(row)
            target = json.loads(row["messages"][1]["content"][0]["text"])
            _, _, transition, history, _, _ = parse_source(ref["original_sample"])
            urls = []
            for phase, path in zip(("before", "after"), ref["converted_images"]):
                original = (source / path).resolve(strict=True)
                link = images / f"{case_id}-{phase}{original.suffix}"
                if link.is_symlink():
                    assert link.resolve() == original
                else:
                    assert not link.exists()
                    link.symlink_to(os.path.relpath(original, images))
                urls.append("case_images/" + link.name)
            if case_id in MISSING:
                title = "当前 action 缺失"
                question = "请根据原始轨迹补充实际执行的 action 类型、参数和意图，再核对 pre/post 与判断。仅凭图片无法确定动作时，请标记“暂无法判断”，并在备注写明缺少的证据。"
                required = ["action_type", "action_arguments", "action_description", "precondition", "postcondition", "answer", "reason"]
            else:
                title, question, required = GUIDANCE[case_id]
            original = {
                "goal": context["goal"],
                "action_type": context["action"]["type"],
                "action_description": context["action"]["description"],
                "action_arguments": context["action"]["arguments"],
                **target, "notes": "",
            }
            cases.append({
                "id": case_id, "split": split, "line": line, "number": len(cases)+1,
                "title": title, "question": question, "attention_fields": required,
                "missing_action": case_id in MISSING, "original": original,
                "images": {"before": urls[0], "after": urls[1]},
                "task_key": Path(ref["source_images"][0]).stem.removesuffix("_before"),
                "source_images": ref["source_images"],
                "history": [{"step": h["step"], "action": normalize_action(h["action"]),
                             "before_description": h["from_state"].get("description", ""),
                             "after_description": h["to_state"].get("description", ""),
                             "raw": h} for h in history],
                "source_issues": pending[(split, line)],
                "legacy_transition": transition,
                "provenance": {"dataset_directory": str(source), "source_line": line,
                               "source_sha256": hashes[f"{split}.jsonl"]},
            })
    assert len(cases) == 29 and sum(c["missing_action"] for c in cases) == 8
    manifest = {"schema_version": 1, "dataset_id": dataset_id, "dataset_version": metadata["dataset_version"],
                "created_at": utc(), "source_directory": str(source), "source_sha256": hashes, "cases": cases}
    manifest["archived_cases"] = [{"id": "train-2", "previous_number": 1,
        "reason": "The user reviewed the remaining fields. The sole flagged supplied-subgoal mismatch is no longer applicable to the v4 input contract; labels and original datasets were not changed."}]
    if previous_review is not None:
        previous_review = previous_review.resolve()
        old = json.loads((previous_review / "cases.json").read_text())
        assert Path(old["source_directory"]).resolve() == Path(metadata["previous_version_directory"]).resolve()
        old_cases = {c["id"]: c for c in old["cases"]}
        for case in cases:
            # Automatic migration is allowed only when all retained fields are
            # identical. A future annotation change must explicitly reconcile it.
            assert case["original"] == {k: v for k, v in old_cases[case["id"]]["original"].items() if k != "subgoal"}
        manifest["legacy_import"] = {"dataset_id": old["dataset_id"],
            "case_ids": list(old_cases), "source_review_directory": str(previous_review)}
    (output / "cases.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output / "cases.js").write_text("window.REVIEW_DATA = " + json.dumps(manifest, ensure_ascii=False) + ";\n")
    for name in ("index.html", "review.css", "review.js"):
        (output / name).write_bytes((ASSETS / name).read_bytes())
    assert all(sha(source / name) == digest for name, digest in hashes.items())
    print(json.dumps({"review_directory": str(output), "cases": len(cases), "image_links": len(list(images.iterdir())),
                      "copied_images": 0, "history_entries": sum(len(c["history"]) for c in cases), "dataset_id": dataset_id}))


def migrate_answers(root):
    """Carry compatible old answers into the new sidecar, leaving the old intact."""
    root = root.resolve()
    manifest = json.loads((root / "cases.json").read_text())
    legacy = manifest.get("legacy_import")
    if not legacy or (root / "answers.json").exists():
        return
    old_path = Path(legacy["source_review_directory"]) / "answers.json"
    if not old_path.exists():
        return
    old = json.loads(old_path.read_text())
    assert old["dataset_id"] == legacy["dataset_id"]
    if not old.get("reviews"):
        return
    cases = {c["id"]: c for c in manifest["cases"]}
    reviews = {}
    for case_id, value in old["reviews"].items():
        if case_id not in cases:
            continue  # Still preserved in the legacy backup and old answer file.
        migrated = {"status": "draft" if value["status"] == "done" else value["status"],
                    "fields": {k: value["fields"][k] for k in REVIEW_FIELDS}}
        checked = validate_review(migrated, cases[case_id])
        checked["updated_at"] = value.get("updated_at", checked["updated_at"])
        reviews[case_id] = checked
    answer = {"schema_version": 1, "dataset_id": manifest["dataset_id"], "reviews": reviews,
              "legacy_backup": old, "updated_at": utc()}
    with (root / "answers.json").open("x") as stream:
        json.dump(answer, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def validate_review(value, case):
    if not isinstance(value, dict) or value.get("status") not in {"draft", "done", "uncertain"}:
        raise ValueError("Invalid review status")
    if not isinstance(value.get("fields"), dict) or set(value["fields"]) != set(REVIEW_FIELDS):
        raise ValueError("Review fields are incomplete")
    f = value["fields"]
    if not all(isinstance(f[key], str) for key in REVIEW_FIELDS):
        raise ValueError("Review fields must be strings")
    if any(len(v) > 100000 for v in f.values()):
        raise ValueError("A review field is too long")
    if value["status"] == "done":
        for field in ENGLISH_FIELDS:
            if not f[field].strip():
                raise ValueError(f"{field} is required")
            if re.search(r"[\u3400-\u9fff]", f[field]) and f[field] != case["original"][field]:
                raise ValueError(f"{field} must be written in English")
        if f["answer"] not in {"Yes", "No"}:
            raise ValueError("Choose Yes or No before confirming")
        if f["action_type"].strip().lower() in {"", "unknown", "none"} or f["action_description"].strip().lower() == "none":
            raise ValueError("The current action is still missing")
        try:
            args = json.loads(f["action_arguments"])
        except ValueError:
            raise ValueError("Action arguments must be valid JSON") from None
        if args is not None and not isinstance(args, dict):
            raise ValueError("Action arguments must be a JSON object or null")
        if case["missing_action"]:
            kind = f["action_type"].strip()
            numeric = {"Tap": ("x", "y"), "Long_Press": ("x", "y"), "Swipe": ("x1", "y1", "x2", "y2"),
                       "Tap_and_Type": ("x", "y"), "Tap_Type_and_Enter": ("x", "y")}.get(kind, ())
            for key in numeric:
                number = (args or {}).get(key)
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                    raise ValueError(f"Recovered {kind} action needs numeric {key}")
            if kind in {"Type", "Tap_and_Type", "Tap_Type_and_Enter"} and not isinstance((args or {}).get("text"), str):
                raise ValueError("Recovered typing action needs text")
    return {"status": value["status"], "fields": f, "updated_at": utc()}


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root):
        self.root = root.resolve()
        self.manifest = json.loads((self.root / "cases.json").read_text())
        self.cases = {c["id"]: c for c in self.manifest["cases"]}
        self.lock = threading.Lock()
        self.answer_path = self.root / "answers.json"
        self.image_files = {v for c in self.cases.values() for v in c["images"].values()}
        super().__init__(address, ReviewHandler)

    def read_answers(self):
        if not self.answer_path.exists():
            return {"schema_version": 1, "dataset_id": self.manifest["dataset_id"], "reviews": {}}
        data = json.loads(self.answer_path.read_text())
        if data.get("dataset_id") != self.manifest["dataset_id"]:
            raise ValueError("Saved answers belong to another dataset")
        return data


class ReviewHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def json_response(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/reviews":
            try:
                with self.server.lock:
                    self.json_response(200, self.server.read_answers())
            except (ValueError, OSError) as error:
                self.json_response(500, {"error": str(error)})
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        allowed = {"index.html", "review.css", "review.js", "cases.js", "cases.json"} | self.server.image_files
        if relative not in allowed:
            self.send_error(404)
            return
        file = self.server.root / relative
        if not file.is_file():
            self.send_error(404)
            return
        data = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_PUT(self):
        path = urlsplit(self.path).path
        case_id = path.removeprefix("/api/reviews/")
        if not path.startswith("/api/reviews/") or case_id not in self.server.cases:
            self.json_response(404, {"error": "Unknown case"})
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.json_response(403, {"error": "Cross-origin writes are disabled"})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            self.json_response(415, {"error": "Use application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1000000:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            if request.get("dataset_id") != self.server.manifest["dataset_id"]:
                raise ValueError("Dataset identity mismatch")
            review = validate_review(request.get("review"), self.server.cases[case_id])
            with self.server.lock:
                data = self.server.read_answers()
                data["reviews"][case_id] = review
                data["updated_at"] = utc()
                fd, filename = tempfile.mkstemp(prefix=".answers-", dir=self.server.root)
                try:
                    with os.fdopen(fd, "w") as stream:
                        json.dump(data, stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(filename, self.server.answer_path)
                finally:
                    if os.path.exists(filename):
                        os.unlink(filename)
            self.json_response(200, {"saved": True, "review": review})
        except (ValueError, TypeError, OSError) as error:
            self.json_response(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("build")
    b.add_argument("--source", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--previous-review", type=Path)
    s = commands.add_parser("serve")
    s.add_argument("--directory", type=Path, required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "build":
        build(args.source, args.output, args.previous_review)
    else:
        migrate_answers(args.directory)
        server = ReviewServer((args.host, args.port), args.directory)
        print(f"Q-Reflector review: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
