"""Exercise the review page in an isolated output directory, never the dataset."""
import copy
import json
import shutil
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import qreflector_review as app
from playwright.sync_api import sync_playwright, expect


class ReviewWebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="qreflector-review-test-")
        cls.output = Path(cls.temp.name)
        # Use the retained review snapshot; intermediate training versions can
        # be removed once v6 is frozen. Never import the real user's answers.
        cls.source = ROOT / '../review/condition_generation_v4_no_subgoal'
        cls.previous = ROOT / '../review/condition_generation_v3'
        cls.protected = {str(p): app.sha(p) for p in cls.source.iterdir() if p.is_file()}
        for name in ("index.html", "review.css", "review.js", "cases.js", "cases.json"):
            shutil.copyfile(cls.source / name, cls.output / name)
        (cls.output / "case_images").mkdir()
        for image in (cls.source / "case_images").iterdir():
            (cls.output / "case_images" / image.name).symlink_to(image.resolve(strict=True))
        cls.server = app.ReviewServer(("127.0.0.1", 0), cls.output)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True, args=["--disable-gpu", "--disable-dev-shm-usage"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.playwright.stop()
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(timeout=2)
        assert all(app.sha(Path(path)) == digest for path, digest in cls.protected.items())
        cls.temp.cleanup()

    def setUp(self):
        answer = self.output / "answers.json"
        if answer.exists():
            answer.unlink()
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [])

    def open(self, case="train-138"):
        self.page.goto(self.url + "/#" + case)
        expect(self.page.locator(".case-link")).to_have_count(29)
        expect(self.page.locator("#case-title")).not_to_be_empty()

    def test_all_cases_have_current_targets_images_and_complete_available_history(self):
        self.open()
        manifest = self.server.manifest
        self.assertEqual(len(manifest["cases"]), 29)
        self.assertEqual(sum(len(c["history"]) for c in manifest["cases"]), 96)
        self.assertEqual([c["id"] for c in manifest["cases"][:3]], ["train-138", "train-155", "train-743"])
        self.assertFalse(any(c["id"] == "train-2" for c in manifest["cases"]))
        expect(self.page.locator("#field-subgoal, #subgoal-text")).to_have_count(0)
        self.assertTrue(all("subgoal" not in c["original"] for c in manifest["cases"]))
        for case in manifest["cases"]:
            self.page.locator(f'[data-case-id="{case["id"]}"]').click()
            expect(self.page.locator("#field-precondition")).to_have_value(case["original"]["precondition"])
            expect(self.page.locator(".history-step")).to_have_count(len(case["history"]))
            for image_id in ("before-image", "after-image"):
                picture = self.page.locator(f"#{image_id}")
                expect(picture).to_have_js_property("complete", True)
                self.assertGreater(picture.evaluate("image => image.naturalWidth"), 0)
            self.assertFalse(self.page.locator("#image-error").is_visible())
            for image in case["images"].values():
                self.assertTrue((self.output / image).is_symlink())
        self.page.locator('[data-case-id="train-138"]').click()
        self.assertIn("Android status bar", self.page.locator("#field-precondition").input_value())
        self.assertFalse((self.output / "answers.json").exists())

    def legacy_bundle(self):
        manifest = json.loads((self.previous / "cases.json").read_text())
        result = {"schema_version": 1, "dataset_id": manifest["dataset_id"], "reviews": {}}
        for case in manifest["cases"]:
            if case["id"] not in {"train-2", "train-138"}:
                continue
            fields = {**case["original"], "action_arguments": json.dumps(case["original"]["action_arguments"])}
            fields["notes"] = "旧版已填写，迁移时保留。"
            result["reviews"][case["id"]] = {"status": "done", "fields": fields,
                "updated_at": datetime.now(timezone.utc).isoformat()}
        return result

    def test_legacy_browser_drafts_and_archived_case_are_preserved(self):
        old = self.legacy_bundle()
        storage = f"qreflector-review:{old['dataset_id']}"
        self.context.add_init_script(f"localStorage.setItem({json.dumps(storage)}, {json.dumps(json.dumps(old))});")
        with self.page.expect_response(lambda r: r.url.endswith("/api/reviews/train-138") and r.request.method == "PUT"):
            self.open("train-2")  # Old bookmark falls back to the first active case.
        expect(self.page.locator("#case-kicker")).to_contain_text("train 138", ignore_case=True)
        expect(self.page.locator("#field-notes")).to_have_value("旧版已填写，迁移时保留。")
        expect(self.page.locator("#case-status")).to_have_text("待复核")
        self.page.locator("#status-filter").select_option("train")
        expect(self.page.locator(".case-link")).to_have_count(3)
        self.page.locator("#status-filter").select_option("test")
        expect(self.page.locator(".case-link")).to_have_count(26)
        with self.page.expect_download() as download:
            self.page.locator("#export-button").click()
        exported = self.output / "migration-export.json"; download.value.save_as(exported)
        value = json.loads(exported.read_text())
        self.assertNotIn("subgoal", value["reviews"]["train-138"]["fields"])
        self.assertNotIn("train-2", value["reviews"])
        self.assertEqual(value["legacy_backup"]["reviews"], old["reviews"])
        self.assertEqual(json.loads(self.page.evaluate("key => localStorage.getItem(key)", storage)), old)
        # A refresh must not overwrite a newer edit with the old browser cache.
        self.page.locator("#status-filter").select_option("train")
        with self.page.expect_response(lambda r: r.url.endswith("/api/reviews/train-138") and r.request.method == "PUT"):
            self.page.locator("#field-notes").fill("新版修改保留")
        self.page.reload()
        expect(self.page.locator("#field-notes")).to_have_value("新版修改保留")

    def test_legacy_file_and_server_answers_migrate_without_overwriting_original(self):
        old = self.legacy_bundle()
        fixture = self.output / "legacy.json"; fixture.write_text(json.dumps(old))
        self.open()
        self.page.locator("#import-file").set_input_files(fixture)
        expect(self.page.locator("#toast")).to_contain_text("已恢复旧版 1 条草稿")
        expect(self.page.locator("#field-notes")).to_have_value("旧版已填写，迁移时保留。")
        # Separate directories exercise the server-side migration without
        # racing the browser's current autosave or touching real answers.
        with tempfile.TemporaryDirectory(prefix="qreflector-migrate-") as temp:
            root = Path(temp); previous = root / "old"; previous.mkdir()
            new = root / "new"; new.mkdir()
            path = previous / "answers.json"; path.write_text(json.dumps(old))
            before = path.read_bytes()
            manifest = copy.deepcopy(self.server.manifest)
            manifest["legacy_import"]["source_review_directory"] = str(previous)
            (new / "cases.json").write_text(json.dumps(manifest))
            app.migrate_answers(new)
            migrated = json.loads((new / "answers.json").read_text())
            self.assertEqual(migrated["reviews"]["train-138"]["status"], "draft")
            self.assertNotIn("subgoal", migrated["reviews"]["train-138"]["fields"])
            self.assertEqual(migrated["legacy_backup"], old)
            self.assertEqual(path.read_bytes(), before)
            written = (new / "answers.json").read_bytes()
            app.migrate_answers(new)
            self.assertEqual((new / "answers.json").read_bytes(), written)

    def test_autosave_refresh_confirm_export_and_import(self):
        self.open()
        edited = "The before and after screenshots were checked for the expected sorting menu."
        with self.page.expect_response(lambda r: r.url.endswith("/api/reviews/train-138") and r.request.method == "PUT") as saved:
            self.page.locator("#field-reason").fill(edited)
        self.assertEqual(saved.value.status, 200)
        self.assertEqual(json.loads((self.output / "answers.json").read_text())["reviews"]["train-138"]["fields"]["reason"], edited)
        self.page.reload(); expect(self.page.locator("#field-reason")).to_have_value(edited)
        with self.page.expect_response(lambda r: r.url.endswith("/api/reviews/train-138") and r.request.method == "PUT"):
            self.page.locator("#confirm-button").click()
        expect(self.page.locator("#progress-text")).to_contain_text("已确认 1 / 29")
        with self.page.expect_download() as download:
            self.page.locator("#export-button").click()
        exported = self.output / "test-export.json"; download.value.save_as(exported)
        value = json.loads(exported.read_text())
        self.assertEqual(value["reviews"]["train-138"]["status"], "done")
        value["reviews"]["train-138"]["fields"]["notes"] = "导入恢复测试"
        value["reviews"]["train-138"]["updated_at"] = (datetime.now(timezone.utc)+timedelta(seconds=20)).isoformat()
        exported.write_text(json.dumps(value))
        self.page.locator("#import-file").set_input_files(exported)
        expect(self.page.locator("#toast")).to_contain_text("已导入 1 条")
        self.page.locator('[data-case-id="train-138"]').click()
        expect(self.page.locator("#field-notes")).to_have_value("导入恢复测试")
        bad = copy.deepcopy(value); bad["dataset_id"] = "different-dataset"
        exported.write_text(json.dumps(bad)); self.page.locator("#import-file").set_input_files(exported)
        expect(self.page.locator("#toast")).to_contain_text("不属于当前 dataset")

    def test_missing_action_and_english_validation_and_uncertain_status(self):
        self.open("test-102")
        self.assertTrue(self.page.locator("#context-editor").get_attribute("open") is not None)
        self.page.locator("#confirm-button").click()
        expect(self.page.locator("#form-errors")).to_contain_text("action 仍缺失")
        self.page.locator("#field-action_type").fill("Tap")
        self.page.locator("#field-action_arguments").fill('{"x": 120, "y": 300}')
        self.page.locator("#field-action_description").fill("Tap the filename field to focus it for editing.")
        old = self.page.locator("#field-precondition").input_value()
        self.page.locator("#field-precondition").fill("输入框可编辑")
        self.page.locator("#confirm-button").click()
        expect(self.page.locator("#form-errors")).to_contain_text("请用英文填写")
        self.page.locator("#field-precondition").fill(old)
        self.page.locator("#confirm-button").click()
        expect(self.page.locator("#progress-text")).to_contain_text("已确认 1 / 29")
        self.page.locator('[data-case-id="test-220"]').click()
        self.page.locator("#field-notes").fill("需要原始 action 日志。")
        with self.page.expect_response(lambda r: r.url.endswith("/api/reviews/test-220") and r.request.method == "PUT"):
            self.page.locator("#uncertain-button").click()
        self.page.locator("#status-filter").select_option("uncertain")
        expect(self.page.locator(".case-link")).to_have_count(1)
        expect(self.page.locator("#case-status")).to_have_text("待补证据")

    def test_zoom_mobile_layout_and_screenshots(self):
        self.open()
        expect(self.page.locator("#before-image")).to_have_js_property("complete", True)
        self.page.screenshot(path="/tmp/qreflector-review-desktop.png", full_page=True)
        self.page.locator("#before-button").click()
        self.assertTrue(self.page.locator("#image-dialog").is_visible())
        self.page.locator("#zoom-button").click()
        expect(self.page.locator("#zoom-button")).to_have_text("适应窗口")
        self.page.keyboard.press("Escape")
        self.assertFalse(self.page.locator("#image-dialog").is_visible())
        self.page.set_viewport_size({"width": 390, "height": 844})
        self.page.screenshot(path="/tmp/qreflector-review-mobile.png", full_page=True)
        self.assertTrue(self.page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"))

    def test_api_limits_writes_to_valid_review_records(self):
        client = self.context.request
        self.assertEqual(client.get(self.url + "/answers.json").status, 404)
        self.assertEqual(client.get(self.url + "/%2e%2e/metadata.json").status, 404)
        case = self.server.cases["test-102"]
        f = {**case["original"], "action_arguments": json.dumps(case["original"]["action_arguments"])}
        review = {"status": "done", "fields": f}
        payload = {"dataset_id": self.server.manifest["dataset_id"], "review": review}
        self.assertEqual(client.put(self.url + "/api/reviews/test-102", data=payload).status, 400)
        self.assertFalse((self.output / "answers.json").exists())
        self.assertEqual(client.put(self.url + "/api/reviews/test-102", data=payload, headers={"Origin":"https://unrelated.example"}).status, 403)
        review["status"] = "uncertain"
        self.assertEqual(client.put(self.url + "/api/reviews/test-102", data=payload).status, 200)
        self.assertEqual(client.put(self.url + "/api/reviews/not-a-case", data=payload).status, 404)


if __name__ == "__main__":
    unittest.main()
