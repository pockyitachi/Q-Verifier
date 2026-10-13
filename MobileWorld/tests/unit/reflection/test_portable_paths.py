"""Relative source paths and ownership checks must work for another checkout/user."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import build_qreflector_persistent35_review as audit
import run_qreflector_sol_round1 as runner


class BrowserSourcePathTests(unittest.TestCase):
    def test_relative_override_uses_mobileworld_root_from_any_working_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mobileworld = root / "MobileWorld"
            mobileworld.mkdir()
            source = root / "android_world_plus" / "browser.py"
            source.parent.mkdir()
            source.write_text("# external BrowserDraw task source\n")
            previous = Path.cwd()
            try:
                os.chdir(source.parent)
                with patch.object(audit, "ROOT", mobileworld):
                    args = audit.parse_args(["--browser-source", "../android_world_plus/browser.py"])
                    self.assertEqual(audit.browser_source(args.browser_source), source)
                    default = audit.parse_args([])
                    self.assertFalse(default.browser_source.is_absolute())
                    self.assertTrue(audit.browser_source(default.browser_source).is_relative_to(mobileworld))
            finally:
                os.chdir(previous)


class TeacherOwnershipTests(unittest.TestCase):
    def test_monitor_accepts_current_user_without_a_fixed_login_name(self):
        run = Path("test-run")
        proc = MagicMock()
        proc.stat.return_value = SimpleNamespace(st_uid=12345)
        cmdline = Mock()
        cmdline.read_bytes.return_value = (
            str(runner.CPU).encode() + b"\0teachers\0" + str(run).encode() + b"\0"
        )
        stat = Mock()
        stat.read_text.return_value = " ".join(["0"] * 21 + ["42"])
        proc.__truediv__.side_effect = {"cmdline": cmdline, "stat": stat}.__getitem__
        with patch.object(runner, "Path", return_value=proc), patch.object(runner.os, "getuid", return_value=12345):
            teacher = runner.ExistingTeacher(run, 123)
        self.assertEqual(teacher.start_ticks, "42")

    def test_monitor_rejects_other_users_before_reading_process_arguments(self):
        proc = MagicMock()
        proc.stat.return_value = SimpleNamespace(st_uid=99999)
        with patch.object(runner, "Path", return_value=proc), patch.object(runner.os, "getuid", return_value=12345):
            with self.assertRaises(PermissionError):
                runner.ExistingTeacher(Path("test-run"), 123)
        proc.__truediv__.assert_not_called()


if __name__ == "__main__":
    unittest.main()
