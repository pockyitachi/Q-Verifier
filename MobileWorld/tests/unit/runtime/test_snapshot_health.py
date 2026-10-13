"""Regressions for snapshots that acknowledge loading before Android is usable."""

import io
import time
from unittest.mock import Mock

import pytest
from PIL import Image

from mobile_world.runtime import controller as module
from mobile_world.runtime.controller import AndroidController, DeviceUnhealthyError
from mobile_world.runtime.utils.helpers import AdbResponse, execute_adb
from mobile_world.tasks.base import BaseTask


def framework(focus="Window{abc u0 com.android.launcher/Launcher}"):
    return "\n".join([
        "1", "Service activity: found", "Service window: found",
        "Service input: found", "Service package: found", f"mCurrentFocus={focus}",
    ])


@pytest.mark.parametrize("output", [
    "1", framework().replace("Service input: found", "Service input: not found"),
    framework("null"), framework("Window{abc Application Not Responding: launcher}"),
    framework("Window{abc Application Error: launcher}"), framework().replace("1\n", "0\n", 1),
])
def test_boot_flag_does_not_hide_missing_services_or_error_dialog(output):
    assert not module._framework_is_ready(output)


def test_healthy_framework_has_services_and_a_focused_window():
    assert module._framework_is_ready(framework())


def test_png_signature_alone_is_not_a_valid_screenshot():
    assert not module._is_valid_png_bytes(module.PNG_SIGNATURE + b"truncated")
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    assert module._is_valid_png_bytes(buffer.getvalue())


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_snapshot_requires_continuous_health_and_a_valid_frame(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    controller = object.__new__(AndroidController)
    controller.device = "emulator-test"
    controller.check_health = Mock(side_effect=[True, False] + [True] * 10)
    controller.check_screenshot_readiness = Mock(side_effect=[False, True])
    controller.wait_for_device_stability(stable_for_seconds=2, timeout_seconds=12)
    assert clock.now == 7
    assert controller.check_screenshot_readiness.call_count == 2


def test_unhealthy_snapshot_is_rejected_within_deadline(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    controller = object.__new__(AndroidController)
    controller.device = "emulator-test"
    controller.check_health = Mock(return_value=False)
    controller.check_screenshot_readiness = Mock()
    with pytest.raises(DeviceUnhealthyError):
        controller.wait_for_device_stability(stable_for_seconds=2, timeout_seconds=4)
    assert clock.now == 4
    controller.check_screenshot_readiness.assert_not_called()


def test_console_ok_does_not_override_snapshot_failure(monkeypatch):
    controller = object.__new__(AndroidController)
    controller.device = "emulator-test"
    controller.check_health = Mock(return_value=True)
    controller.wait_for_device_stability = Mock()
    monkeypatch.setattr(module, "execute_adb", Mock(return_value=AdbResponse(
        success=True, output="KO: Snapshot load failure: different AVD configuration\nOK"
    )))
    assert controller.load_snapshot("init_state") is False
    controller.wait_for_device_stability.assert_not_called()


def test_task_cannot_retain_successful_initialization_after_bad_reset():
    class Task(BaseTask):
        goal = "Test task reset"
        app_names = {"Settings"}

    task = Task()
    task.initialized = True
    controller = Mock(load_snapshot=Mock(return_value=False))
    assert task.initialize_task(controller) is False
    assert task.initialized is False
    controller.home.assert_not_called()


def test_adb_timeout_terminates_only_its_own_command_group(tmp_path, monkeypatch):
    # A local fake ADB command: this test never contacts a device or ADB server.
    adb = tmp_path / "adb"
    adb.write_text("#!/bin/sh\nsleep 30\n")
    adb.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    start = time.monotonic()
    response = execute_adb("adb shell test", output=False, timeout_seconds=0.1)
    assert not response.success
    assert "timed out" in response.error
    assert time.monotonic() - start < 3


def test_health_probe_does_not_restart_during_task_initialization(monkeypatch):
    from mobile_world.core import server

    controller = Mock()
    restart = Mock()
    task = Mock(initialized=False)
    def initialize(_controller):
        response = server.health()
        assert response["transition_in_progress"] is True
        controller.check_health.assert_not_called()
        task.initialized = True

    task.initialize_task.side_effect = initialize
    monkeypatch.setattr(server, "CONTROLLERS", {"emulator-test": controller})
    monkeypatch.setattr(server, "ensure_controller", Mock(return_value=controller))
    monkeypatch.setattr(server, "task_registry", Mock(get_task=Mock(return_value=task)))
    monkeypatch.setattr(server, "restart_emulator_with_avd", restart)
    monkeypatch.setattr(server, "RUNNING_TASK", None)
    server.init_task(server.TaskOperationRequest(task_name="test", req_device="emulator-test"))
    restart.assert_not_called()


def test_unhealthy_active_task_is_not_silently_restarted(monkeypatch):
    from mobile_world.core import server

    restart = Mock()
    monkeypatch.setattr(server, "CONTROLLERS", {
        "emulator-test": Mock(check_health=Mock(return_value=False)),
    })
    monkeypatch.setattr(server, "RUNNING_TASK", Mock())
    monkeypatch.setattr(server, "restart_emulator_with_avd", restart)
    result = server.health()
    assert result["ok"] is False
    assert result["recovery_deferred_for_active_task"] is True
    restart.assert_not_called()
