"""Do not accept a running container before the app API can serve requests."""
from unittest.mock import Mock

from mobile_world.runtime.app_helpers import mattermost
from mobile_world.tasks.base import BaseTask


def test_running_container_with_failed_http_is_not_healthy(monkeypatch):
    monkeypatch.setattr(mattermost, "get_mattermost_backend_status", lambda: "running")
    monkeypatch.setattr(mattermost.requests, "get", Mock(return_value=Mock(ok=False)))
    assert not mattermost.is_mattermost_healthy()


def test_http_wait_is_bounded_and_accepts_delayed_readiness(monkeypatch):
    now = [0]
    monkeypatch.setattr(mattermost.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(mattermost.time, "sleep", lambda seconds: now.__setitem__(0, now[0]+seconds))
    probe = Mock(side_effect=[False, False, True])
    monkeypatch.setattr(mattermost, "_mattermost_http_ready", probe)
    assert mattermost._wait_for_mattermost_http(timeout_seconds=5)
    assert now[0] == 4
    monkeypatch.setattr(mattermost, "_mattermost_http_ready", Mock(return_value=False))
    assert not mattermost._wait_for_mattermost_http(timeout_seconds=5)
    assert now[0] == 9


def test_hook_that_discards_backend_failure_cannot_mark_task_ready(monkeypatch):
    from mobile_world.tasks import base

    class Task(BaseTask):
        goal = "test app readiness"
        app_names = {"Mattermost"}
        snapshot_tag = None
        def initialize_task_hook(self, controller):
            return None

    monkeypatch.setattr(mattermost, "stop_mattermost_backend", Mock())
    monkeypatch.setattr(base.mastodon, "stop_mastodon_backend", Mock())
    monkeypatch.setattr(base, "clear_config", Mock())
    monkeypatch.setattr(base, "clear_callback_files", Mock())
    monkeypatch.setattr(mattermost, "is_mattermost_healthy", lambda: False)
    task = Task()
    assert task.initialize_task(Mock()) is False
    assert task.initialized is False
