"""Sharing is allowed only with the explicitly recorded, still-owned companion."""
import json
from unittest.mock import Mock

import pytest

from mobile_world.reflection import qreflector_v9_worker as worker

GPU = 'GPU-00000000-0000-0000-0000-000000000000'


@pytest.fixture
def setup(tmp_path, monkeypatch):
    manifest = tmp_path / 'actor.json'
    manifest.write_text(json.dumps({'pid': 100, 'start_ticks': '200', 'gpu_uuid': GPU}))
    query = Mock(return_value='101\n')
    identities = {
        100: {'uid': 123, 'pgid': 100, 'start_ticks': '200', 'state': 'S'},
        101: {'uid': 123, 'pgid': 100, 'start_ticks': '201', 'state': 'S'},
    }
    monkeypatch.setattr(worker.subprocess, 'check_output', query)
    monkeypatch.setattr(worker, 'process_identity', lambda pid: identities[pid])
    monkeypatch.setattr(worker.os, 'getuid', lambda: 123)
    return manifest, query, identities


def test_explicit_owned_companion_is_allowed_and_empty_default_stays_strict(setup):
    manifest, _, _ = setup
    worker.check_selected_gpu(GPU, manifest)
    with pytest.raises(RuntimeError, match='existing compute processes'):
        worker.check_selected_gpu(GPU)


@pytest.mark.parametrize('field,value', [('uid', 999), ('pgid', 102), ('start_ticks', '999'), ('state', 'Z')])
def test_changed_companion_identity_is_rejected(setup, field, value):
    manifest, _, identities = setup
    identities[100][field] = value
    with pytest.raises(RuntimeError, match='identity or ownership'):
        worker.check_selected_gpu(GPU, manifest)


@pytest.mark.parametrize('uid,pgid', [(999, 100), (123, 999)])
def test_unrelated_process_is_rejected_even_when_same_user(setup, uid, pgid):
    manifest, query, identities = setup
    query.return_value = '101\n102\n'
    identities[102] = {'uid': uid, 'pgid': pgid, 'state': 'S', 'start_ticks': '300'}
    with pytest.raises(RuntimeError, match='unrelated'):
        worker.check_selected_gpu(GPU, manifest)


def test_wrong_gpu_manifest_is_rejected(setup):
    manifest, _, _ = setup
    manifest.write_text(json.dumps({'pid': 100, 'start_ticks': '200', 'gpu_uuid': 'GPU-other'}))
    with pytest.raises(RuntimeError, match='different GPU'):
        worker.check_selected_gpu(GPU, manifest)
