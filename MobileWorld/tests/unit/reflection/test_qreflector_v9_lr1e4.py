import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import qreflector_sft_v9_lr1e4 as worker
import run_qreflector_v9_lr1e4 as runner


def entries(correct=60, epoch=3):
    return [dict(epoch=e, step=e * 110, split='val', total=71,
                 correct=correct if e == epoch else 50, accuracy=(correct if e == epoch else 50) / 71,
                 success_recognition_rate=0.9, failure_recognition_rate=0.7, invalid_outputs=0)
            for e in range(1, 16)]


def test_training_routes_only_to_gpu6_and_does_not_continue_control(tmp_path):
    command = runner.command(tmp_path, 'train')
    assert command[command.index('--gpu-uuid') + 1] == 'GPU-5182a8a6-4d6f-af6f-0415-55fd854c2989'
    assert '--resume' not in command and '--adapter' not in command
    assert str(tmp_path / 'train') in command
    assert command[command.index('--memory-cap-gib') + 1] == '32'
    with pytest.raises(ValueError):
        runner.command(tmp_path, 'test')


@pytest.mark.parametrize('candidate_correct,expected_lr', [(60, 2e-4), (61, 1e-4), (59, 2e-4)])
def test_lr_selection_uses_validation_and_retains_control_on_tie(tmp_path, monkeypatch, candidate_correct, expected_lr):
    control = tmp_path / 'control'
    control.mkdir()
    (control / 'validation_results.json').write_text(json.dumps({'results': entries()}))
    # A test result is deliberately contradictory; it must never be consulted.
    (control / 'results.json').write_text(json.dumps({'test': {'accuracy': 0}}))
    monkeypatch.setattr(runner, 'CONTROL', control)
    comparison = runner.compare_learning_rates(tmp_path, entries(candidate_correct))
    assert comparison['selected_learning_rate'] == expected_lr
    assert comparison['correct_delta'] == candidate_correct - 60
    assert comparison['test_used_for_selection'] is False


def test_insufficient_headroom_stops_before_model_allocator_setup(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=Mock(return_value=(55 * 1024**3, 140 * 1024**3))))
    setup = Mock()
    monkeypatch.setattr(worker, 'base_gpu_setup', setup)
    with pytest.raises(RuntimeError, match='headroom'):
        worker.gpu_setup(torch, 32)
    setup.assert_not_called()
    torch.cuda.mem_get_info.return_value = (60 * 1024**3, 140 * 1024**3)
    worker.gpu_setup(torch, 32)
    setup.assert_called_once_with(torch, 32)


def test_control_rejects_changed_training_configuration(monkeypatch, tmp_path):
    control = json.loads((runner.CONTROL / 'plan.json').read_text())
    control['pinned_files'] = {}
    (tmp_path / 'plan.json').write_text(json.dumps(control))
    monkeypatch.setattr(runner, 'CONTROL', tmp_path)
    plan = {**control, 'training': dict(control['training'], learning_rate=1e-4)}
    runner.validate_control(plan)
    plan['training']['seed'] = 43
    with pytest.raises(AssertionError, match='only training-config change'):
        runner.validate_control(plan)
