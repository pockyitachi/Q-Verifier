# Maintained repository layout

`pockyitachi/Q-Verifier` is the project repository. MobileWorld is its online
evaluation and host-agent support component.

```text
Q-Verifier/
  README.md
  MobileWorld/
    pyproject.toml, uv.lock, LICENSE, UPSTREAM.md
    src/mobile_world/       # hosts, reflection, runtime, task evaluators
    scripts/               # current training, data conversion and evaluation
    tests/                 # CPU unit tests and integration checks with mocks
    docs/prompts/          # saved prompt variants
    docker/                # recipes and scripts; provision runtime data separately
    resources/             # pinned app-source submodules
    artifacts/             # ignored: models, environments, runs and screenshots
  GRPO/                    # legacy trainer and current v9 reward definition
  SFT/                     # legacy trainers and shared output parser
  Teacher_Student_Distill/  # legacy distillation workflow
  Rejection_Sampling_DPO/   # legacy pair generation and DPO trainers
  datasets/                # frozen contracts tracked; row data/images external
  docs/experiments/        # compact provenance and configuration records
  docs/archive/            # older documentation, explicitly historical
```

## Source and artifacts

Run new commands from `MobileWorld/` unless a legacy workflow says otherwise.
Current scripts remain in this directory to preserve their shared imports and
their `artifacts/` layout. Their paths to the legacy training folders and frozen
datasets now resolve to the parent Q-Verifier directory.

Install the support package in a dedicated environment:

```bash
cd MobileWorld
uv sync --extra dev
uv run mobile-world --help
```

The project uses Python 3.12. Training additionally needs the relevant PyTorch,
Transformers, PEFT and quantization dependencies; existing experiments record
their environment under their external artifact directory. Importing this
source does not install packages, allocate GPUs, call teachers or start Docker.

Provide frozen dataset row files and referenced screenshots below the existing
`datasets/` locations before running data-dependent tests or training. The
tracked `FROZEN.json` and prompt templates describe the expected contracts.
Some historical intermediate datasets were intentionally removed; do not
silently reconstruct them and claim they are the original data.

Weights and experiment outputs belong under `MobileWorld/artifacts/` or an
explicit path accepted by the selected script. A source-only clone does not
contain these assets. The current server's original artifact store is retained
at `/shared/linqiang/MobileWorld/artifacts`.

## Tests

With the support dependencies installed, run selected CPU-only checks from
`MobileWorld/`:

```bash
PYTHONPATH=src python -m pytest tests/test_qreflector_v9_distill_dpo.py \
  tests/test_qreflector_reward_v9.py tests/unit/reflection/test_v9_protocol.py
```

The protocol parity test also requires the frozen v9 prompt contract in
`../datasets/condition_generation_v9_after_evidence/prompt_template.txt`.
Tests that explicitly check full datasets or local model files require those
external assets; they are not smoke tests for a source-only installation.
The GUI-Reflection native-prompt checks skip when its separately provisioned
upstream source is absent from `MobileWorld/artifacts/vendor/GUI_Reflection`.

Run `tests/unit/reflection/test_qreflector_opsd.py` in the dedicated training
environment, not the MobileWorld host environment. The recorded environment
uses Transformers 4.57.3 and PEFT 0.18.0; its tiny CPU multimodal integration test
is not compatible with the newer Transformers interface in the host environment.
No global environment or training dependency was changed during this import.

## Historical workspaces

The previous outer `/shared/linqiang/MobileWorld` checkout still supports
in-progress runs. Its Git remote is upstream MobileWorld. The maintained
Q-Verifier checkout is currently nested at
`/shared/linqiang/MobileWorld/Q-Verifier`; this physical server arrangement is
not required by a fresh clone.

`Agent-SAMA-main/` is removed from the maintained tree. Its earlier versions
remain available through Git history. Deleting that directory does not rewrite
historical commits or shrink the existing Git object database.
