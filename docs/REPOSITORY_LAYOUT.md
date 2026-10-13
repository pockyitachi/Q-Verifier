# Repository layout

This is the maintained project repository. Its tracked top
level contains four functional directories:

```text
Q-Verifier/
  README.md
  training/
    README.md                  # original paper workflow and v9 entry points
    SFT/                       # original SFT, evaluation and shared parser
    Teacher_Student_Distill/    # original teacher-response distillation
    Rejection_Sampling_DPO/     # original tools + sole frozen v9 pair dataset
    GRPO/                      # original trainer + separate v9 reward
    Data_Corruption/           # original negative-construction tool
    Self_Improvement/          # original error-correction workflow
    environment/               # original dependency/setup recipe
    docs/                      # additional training methods, including OPSD
  MobileWorld/
    src/mobile_world/          # hosts, reflectors, runtime and task evaluators
    scripts/                   # current experiment and training entry points
    tests/                     # CPU regression and integration checks
    docs/                      # benchmark setup guides
    docker/, resources/        # container recipes and pinned app sources
    artifacts/                 # ignored runtime assets
  datasets/                    # one v9 release contract; row files/images external
  docs/
    README.md                  # documentation index
    experiments/               # dated protocol/configuration records
```

## Running code

Run original-paper tools from their stage directory, for example
`cd training/SFT`. Their relative model/data arguments address sibling training
stages. Run the later experiment scripts from `MobileWorld/`; they share imports,
evaluation helpers and an `artifacts/` layout there. See
[training/README.md](../training/README.md) for the distinction.

The root no longer presents the original global setup script as a universal
installer. Its source and requirements are preserved under `training/environment/`.
MobileWorld dependencies are installed from `MobileWorld/pyproject.toml` and
`uv.lock`. Use the recorded training environment for each model workflow.

## External assets and existing server paths

Weights, duplicate data exports, screenshot stores, raw results and generated
teacher/candidate responses are ignored. A fresh clone requires separately
provisioned assets. The frozen 559-pair DPO file, its manifest/audit and MobileWorld
fixtures remain tracked. DPO image paths resolve relative to
`training/Rejection_Sampling_DPO/` and expect the existing images under
`training/SFT/training_data_64/images/`.

On the current server, the original stage directories still contain ignored
runtime data and model weights. Local compatibility links expose those assets
under `training/` and keep old source paths usable. These links are not committed;
the GitHub tree shows the maintained source layout. Data and models were not
copied or moved, and frozen pair IDs, prompt bytes and hashes were preserved.
Only the v9 dataset contract is tracked. The named prompt pair lives in
`MobileWorld/src/mobile_world/reflection/prompts/`; the dataset prompt path is
a relative link to `with_condition_generation.txt`.

In the original server layout, the maintained checkout is `Q-Verifier/` inside
an outer workspace. From the repository root, `../artifacts/` addresses that
workspace's original experiment store and frozen source snapshots. This
arrangement is not required by a fresh clone. Editing the outer checkout does
not update this repository.

## Relative-path conventions

- Experiment configuration records declare `path_base: repository_root`.
  Resolve their file paths from the top-level `Q-Verifier/` directory.
  These records describe the historical external asset layout; they do not
  provision files or change current script defaults.
- The dataset manifest declares `path_base: manifest_directory`. Its `source`
  and `prompt_source` paths are relative to the directory containing the manifest.
- The import inventory's `source_workspace` is relative to the repository root;
  its individual file entries are relative to that original workspace.
- Scripts anchor defaults to their own location and resolve paths at runtime.
  The persistent-case audit accepts `--browser-source` for an external AndroidWorld
  checkout; relative values are based on `MobileWorld/`.
- Docker and Android paths refer to their container/device filesystems and keep
  their required absolute form.

## Tests

From the repository root, with pytest installed:

```bash
python -m pytest MobileWorld/tests/test_frozen_dpo_pairs.py
python training/Rejection_Sampling_DPO/freeze_dpo_pairs.py validate --text-only
```

Both commands validate the public text release without downloading images.
The public email placeholders do not match the private original screenshots;
image validation requires the complete private dataset and original manifest.
With the MobileWorld dependencies installed, run from `MobileWorld/`:

```bash
PYTHONPATH=src python -m pytest tests/test_qreflector_v9_distill_dpo.py \
  tests/test_qreflector_reward_v9.py tests/unit/reflection/test_v9_protocol.py
```

The protocol test uses the frozen v9 prompt contract in `../datasets/`.
Data-dependent tests additionally require the row files and image stores.
GUI-Reflection native-prompt tests require its separately provisioned upstream
source under `MobileWorld/artifacts/vendor/GUI_Reflection`.

The OPSD multimodal tests use Transformers 4.57.3 and PEFT 0.18.0 in their
recorded native training environment. They are not compatible with every newer
host-environment Transformers release; see [OPSD notes](../training/docs/opsd.md).

## Provenance

`docs/mobileworld-import.json` records the original import snapshot. The workspace
location is expressed relative to this repository; file entries and original
content hashes continue to describe the import, not later source edits.
MobileWorld's licenses and attribution are preserved in
[MobileWorld/UPSTREAM.md](../MobileWorld/UPSTREAM.md).
