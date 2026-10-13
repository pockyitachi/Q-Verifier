# Experiment entry points

Run these scripts from `MobileWorld/`. They share helpers in this directory;
moving individual helpers independently would break imports and run snapshots.
The [training index](../../training/README.md) distinguishes the original paper
pipeline from subsequent experiment variants.

| Work | Main script |
| --- | --- |
| V9 BF16 SFT | `qreflector_sft_v9.py` |
| V9 supervision and DPO candidate construction | `qreflector_v9_distill_dpo.py` |
| Sol correction rounds | `qreflector_sol_round1.py`, `qreflector_sol_round2.py` |
| OPSD | `qreflector_opsd.py` |
| Offline V9 local-model baselines | `run_v9_local_baselines.py` |
| GUI-Reflection native-prompt baseline | `eval_gui_reflection_native_androidworld.py` |
| Resolution comparison | `eval_qreflector_resolution.py`, `eval_qreflector_test_resolution.py` |

The maintained dataset is `../datasets/condition_generation_v9_after_evidence/`.
The two student prompt templates are
[`with_condition_generation`](../src/mobile_world/reflection/prompts/with_condition_generation.txt)
and [`without_condition_generation`](../src/mobile_world/reflection/prompts/without_condition_generation.txt).
The latter is the saved prompt ablation; no online experiment is launched by
its presence. Host-agent prompts and teacher annotation instructions serve
different roles and are not alternative student-verification templates.

Scripts named for v6/v8, earlier annotation conversions and original API
baselines preserve those experiments' implementation. They may require the
external original data or run snapshots, which are no longer published as
additional dataset/template releases. Do not substitute the current data and
report the result as a reproduction of an earlier configuration.

`run_*` scripts and training subcommands can start GPU/API work. Saved defaults
often include experiment-specific paths; inspect the run configuration before
execution. Preparation, summarization and unit tests do not authorize a new run.
