# Training workflows

This directory preserves the original paper's training pipeline alongside the
v9 additions. Moving the files here does not change the original algorithms,
training defaults, model weights or experiment results.

## Original paper workflow

| Component | Code and method notes |
| --- | --- |
| SFT and transition evaluation | [SFT](SFT/README.md) |
| Teacher-response distillation | [Teacher_Student_Distill](Teacher_Student_Distill/README.md) |
| Candidate sampling and DPO | [Rejection_Sampling_DPO](Rejection_Sampling_DPO/README.md) |
| GRPO and the original asymmetric reward | [GRPO](GRPO/README.md) |
| Synthetic negative construction | [Data_Corruption](Data_Corruption/README.md) |
| Error-directed teacher correction | [Self_Improvement](Self_Improvement/README.md) |

SFT, teacher distillation, DPO and GRPO remain available. The two data/correction
utilities are supporting workflows; their presence does not establish that every
reported checkpoint used them. Use each run's saved configuration to establish
the order of stages and exact data, checkpoint, precision and hyperparameters.

Run the original scripts from their respective directories, for example
`cd training/SFT` from the repository root. Their relative data/model arguments
still refer to sibling stage directories. Weights, images and generated outputs
must be provisioned separately. For a new supervised run, explicitly pass
`--data_dir ../../datasets/condition_generation_v9_after_evidence` from a stage
directory. Reproducing an original run requires its external original inputs.
The original environment recipe is preserved in
[environment/](environment/README.md).

## V9 additions

The current experiment scripts remain under `MobileWorld/scripts/`, where their
shared imports, dataset contracts and evaluation helpers are maintained. Run
these scripts from `MobileWorld/`.

| Work | Entry point / contract |
| --- | --- |
| BF16 LoRA SFT and validation-selected test | [qreflector_sft_v9.py](../MobileWorld/scripts/qreflector_sft_v9.py) |
| Four-field teacher supervision and candidate pairs | [qreflector_v9_distill_dpo.py](../MobileWorld/scripts/qreflector_v9_distill_dpo.py) |
| Sol correction rounds | [Round 1](../MobileWorld/scripts/qreflector_sol_round1.py), [round 2 with geometry](../MobileWorld/scripts/qreflector_sol_round2.py) |
| OPSD experiments | [Method and commands](docs/opsd.md) |
| Frozen DPO preferences | [559-pair contract](Rejection_Sampling_DPO/README.md) |
| V9 GRPO reward | [Reward definition](GRPO/REWARD_V9.txt) |

V9 uses `precondition → postcondition → reason → answer`. Original two-field
training, v6 OPSD and later v9 experiments have distinct prompts and data; do not
combine their descriptions into one training recipe. The agreed future standard
DPO loss and the v9 GRPO reward are documented separately from the original
trainer implementations. Their availability does not mean a corresponding v9
DPO or GRPO optimization run has been completed.

## Data and reproducibility

- [Dataset index](../datasets/README.md): frozen prompt and release contracts.
- [Experiment records](../docs/experiments/README.md): protocols and run metadata.
- `Rejection_Sampling_DPO/dpo_pairs.jsonl` is the sole frozen preference set:
  559 pairs covering 263 source transitions. Its original relative image paths,
  pair IDs and file hashes are unchanged by the directory reorganization.

On the existing server, ignored compatibility links expose the existing asset
directories to this layout. Those links are local, not part of a fresh clone.
For a new checkout, provision data and models at the paths expected by the
selected stage, or pass explicit paths. The DPO image contract expects
`training/SFT/training_data_64/images/`; no screenshots are copied into Git.
