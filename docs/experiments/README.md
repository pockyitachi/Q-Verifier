# Experiment records

These records describe fixed experiments. They are separate from the
[original paper training workflow](../../training/README.md) and the
[single maintained v9 dataset](../../datasets/README.md). Historical dataset
sizes, source checkpoints, prompts and GPU IDs apply only to the named run.
They are not current progress reports or instructions to reuse a device.

| Record | Scope |
| --- | --- |
| [MobileWorld configurations](mobileworld-v9-configs-20261013.json) | Saved host/reflector configuration and model provenance; not a results table |
| [Luna correction on v6](luna-correction-v6.md) | Response-correction protocol, distinct from token-level KL distillation |
| [V9 teacher pilot](teacher-pilot-v9.md) | Comparison of teacher corrections on 100 training responses |
| [Sol round 1](sol-round1-v9.md) | Fixed student-generation, correction and training protocol |
| [OPSD method](../../training/docs/opsd.md) | Frozen same-vocabulary teacher and response-token loss |

Absolute artifact paths refer to the server's original experiment store. Weights,
trajectories and raw API responses are not included in Git. Later corrections,
reruns and selected-trajectory collections require their own run records; a
configuration snapshot does not establish a single-run success rate.

The sole frozen [DPO preference dataset](../../training/Rejection_Sampling_DPO/README.md)
and its audit are maintained under `training/`. Intermediate pair summaries
are not alternative datasets.
