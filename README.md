# Q-Verifier / Q-Reflector

Q-Reflector studies post-action verification for GUI agents: whether the executed
action achieved its intended local effect and made valid progress toward the task.
This is the maintained project repository. **MobileWorld is included under
[`MobileWorld/`](MobileWorld/)** as the host-agent and online evaluation support
component; Agent-SAMA is no longer part of the maintained tree.

## Current interface

The v9 verifier receives a task goal, the current action, recent action history,
and ordered BEFORE/AFTER screenshots. It also receives image dimensions and
normalized action coordinates. A separate subgoal and text screen descriptions
are not required. The online interface keeps the last three actions.

The response has four ordered fields:

```json
{
  "precondition": "Relevant starting state and prerequisites for this action.",
  "postcondition": "Expected local effect of this action.",
  "reason": "Precondition check: ... Postcondition check: ... Task progress: ...",
  "answer": "Yes"
}
```

The prompt directs the model to construct expectations from the context and
BEFORE, then check them against AFTER. Both images are supplied in one call;
this does not isolate AFTER from condition generation. The online host receives
a failure hint only for `No`, using the outcome/task evidence in `reason`.
Generated conditions are retained in logs, rather than injected as separate host
instructions. The older shared prompt is retained for reproducible comparisons.

See the [v9 prompt](MobileWorld/src/mobile_world/reflection/prompts/v9.txt),
[protocol implementation](MobileWorld/src/mobile_world/reflection/protocol.py),
and [feedback formatting](MobileWorld/src/mobile_world/reflection/format.py).

## Code and workflows

| Component | Maintained entry point |
| --- | --- |
| Online hosts, reflector backends, token accounting | [MobileWorld package](MobileWorld/src/mobile_world/) |
| v9 BF16 LoRA SFT and validation-selected evaluation | [qreflector_sft_v9.py](MobileWorld/scripts/qreflector_sft_v9.py) |
| Four-field teacher supervision and DPO-pair preparation | [qreflector_v9_distill_dpo.py](MobileWorld/scripts/qreflector_v9_distill_dpo.py) |
| OPSD experiments | [qreflector_opsd.py](MobileWorld/scripts/qreflector_opsd.py) |
| GRPO v9 reward definition | [reward_v9.py](GRPO/reward_v9.py), [reward notes](GRPO/REWARD_V9.txt) |
| Offline evaluation and experiment orchestration | [MobileWorld/scripts](MobileWorld/scripts/) |
| Prompt variants and frozen dataset contracts | [MobileWorld/docs/prompts](MobileWorld/docs/prompts/), [datasets](datasets/) |
| CPU regression tests | [MobileWorld/tests](MobileWorld/tests/) |

These are separate experimental workflows, not a claim that every released
model went through SFT, distillation, OPSD, DPO and GRPO in that order. Legacy
trainer source remains in `SFT/`, `Teacher_Student_Distill/`,
`Rejection_Sampling_DPO/`, and `GRPO/`. Generated model exports, duplicate dataset
copies, candidate dumps and raw experiment results are external artifacts,
excluded from Git. Old manuscript drafts and duplicated historical documentation
are not part of the maintained source tree.

The sole maintained [DPO dataset](Rejection_Sampling_DPO/README.md) contains
**559 four-field pairs covering 263 distinct transitions from the 882-instance
training pool**. It consolidates legacy and recent candidates against current
v9 labels, excludes quarantined sources, and removes duplicate pairs. The
[frozen manifest](Rejection_Sampling_DPO/DPO_MANIFEST.json) records its construction
and hashes. The latest Luna supervised distillation used 849 accepted sources;
DPO and GRPO optimization have not been launched for this consolidated dataset.

## Setup and repository layout

```bash
git clone https://github.com/pockyitachi/Q-Verifier.git
cd Q-Verifier/MobileWorld
uv sync --extra dev
uv run mobile-world --help
```

MobileWorld uses Python 3.12. Training requires its own compatible GPU
environment. Weights, screenshot datasets, Android images, application database
volumes, credentials, and full experiment logs are external assets; cloning
source alone does not provision an experiment. Launch scripts retain explicit
experiment settings, including GPU assignments, and should be reviewed before
starting a new run.

Read the [repository layout and test instructions](docs/REPOSITORY_LAYOUT.md)
and [MobileWorld provenance and runtime requirements](MobileWorld/UPSTREAM.md).
Recorded online configuration metadata is in
[docs/experiments](docs/experiments/); it is not a substitute for the original
trajectories or a new evaluation.

## Attribution

MobileWorld is derived from [Tongyi-MAI/MobileWorld](https://github.com/Tongyi-MAI/MobileWorld).
Its upstream [Apache-2.0 license](MobileWorld/LICENSE) and source notices are
retained. Import provenance and local modifications are documented in
[MobileWorld/UPSTREAM.md](MobileWorld/UPSTREAM.md).
