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
instructions. The two maintained prompt templates are [with condition generation](MobileWorld/src/mobile_world/reflection/prompts/with_condition_generation.txt) and [without condition generation](MobileWorld/src/mobile_world/reflection/prompts/without_condition_generation.txt). Their text is unchanged from v9 and its condition-generation ablation.

See the [v9 prompt](MobileWorld/src/mobile_world/reflection/prompts/with_condition_generation.txt),
[protocol implementation](MobileWorld/src/mobile_world/reflection/protocol.py),
and [feedback formatting](MobileWorld/src/mobile_world/reflection/format.py).

## Repository map

| Directory | Contents |
| --- | --- |
| [training/](training/README.md) | Original paper pipeline, v9 training contracts and method notes |
| [MobileWorld/](MobileWorld/README.md) | Host agents, reflection integration, training/evaluation scripts and tests |
| [datasets/](datasets/README.md) | Frozen data-release manifests and prompt contracts |
| [docs/](docs/README.md) | Repository guide and experiment-specific records |

## Training workflows

The original paper's SFT, teacher distillation, DPO and GRPO implementations
remain under `training/`, with their data-corruption and error-correction tools.
Their algorithms and defaults are preserved. Start with the
[training guide](training/README.md) for the individual stages.

The later v9 experiments use separate BF16 LoRA SFT, teacher-correction and OPSD
entry points under `MobileWorld/scripts/`. These are distinct workflows: do not
infer that every checkpoint passed through every method or that the old and new
prompts, datasets and objectives are interchangeable.

The sole maintained [DPO dataset](training/Rejection_Sampling_DPO/README.md)
contains **559 four-field pairs covering 263 transitions from the 882-instance
source pool**. The [manifest](training/Rejection_Sampling_DPO/DPO_MANIFEST.json)
pins its construction and hashes. The latest Luna supervision accepted 849
sources. This data consolidation did not run DPO or GRPO optimization.

## Setup

```bash
git clone https://github.com/pockyitachi/Q-Verifier.git
cd Q-Verifier/MobileWorld
uv sync --extra dev
uv run mobile-world --help
```

MobileWorld uses Python 3.12. Training uses the environment recorded for the
chosen workflow; the [original environment recipe](training/environment/README.md)
is retained separately. Weights, screenshot datasets, Android images, database
volumes, credentials and full experiment logs are external assets.

Read the [layout and test guide](docs/REPOSITORY_LAYOUT.md) before running scripts.
Recorded GPU assignments describe past runs and must be checked before a new
launch. [Experiment records](docs/experiments/README.md) document their own
versions; they are not a consolidated results table.

## Attribution

MobileWorld is derived from [Tongyi-MAI/MobileWorld](https://github.com/Tongyi-MAI/MobileWorld).
Its upstream [Apache-2.0 license](MobileWorld/LICENSE) and source notices are
retained. Import provenance and local modifications are documented in
[MobileWorld/UPSTREAM.md](MobileWorld/UPSTREAM.md).
