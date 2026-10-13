# Q-Reflector OPSD prototype

This is a new, independent BF16 LoRA training entry point. It does not replace or
modify the SFT/evaluation scripts, dataset, checkpoints, or running jobs. Preparing
or auditing a run never loads model weights or starts GPU work. Only the explicit
`train` subcommand does so. No environment packages need to be upgraded.

## Method and scope

Implementation: [`scripts/qreflector_opsd.py`](../scripts/qreflector_opsd.py).

The initial checkpoint is supplied explicitly; the script does not select the
highest-scoring test checkpoint. One frozen BF16 Qwen3-VL base is shared between:

* `default`: the trainable student LoRA, initialized from the supplied SFT adapter.
* `opsd_teacher`: a frozen copy of that same initial SFT adapter.

The frozen teacher follows the experimental recipe in
[Self-Distilled Reasoner, sections 3.2 and 4.1](https://arxiv.org/html/2601.18734v3).
The teacher is **not** the unadapted base and does not track student updates.

For each of the 882 training inputs, the student receives the unmodified frozen
v6 user message and the two screenshots, resized with the existing 512-pixel
maximum-side protocol. It samples one fresh response at temperature 1, top-p 1,
top-k 0. Correct, incorrect, invalid and length-limited responses all contribute
to distillation; they are not filtered by the final label.

The teacher receives the same observations plus a training-only reference JSON.
`--reference-fields full` supplies the four existing reference fields;
`--reference-fields answer` supplies only the verdict. Neither mode adds any
reference fields to the student input. The teacher does not generate a new target
response: it scores the student's exact token IDs with the same response prefix.

The default objective is:

```
mean_response_tokens sum_vocabulary min(p_teacher * (log p_teacher - log p_student), 0.05)
    + 0.1 * assistant_response_cross_entropy(existing_reference)
```

Teacher probabilities are detached. Vocabulary contributions are clipped before
summation, as in the paper's pointwise clipping scheme. The clipped loss can be
negative; it is not an ordinary nonnegative KL metric. Setting
`--pointwise-clip 0` disables clipping. The full vocabulary is used, with
checkpointed token chunks to limit temporary FP32 memory. No sampled log-ratio
is incorrectly treated as a differentiable KL loss.

Auxiliary reference SFT is **our extension**, not a claim of exact reproduction
of the paper. `--sft-weight 0` gives pure distillation. Defaults (LR 5e-6, one
epoch, accumulation 8, 512 response tokens, SFT weight 0.1) are provisional,
untuned pilot settings, not claimed optimal hyperparameters. One full epoch has
111 optimizer updates; the last update correctly averages its two examples.

The code does not silently change frozen v6's task wording to a new conjunction
prompt. Any future task-definition change needs a separately versioned protocol.
The existing metadata is reused, not assumed to be infallible. Giving the teacher
an answer does not prove it supplies correct explanations or useful gradients.
Transfer from the paper's text reasoning tasks to this GUI VLM remains untested.

## Prepare and audit on CPU

Use the existing native environment (Transformers 4.57.3 / PEFT 0.18.0), not the
Unsloth training environment. Replace the example adapter/run directory as needed.
Step333 below is an example initialization, not a checkpoint-selection decision.

```bash
artifacts/envs/qreflector-native/bin/python scripts/qreflector_opsd.py prepare \
  --adapter artifacts/training/qreflector_bf16_v6_20261004_gpu4/training/final \
  --output artifacts/training/qreflector_opsd_v6_trial \
  --epochs 1 --learning-rate 5e-6 --sft-weight 0.1

artifacts/envs/qreflector-native/bin/python scripts/qreflector_opsd.py audit \
  --run-dir artifacts/training/qreflector_opsd_v6_trial
```

`prepare` verifies the frozen training file and referenced image hashes and pins
adapter, processor, model configuration, and worker hashes. It refuses to reuse
an existing output directory. It writes `plan.json` and an inspectable
`prompt_example.json`. It does not open validation/test rows or copy images.

`audit` disables CUDA, loads only the processor, and tokenizes all 882 examples.
It checks identical ordered visual inputs, exact student/target prefix matching,
and sequence-length budgets for both student and teacher, with no truncation.
It writes `token_audit.json`. A full passing audit is required before training.
If the worker or inputs change, prepare a new run and audit it again.

## Explicit future GPU launch

**Do not execute this during code review.** GPU availability and the initialization
checkpoint must be decided separately. There is no default GPU or automatic
launch, no termination of other processes, and no Docker interaction.

```bash
artifacts/envs/qreflector-native/bin/python scripts/qreflector_opsd.py train \
  --run-dir artifacts/training/qreflector_opsd_v6_trial \
  --gpu-uuid GPU-REPLACE-WITH-THE-EXPLICITLY-SELECTED-UUID \
  --memory-cap-gib 40 --reserve-free-gib 8
```

The launcher requires a full GPU UUID, checks free memory before loading, sets a
per-process allocator cap, and checks headroom during training. It fails its own
run when headroom is insufficient; it never unloads models or signals processes.
These controls do not eliminate shared-GPU compute contention or guarantee that
the production model fits. A separate GPU smoke test is still required.

## Checkpoints, resumption and evaluation

All periodic checkpoints and epoch-end checkpoints are retained, along with a
final adapter. The checkpoint root contains only the student adapter in the
format accepted by the existing BF16 evaluator. Each save also contains
`optimizer_rng.pt` and `opsd_state.json`: optimizer moments, random states,
deterministic data cursor, step, original teacher fingerprint, and plan hash.
The LR is a deterministic function of the recorded step and plan.

Saving is atomic and existing checkpoints are never overwritten. A run lock
prevents two training processes from writing the same run. Resume explicitly:

```bash
artifacts/envs/qreflector-native/bin/python scripts/qreflector_opsd.py train \
  --run-dir artifacts/training/qreflector_opsd_v6_trial \
  --resume artifacts/training/qreflector_opsd_v6_trial/checkpoint-25 \
  --gpu-uuid GPU-REPLACE-WITH-THE-EXPLICITLY-SELECTED-UUID
```

Use the newest complete checkpoint; resuming from an older checkpoint in a run
that already contains later checkpoints will refuse to overwrite those files.
Interrupted work after the checkpoint is regenerated. Append-only metrics and
rollout logs include `attempt_id`, so resumed attempts can be distinguished.
The teacher is always restored from the original SFT adapter, never the resumed
student. GPU kernels may still introduce nondeterminism despite restored RNG.

`training_rollout_accuracy` is a sampled **training** metric, not validation or
test accuracy. No automatic evaluation or model selection is started. The
existing `qreflector_sft_bf16.py evaluate --adapter <new checkpoint> ...` accepts
the saved adapters and scores final Yes/No. Choose validation/selection policy
before comparing test results; teacher-only reference inputs must never be used
for deployment evaluation.

## Checks and remaining limitation

Run the CPU regression tests using the native environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  artifacts/envs/qreflector-native/bin/python -m unittest discover \
  -s tests/unit/reflection -p test_qreflector_opsd.py -v
```

These test leakage, shifted response alignment including EOS, full-vocabulary
loss/gradient equivalence, clipping, partial batches, and a tiny randomly
initialized **two-image Qwen3-VL** with PEFT. The tiny-model test exercises
sampling, teacher scoring, student backpropagation, frozen teacher/base weights,
and checkpoint save/reload without loading any production weights.

CPU tests and full-data token audits cannot establish production GPU memory
usage, throughput, training stability, or accuracy gains. The 8B GPU training
path requires an explicitly authorized smoke run before a real experiment.

## Authorized epoch-6 pilot (2026-10-05)

The user requested GPU 5 and the epoch-6 / cumulative-step-666 adapter. The pilot
is prepared in `artifacts/training/qreflector_opsd_v6_epoch6_20261005_gpu5`, with a
separate `_smoke` run containing one eight-example optimizer update. Both runs
start from the original epoch-6 adapter; the smoke weights are not reused.

The optional `scripts/run_qreflector_opsd.py` runner requires a completed smoke
run with matching configuration. When explicitly launched, it serially runs the
111-update pilot and the existing BF16 evaluator on all 1,082 test instances.
Evaluation uses only the predetermined final student adapter, the unchanged v6
student prompt, 512-pixel images, and greedy decoding with a 1,024-token limit.
Training uses the default settings above, including the 0.1 auxiliary SFT term.

`pipeline_status.json` tracks the runner separately from the training
`status.json`. The final `comparison.json` compares Yes/No metrics with the
already completed epoch-6 test, after checking labels, prompt hashes and input
token counts. Neither the runner nor the trainer signals existing processes or
changes GPU settings. Failed stages stop this pipeline and preserve its logs.
