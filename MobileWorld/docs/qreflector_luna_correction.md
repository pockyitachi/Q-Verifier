# Luna online correction distillation

This pilot is **online student-conditioned correction distillation**, not
token-level KL OPD. The user selected this alternative on 2026-10-06 after the
API limitations were explained. GPT-6 Luna runs through OpenAI's API; only the
Qwen3-VL student runs on the explicitly selected GPU 5.

The current student samples a fresh four-field response for each frozen training
input. Luna receives the same goal, action, history and two ordered screenshots,
the student's draft, and the full training-only reference annotation. It returns
an evidence-grounded corrected response plus a training usability flag and short
review note. The student is trained with assistant-only cross entropy on the
corrected response under the **original, unprivileged v6 input**. The reference,
draft, and editing instructions never enter the student prompt. Review notes are
not training targets. This loss is SFT on newly generated correction targets; it
is not the 0.1 auxiliary SFT loss from the OPSD experiment, nor a KL loss.

The initial student is the original BF16 LoRA SFT epoch-6 checkpoint, not an OPSD
output. One pass visits all 882 training examples, in the same seeded order as
the OPSD pilots, with accumulation 8 (111 batches), AdamW LR 5e-6, warmup 0.1,
and cosine decay. Sampling uses temperature 1, top-p 1, top-k 0, max 512 tokens.
Images have maximum side 512. Luna uses `gpt-6-luna`, medium reasoning and at most
2048 output tokens including reasoning. Student targets are capped at 1024 tokens
without truncation; input plus response is limited to 8192 tokens.

Invalid, incomplete, overlong, teacher-flagged, or label-disagreeing corrections
are logged and excluded from updates. The original labels are never overwritten.
Gradients are averaged over accepted examples, including partial batches; empty
batches perform no optimizer update. Batch and optimizer-update counts are
reported separately. A low teacher acceptance rate stops this pilot for review.
These rules are provisional quality checks, not tuned research hyperparameters.

Only this run's student LoRA is updated. An independent eight-example smoke run
must pass a real backward/update/save test and a response review before the full
runner can start. Both smoke and production initialize from epoch 6 independently.
Completed checkpoints retain optimizer/RNG state for explicit resumption. Teacher
responses are cached by an exact request hash. Up to three transient API retries
are allowed, with a hard 1100-request cap per run. API usage is recorded; keys,
authorization headers and request image payloads are not logged.

The runner evaluates only the predetermined final checkpoint on all 1082 test
rows using the unchanged v6 prompt, greedy decoding, and final Yes/No scoring.
No test input or label is sent to Luna for training or checkpoint selection.
The original dataset, previous results and source checkpoints stay unchanged.

Entrypoints: `scripts/qreflector_luna_correction.py prepare|audit|train` and
`scripts/run_qreflector_luna_correction.py`. Preparing and auditing are CPU-only.
Launching requires a complete audit, an explicit GPU UUID, 40 GiB per-process
allocator cap, and at least 8 GiB free reserve. The program never signals existing
processes or changes GPU settings. Shared-GPU compute contention remains possible.

Method distinction: [Thinking Machines' OPD description](https://thinkingmachines.ai/blog/on-policy-distillation/)
requires teacher log-probabilities on student trajectories. Luna's public API
supports generation and, in applicable configurations, probabilities of its own
output; it does not provide the arbitrary student-trajectory scoring interface
used by that recipe. Different GPT/Qwen tokenizations also prevent directly
substituting its outputs into our same-vocabulary OPSD loss.
