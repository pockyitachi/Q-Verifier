# Sol correction, round 1, from v9 epoch 9

This experiment was authorized on 2026-10-08: generate training-set responses
from the existing student, ask GPT-6.1 Sol to correct them, train the student,
and evaluate the final adapter on all 1,082 test transitions. No validation is
used, and no further correction round is scheduled.

Run directory:
`artifacts/training/qreflector_sol_round1_v9_epoch9_20261008_gpu7`.

## Fixed protocol

- Initial student: BF16 Qwen3-VL-8B with the v9 LoRA adapter from
  `artifacts/training/qreflector_bf16_v9_20261007_gpu7/train/checkpoint-990`
  (epoch 9 of the original LR 2e-4 SFT run).
- Training inputs: all 880 retained v9 training examples. The earlier dataset
  contained 882 examples; two had already been excluded during dataset repair.
- Rollouts: 100 matching initial-student responses are reused from the pilot;
  the remaining 780 are generated on GPU 7. Sampling uses temperature 1,
  top-p 1, top-k 0, and at most 1,024 new tokens. The initial adapter stays fixed
  throughout this round's generation.
- Teacher: `gpt-6.1-sol`, medium reasoning, four API workers, at most 4,096 output
  tokens. All 880 teacher corrections are generated for this run. The teacher
  receives the original training input, the student's answer, a fallible
  training reference, and explicit original/resized image dimensions and action
  coordinate mappings. That additional geometry addresses coordinate mistakes
  observed in the pilot; it is not added to the student prompt.
- Filtering: accepted targets must have the required English four-field JSON,
  all three evidence-check markers, a usable teacher assessment, and a verdict
  agreeing with the retained training label. Disagreements are quarantined;
  existing labels are not silently changed. `train-90` is held out because its
  original annotation remains disputed. Training requires at least 75% of the
  880 corrections to pass these gates. Passing gates is not a claim of manual
  verification of every explanation.
- Student update: one epoch of assistant-only cross entropy on accepted Sol
  corrections, continuing the original LoRA adapter with a fresh AdamW
  optimizer. Learning rate 5e-6, cosine schedule, 10% warmup, microbatch 1,
  accumulation 8, gradient clipping 1, and no auxiliary original-target SFT.
  The BF16 base is frozen; 43,646,976 LoRA parameters are trainable. Optimizer
  recovery checkpoints are saved every 25 steps and at the end.
- Evaluation: only the predetermined final correction checkpoint is evaluated
  on the unchanged 1,082-example v9 test set, with greedy generation and a
  1,024-token limit. Score final Yes/No only; invalid answers are incorrect.
  Report overall accuracy, success recognition, failure recognition, and invalid
  outputs against the original epoch-9 predictions. The test is not used to
  select between checkpoints in this round.

This is response-correction distillation on student-generated responses, not
token-level KL OPD. A round includes generation, teacher correction, and student
training; it is distinct from an epoch over fixed training targets.

## Execution and process isolation

The supervisor is `scripts/run_qreflector_sol_round1.py`. CPU stages are in
`scripts/qreflector_sol_round1.py`; GPU stages are in
`scripts/qreflector_sol_round1_gpu.py`.

Only new GPU children for this run are launched, sequentially on GPU 7. The
PyTorch allocator cap is 32 GiB; a GPU stage requires at least 56 GiB free before
loading and checks for at least 24 GiB free while running. These checks limit
memory pressure; shared GPU compute can still slow concurrent jobs. No existing
process is signaled, stopped, reset, or reconfigured. Source checkpoints and
datasets are retained unchanged.

An initial supervisor status-writing bug was fixed before full generation or
training. The already-running CPU teacher worker is observed read-only by the
replacement supervisor and continues consuming its original rollout stream.
Automatic approval review rejected a proposed cleanup signal to that worker;
no signal was sent. Source revisions and plan revisions are archived inside
the run directory under `code_revisions/`.

A disposable eight-example training check completed successfully before full
generation. Adapter weights changed, the frozen base received no gradients,
loss and gradient norm were finite, and peak allocated memory was 17.94 GiB.
Those temporary weights were discarded; full training starts from epoch 9.

On 2026-10-08, generation finished all 880 responses. One API worker had received
an incomplete HTTP body and exited after 159 of its 220 assigned corrections;
the other three completed, leaving 819 corrections in total. The supervisor
then exited without starting training. Transport retries now cover incomplete
HTTP bodies, interrupted connections and malformed response JSON, with at most
three attempts and the existing API-call cap. A failed worker also writes its
failure status immediately. Fifteen focused tests passed, including replay of
the incomplete-body failure and bounded retries. All earlier responses and
corrections were preserved; the supervisor was relaunched to complete only the
remaining 61 corrections before the original training/test stages. No existing
process was stopped or modified. The previous code, plan and failure statuses
are archived under `code_revisions/transport_retry_20261008/` in the run.

## Outputs

- `plan.json`: pinned data, code and source-adapter hashes and experiment choices.
- `pipeline_status.json`, `generate_status.json`, `teacher_*_status.json`,
  `train_status.json`, `test_status.json`: current stage progress.
- `student_rollouts.jsonl`, `teachers/*/corrections.jsonl`: generation and teacher
  audit records; API credentials are not written to these files.
- `corrected_train.jsonl`, `quarantined_corrections.jsonl`,
  `correction_manifest.json`: frozen correction dataset and filtering counts.
- `train/final`: final student adapter and processor.
- `test/predictions.jsonl`, `test/summary.json`: full test responses and metrics.
- `comparison.json`: original-versus-corrected accuracy, paired error changes,
  and estimated teacher API usage and cost.

These final outputs are created as the corresponding stages finish. Their
absence while generation is running does not indicate completed training or
evaluation.
