# V9 epoch 9 correction pilot

Protocol record for the named experiment. Run commands from `MobileWorld/`;
GPU IDs and source versions describe that run, not current allocation or status.
Historical data and run snapshots are external to the maintained v9 dataset.

This pilot compares GPT-6 Luna and GPT-6.1 Sol on the same 100 fresh student
responses. It produces candidate training targets and an audit report, without
an optimizer, student weight updates, or a new validation/test evaluation.

## Fixed protocol

- Student: BF16 Qwen3-VL-8B plus the v9 LoRA trained at learning rate 2e-4,
  epoch 9, checkpoint 990. The loaded adapter tensors are checked against the
  saved checkpoint. All source artifacts are hashed in the run manifest.
- Data: 100 distinct rows sampled with seed 20261008 from the 880 accepted v9
  training rows, proportionally stratified as 58 Yes and 42 No. Excludes the
  two quarantined rows. No validation or test inputs are used for this pilot.
- The student sees exactly the original v9 user input: task, actual action,
  available history, and ordered BEFORE/AFTER images. No reference labels or
  reference conditions/explanations are inserted into student input.
- Student generation: one fresh sampled response per selected input,
  temperature 1, top-p 1, top-k 0, at most 1024 output tokens; per-example seed
  `20261008 + selected_index`. These are on-policy training drafts, not the
  greedy decoding protocol used for reported test results.
- Both teachers receive the identical input, student draft, and fallible
  training reference (including label), with the same editing instructions,
  strict JSON schema, medium reasoning, and 4096-token output limit. Images
  use the same maximum side length of 512 pixels. Only the model ID changes.
- A label disagreement or teacher flag excludes that target from automatic
  training eligibility. Raw responses are retained for review. Eligibility is
  only a mechanical filter, not a guarantee of semantic quality.
- API cache keys depend on the entire request, including draft and model.
  Credentials are loaded in memory and never written to logs or reports.
  The request ledger includes usage from every completed attempt. Up to 120
  requests per model allow limited retries; normal completion uses 100 each.
- GPU 2 must be empty before the student is loaded. CUDA allocation is capped
  at 32 GiB and generation stops if free device memory falls below 24 GiB.
  The process exits after 100 responses; no existing process is signaled.

## What the comparison can establish

Report label agreement, label changes, schema/three-check compliance, teacher
flags, token usage, and estimated standard API cost for all 100 records.
Because teachers see training labels and reference explanations, label
agreement is **not held-out classification accuracy**. This experiment cannot
establish a downstream benefit without a subsequent student training run and
evaluation. The student has already trained on these examples.

Inspect screenshot evidence for the first ten samples in the fixed random
order, plus any teacher label disagreements and annotation flags. Additional
interesting cases may be inspected but must be identified as purposive.
For each inspected case, ask:

1. Does the precondition use only minimal prerequisites relevant to this
   action, grounded in BEFORE without invented or invisible requirements?
2. Does the postcondition state the expected local effect of this action,
   without copying the observed AFTER result or demanding later task steps?
3. Do the checks accurately read the two screenshots, distinguish absent
   evidence from contradiction, and assess useful intermediate progress?
4. Does the correction fix a concrete problem or introduce a new one?

This audit is not blinded and does not measure the overall rate of semantic
errors. The single-call format also cannot prove that generated conditions
were causally independent of AFTER.

## Three online rounds

If later authorized, round 1 uses the selected SFT checkpoint to generate
fresh drafts; a teacher corrects them and the student learns from eligible
corrected responses. Round 2 generates new drafts with the updated student,
then obtains new corrections and updates the student again. Round 3 repeats
the process. Reusing a pool of training inputs is allowed; cached corrections
from old drafts must not replace fresh corrections of changed drafts.

A round is a generate/correct/update cycle, not intrinsically one epoch.
Data size, number of passes over each round's targets, learning rate, replay,
checkpoint selection, and stopping criteria require a separate training
protocol. This response-correction approach is not token-distribution KL OPD.

## Commands

Use `artifacts/envs/qreflector-native/bin/python` with
`scripts/qreflector_teacher_pilot_v9.py` and a new `--output` directory:
`prepare`, then `student`, then `teachers`, then `report`.
The teachers command supports `--limit 2` for two real pilot examples per
teacher before resuming to 100. Completed corrections are reused on resume.
Only the student stage requires GPU access; teacher calls use the configured
OpenAI API key and require network access.

After completion, run `scripts/summarize_qreflector_teacher_pilot.py --output
<run> --verify-requests`. Its `comparison.json` is the authoritative quantitative
report: it recomputes both sets of request hashes and accounts for ordinary
input, cached input, cache writes, and output (including reasoning tokens).
The worker's interim `summary.json` does not include the cache-write premium.
Prices are estimates based on observed API usage and the official standard
rates, rather than account invoices. Candidate target files retain source-image
paths relative to `original_data_root`; they are not a newly frozen dataset.
Automatic eligibility must not be mistaken for completed semantic review.
