# Repository consolidation checks

The maintained source was checked independently of the active outer checkout.

- 129 CPU tests passed, plus 3 subtests: v9 prompt/coordinate parity, feedback,
  host integration with mocks, usage accounting, stopping policy, environment
  recovery, distillation/pair construction, GRPO rewards, and prompt evaluation.
- 5 GUI-Reflection native-prompt checks skipped because its external upstream
  source is not included in a source-only clone.
- 11 OPSD tests passed in the original dedicated training environment, including
  a tiny CPU multimodal forward/backward and adapter save/reload check.
- 413 Python files passed syntax parsing.
- No experiment models were loaded onto GPUs, and no API evaluation or Android
  task was launched for these checks. Mock tests do not establish new experiment accuracy.

The host environment's newer Transformers interface does not pass the historical
OPSD multimodal integration test; use the separately maintained training
environment (Transformers 4.57.3, PEFT 0.18.0). Native-prompt tests require the
official GUI-Reflection source as documented in the repository layout.

Imported upstream formatting and frozen prompt bytes are preserved. Model
weights, dataset rows, credentials, ADB keys, database volumes and runtime logs
are excluded from the new import. Existing historical tracked data is unchanged.
