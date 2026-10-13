# Canonical v9 DPO preferences

**The sole maintained dataset is `dpo_pairs.jsonl`: 559 pairs from 263 distinct
training transitions in the 882-instance training source pool.** There is no
separate active 371-, 488-, or 119-pair dataset. The 882 figure describes source
instances, not preference pairs or complete pair coverage.

| Construction | Count |
| --- | ---: |
| Input pair candidates | 607 |
| Quarantined-source pairs excluded | 25 |
| Duplicates removed after normalization | 23 |
| Final pairs | 559 |
| Retained legacy-derived pairs | 440 |
| Retained post-distillation pairs | 119 |
| Unique covered training instances | 263 |
| Chosen Yes / No | 380 / 179 |

The current accepted supervision has 849 of the 882 source instances. The other
33 remain quarantined (2 earlier cases and 31 teacher/reference disagreements).
Pairs from those sources were excluded. Training examples without a usable
negative need not produce a pair. No validation/test transition was added.

## Input, answers and preference rule

All pairs use the frozen v9 goal/action/history prompt with dynamic image
geometry, followed by BEFORE and AFTER. Image paths are relative to this
`training/Rejection_Sampling_DPO/` directory and point to the existing image store; images
are not copied. Preprocessing is RGB with an aspect-preserving 512-pixel maximum
side and LANCZOS resizing, matching the recent pair-generation run.

Both completions are ordered JSON objects:
`precondition -> postcondition -> reason -> answer`.
The reason uses `Precondition check:`, `Postcondition check:`, and `Task progress:`.
The chosen answer matches the current accepted training label; the rejected
answer has the opposite label. This is label-correctness preference supervision,
not an independently scored preference for condition or explanation quality.

Chosen completions use the accepted v9 Luna target. For 57 legacy input pairs,
the old preference direction contradicted the revised label: the former chosen
response supplies the negative instead. These are input-pair corrections, counted
before deduplication; the old pair's wrong label is not silently retained.

The legacy negatives originally had only answer/reason. Their old rationale is
preserved while numbered observation/evaluation sections are mapped to the v9
reason sections. For an unsectioned rationale, the accepted starting-state check
is reused, the complete old rationale is retained in the outcome check, and a
verdict-consistent task-progress conclusion is added. Both expected conditions
are copied from the accepted target. Thus these 440 retained legacy negatives are
**format-adapted legacy responses, not newly sampled four-field student outputs**.
The recent 119 rejected completions retain their original four-field text.
No teacher API, student inference or training was run for consolidation.

Each row records its source input(s), conversion, and whether the original
chosen or rejected side supplied the negative. `dpo_pair_audit.jsonl` accounts
for every input candidate, including exclusions and merged duplicates; it is an
audit of this dataset, not an alternative preference dataset.

## Frozen contract and validation

`DPO_MANIFEST.json` pins the pair/audit files, prompt, all 526 referenced images,
source hashes, covered IDs, label counts and the intended reference adapter.
The freeze builder reproduced the pair and audit files byte-for-byte before
retiring the separate source pair files. New writes must not overwrite this set.

From the repository root:

```bash
python training/Rejection_Sampling_DPO/freeze_dpo_pairs.py validate
python -m pytest MobileWorld/tests/test_frozen_dpo_pairs.py
```

The first command requires the externally provisioned images. CPU tests verify
the tracked dataset without requiring image downloads or GPU access.

## Training scope

This consolidation fixes the **data**, and does not launch DPO training.
The agreed optimizer is standard DPO with a frozen reference model and the sum
of full chosen/rejected response-token log-probabilities: exclude prompt and
padding, include the first response token and EOS. The reference adapter hash
is pinned in the manifest.

`train_dpo_simple.py` is a historical reference-free implementation. It is not
the agreed loss and deliberately refuses this four-field dataset before model
loading. `create_dpo_pairs.py` also refuses to overwrite the frozen files.
