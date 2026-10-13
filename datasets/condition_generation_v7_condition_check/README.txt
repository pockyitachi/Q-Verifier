Q-Reflector condition-check data adaptation (v7)

This is a new version of frozen condition_generation_v6_user_reviewed.
The original 882 train / 71 validation / 1082 test rows and labels are preserved.
Original images are referenced through a symlink; no image files were copied.
Read conversion_report.json and DATASET_MANIFEST.json for the FINAL readiness
status and exact counts. Until packaging finishes these may describe an earlier
stage. This directory is not a frozen training release.

Student inputs
  prompt_template.txt is byte-identical to
  docs/prompts/qreflector_condition_check_v1.txt.
  train.inputs.jsonl / val.inputs.jsonl / test.inputs.jsonl contain only user
  input: goal, action, action-only history, BEFORE image, AFTER image.
  No subgoal, ground-truth label, old reason, or generated target is added to input.

New targets
  train.candidates.jsonl and val.candidates.jsonl preserve source row order.
  Four English string fields: precondition, postcondition, reason, answer.
  reason has Precondition check / Postcondition check / Task progress sections.
  Candidate answers always preserve the original Yes/No labels. Consequently,
  flagged candidates can contain a disagreement and MUST NOT be used unreviewed.
  train.ready.jsonl and val.ready.jsonl contain only internally consistent rows,
  with source IDs in the corresponding ready.line_map.jsonl. These are subsets,
  not a replacement for the full 882/71 benchmark protocol.

Annotation method and limitations
  gpt-6-luna inspected images with maximum side 1280 (higher than the 512px used
  in prior student runs). Teacher saw source conditions but NOT original labels
  or original reasons. API access used the existing configured OpenAI key.
  953 first-pass responses are kept verbatim in annotations/.
  Flagged records received a separate check. Materially suspect conditions were
  revised with BEFORE ONLY, task/action/history, original dimensions, and existing
  conditions; AFTER was withheld from that condition-repair API request.
  Repaired criteria were then checked with both screenshots. All versions remain
  in refinements/ and refinements_before_only/. Label comparison occurred locally.
  Coordinate-sensitive explanations received another pass with verified original
  screen dimensions and explicit mappings to resized upload coordinates, saved
  in coordinate_review/. Five remaining condition defects were visually reviewed
  by the assistant, with decisions preserved in review_decisions.jsonl.
  This is API-assisted supervision plus an audit, not full human validation and
  not a pure prompt ablation. A model disagreement does not prove a label is wrong.
  Training and inference still use one call with both images visible from the start.

Review and reproducibility
  review_required.jsonl contains unresolved cases with both targets, source IDs,
  context, image references, and audit flags. No automatic label changes occur.
  *.references.jsonl record source-to-candidate changes for every row.
  examples.before_after.jsonl contains selected concrete before/after examples.
  PREPARATION.json pins source data and prompt; DATASET_MANIFEST.json hashes the
  packaged artifacts. The original frozen data and image hashes are verified.
  No training was launched, no GPU was used, and no existing process was signaled.

Test handling
  test.inputs.jsonl uses the new prompt with unchanged context and image paths.
  test.labels.jsonl preserves the 1082 original Yes/No labels.
  test.jsonl retains the entire old assistant reference verbatim for compatibility.
  Its old reasons are NOT condition-check SFT or validation-loss targets.
  No test row was sent to the annotation API or used to generate new supervision.

Do not train directly from *.candidates.jsonl or silently substitute *.ready.jsonl
for the full dataset. Resolve pending cases and freeze a training release first.
