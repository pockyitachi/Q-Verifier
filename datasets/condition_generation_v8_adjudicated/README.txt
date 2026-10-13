Q-Reflector v8: condition-check supervision with disputed cases adjudicated

This is a NEW version. v6 and v7 are preserved. No training was started.
Canonical training files: train.jsonl (880 accepted), val.jsonl (71 accepted).
The source training set still has 882 records; two are explicitly quarantined.
Test: 1,082 rows copied byte-for-byte from v7, including its updated prompt.
Test labels and assistant references remain unchanged from frozen v6.
Screenshots remain path references via the images symlink; no images are copied.

Scope and provenance
All 87 v7 disputes (76 train, 11 validation) were visually adjudicated using
task, actual action, available history, and both screenshots. 53 labels changed,
32 labels were retained, and 2 records were quarantined. Explanations and
conditions were corrected where needed. The other v7 records inherit their
API-assisted annotations, including five previous visual corrections. This is
not a claim that all 953 source records received a new independent manual audit.

Decision standard
Yes requires a correctly realized useful local effect AND valid task progress.
The whole task or an old subgoal need not be completed by one transition.
Subgoal is absent from the model input. Conditions describe the task-relevant
expectation from goal, current action, history and BEFORE, not the observed
failure rewritten as a desired outcome. Reason checks the evidence and progress.

Useful progress includes opening a relevant folder or app switcher, entering a
correct query, moving forward through relevant content, necessary permissions,
and entering a search's loading view. For a compound shortcut, a completed useful
component may earn Yes; the reason must explicitly say which component remains
unverified. This is task-relevant transition acceptance, not an assertion that
every low-level subcommand completed successfully.

Actual coordinates and visible effects take precedence over mistaken action
narratives. A different but useful route can be accepted when its relevance is
concrete. Inspecting a plausible candidate and newly learning it is ineligible
can be useful information gain; continuing to wait on an already-known wrong
candidate does not provide that same progress.

Note entry tolerates nonessential punctuation, quotation and automatic list
formatting differences. Substantive target/field errors remain failures. Known
transcription imperfections are explicitly described rather than hidden.
The screenshots verify entry of supplied notes, not the truth of every original
source claim or persistence after leaving the editor.

Routine focus preparation can be accepted when the correct editable field and
keyboard are ready. Explicitly requested stopping at a relevant screen may be
valid even without a visual change. Arbitrary desktop waits, repeated boundary
scrolls, unrelated setup, and premature navigation away are not useful progress.
Claims about global best/nearest choices or hidden persistence are not inferred
from a limited visible comparison.

Quarantine
train-243 and train-589 have conflicting Triller/Troolls app identity evidence.
One pair shows Play Store's Triller Open control launching a Troolls-branded
screen. The other reopens Troolls from the launcher. The screenshots verify the
launch but cannot settle whether this is the intended app. Both are excluded
from training; final_label is null. Historical source/candidate labels in audit
records are provenance, not approved ground truth. No user action is required
to use the accepted subset.

Files
train/val.jsonl: accepted SFT examples; all targets are English and ordered as
precondition, postcondition, reason, answer. Reason has three explicit checks.
train/val.line_map.jsonl: output line to ORIGINAL source line and stable ID.
train/val.inputs.jsonl, references.jsonl, labels.jsonl: aligned sidecars.
all_source_audit.jsonl: all 953 source records, including quarantine dispositions.
adjudication_decisions.jsonl: every disputed case with old/new targets and reason.
quarantine.jsonl: excluded records, contexts and evidence, not an SFT file.
prior_v7_review_decisions.jsonl: previous five visual corrections, retained.
adjudication_report.json: counts, checks and practical limitations.
DATASET_MANIFEST.json: hashes and source provenance.

Experiment interpretation
This revision changes supervision, 53 labels, and training membership; it is not
just a prompt-format change. Validation has six revised labels (three in each
direction), so comparisons must use a common validation version. Test was not
relabelled. The old loader pins v6 and assumes 882 training rows; configure and
validate a new run explicitly before training on v8. Do not silently redirect an
existing experiment. This version is prepared, not automatically frozen or run.
