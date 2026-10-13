Q-Reflector v9: before-derived expectations, after-based verification.
Only the user prompt is changed from v8. All 880 training, 71 validation,
and 1082 test examples retain their context, image order, labels and full
assistant targets. No teacher annotation or relabeling is performed.
Images are referenced by symlink, not copied. Source data is preserved.
The existing targets inherit v8 limitations; new wording does not prove
that every condition was generated independently of AFTER.
The prompt was revised after examining existing test errors. Report a
subsequent test result as exploratory, not an untouched blind holdout.
Checkpoint selection uses final Yes/No accuracy on validation only.
