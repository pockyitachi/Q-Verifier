# Recorded experiment metadata

These JSON files retain compact metadata from existing runs. Absolute paths
refer to the original server artifact store; weights and trajectories are not
included in Git. GPU identifiers and launch settings describe those runs and
are not an instruction to reuse a device.

- `mobileworld-v9-configs-20261013.json`: saved online configurations and model
  provenance. Effective API parameters must also be checked against adapter
  code; configuration alone is not a full trajectory or a result table.
- The sole DPO preference dataset and its construction record now live in
  [Rejection_Sampling_DPO](../../Rejection_Sampling_DPO/README.md). Separate
  intermediate pair summaries are retired.
