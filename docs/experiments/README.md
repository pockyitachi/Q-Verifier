# Recorded experiment metadata

These JSON files retain compact metadata from existing runs. Absolute paths
refer to the original server artifact store; weights and trajectories are not
included in Git. GPU identifiers and launch settings describe those runs and
are not an instruction to reuse a device.

- `mobileworld-v9-configs-20261013.json`: saved online configurations and model
  provenance. Effective API parameters must also be checked against adapter
  code; configuration alone is not a full trajectory or a result table.
- `luna-v9-distill-dpo-20261012.json`: completed distillation and DPO-pair
  preparation. The flags explicitly record that DPO and GRPO training did not
  start and validation/test data were not used.
