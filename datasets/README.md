# Current dataset

The sole maintained train/validation/test release is
[condition_generation_v9_after_evidence](condition_generation_v9_after_evidence/README.txt).

| Split | Records |
| --- | ---: |
| Train | 880 accepted from the 882-instance source pool |
| Validation | 71 |
| Test | 1,082 |

The two excluded source rows remain identified in the frozen manifest.
Later Luna distillation accepted 849 source instances after additional filtering;
that teacher-target file is an external training artifact, not another published
train/validation/test release.

The release keeps goal, current action, history and ordered BEFORE/AFTER images.
Its default response is `precondition → postcondition → reason → answer`.
Only two named prompts are maintained: [with condition generation](../MobileWorld/src/mobile_world/reflection/prompts/with_condition_generation.txt)
and [without condition generation](../MobileWorld/src/mobile_world/reflection/prompts/without_condition_generation.txt).
The release's `prompt_template.txt` points to the first file; its content and
frozen hash are unchanged.

Manifest `source` and `prompt_source` paths are relative to the manifest's
directory, as declared by `path_base`. The prompt path refers to the maintained
template with the same frozen content hash. `original_manifest_sha256` records
the manifest before this path-only metadata normalization. Its full-file hash
therefore changes; example, label, image and prompt artifact hashes do not.
Historical evaluations that pin the original manifest must use the original
archived manifest. Existing experiment snapshots retain their original pins.

Manifests and the prompt link are tracked. Row files, screenshots and annotation
caches must be provisioned separately. Historical v6/v7/v8 data and prompt copies
are excluded from the Git tree; existing server assets remain available for
reproducing earlier runs, without being advertised as additional releases.

The only DPO preference set is the derived
[559-pair dataset](../training/Rejection_Sampling_DPO/README.md), covering 263
training transitions. It is a preference dataset derived from the same source
pool, not a second train/validation/test split. Its pair IDs and hashes are
unchanged.
