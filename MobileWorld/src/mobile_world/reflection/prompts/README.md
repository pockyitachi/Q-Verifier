# Verifier prompts

Only two standalone Q-Reflector prompt templates are maintained:

| Template | Output |
| --- | --- |
| [with_condition_generation.txt](with_condition_generation.txt) | `precondition → postcondition → reason → answer` |
| [without_condition_generation.txt](without_condition_generation.txt) | `reason → answer` |

These are the unchanged v9 template and its existing condition-generation
ablation, renamed for clarity. Both judge whether the current action was
correctly executed and made useful task progress. Both images enter the same
call; the with-condition instruction orders expectation construction and
evidence checking but does not hide AFTER during generation.

The online `v9_geometry` protocol reads `with_condition_generation.txt` and
adds image dimensions, normalized coordinates and the last three actions.
The without-condition file is a template for controlled comparison; renaming
it does not launch a run or register a new online protocol. Its
[metadata](without_condition_generation.metadata.json) records the controls
needed for a fair comparison, including equivalent feedback processing.

The frozen v9 dataset's `prompt_template.txt` is a relative link to the
with-condition file, not a third template. Original paper implementations and
recorded experiment protocols are preserved for provenance; their historical
inputs must not be silently treated as these new named templates.
