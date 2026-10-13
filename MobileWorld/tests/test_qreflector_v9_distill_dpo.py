import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import qreflector_v9_distill_dpo as pipeline


def target(answer='Yes', suffix=''):
    return dict(precondition='Search is available.', postcondition='The query is entered.',
        reason='Precondition check: The field is visible. Postcondition check: The result '+suffix+
               ' is visible. Task progress: Relevant to the task.', answer=answer)


class PipelineTests(unittest.TestCase):
    def test_format_repair_preserves_original_label_and_refuses_disputes(self):
        old = target()
        fixed = {**old,'reason':old['reason'].replace('The field is visible.','The search field is visible.')}
        record = dict(sample_id='train-1',request_sha256='request-hash',expected='Yes',
                      output_text=json.dumps(dict(target=old,review_needed=False)))
        repair = dict(sample_id='train-1',request_sha256='request-hash',target=fixed,
                      changed_fields=['reason'],kind='format_only',note='clarify wording',raw_annotation_sha256='raw-hash')
        self.assertTrue(pipeline.repaired_annotation(record,repair)['accepted'])
        wrong_label = {**repair,'target':{**fixed,'answer':'No'},'changed_fields':['reason','answer']}
        with self.assertRaises(AssertionError): pipeline.repaired_annotation(record,wrong_label)
        disputed = {**record,'output_text':json.dumps(dict(target=old,review_needed=True))}
        with self.assertRaises(AssertionError): pipeline.repaired_annotation(disputed,repair)

    def test_duplicates_and_field_order_rejected(self):
        with self.assertRaises(ValueError): pipeline.parse_json('{"answer":"Yes","answer":"No"}')
        value = target()
        self.assertTrue(pipeline.valid_target(value))
        self.assertFalse(pipeline.valid_target({k:value[k] for k in reversed(value)}))

    def test_pair_quality_coverage_and_no_input_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); (run/'train/final').mkdir(parents=True)
            (run/'train/final/adapter_model.safetensors').write_bytes(b'fake-test-adapter')
            adapter_hash = pipeline.digest(run/'train/final/adapter_model.safetensors')
            samples = []
            for index in range(3):
                samples.append(dict(sample_id=f'train-{index+1}', source_line=index+1, expected='Yes',
                    row=dict(messages=[dict(role='user', content=[dict(type='text',text='v9 prompt'),
                        dict(type='image',image='before.png'),dict(type='image',image='after.png')]),
                        dict(role='assistant',content=[dict(type='text',text=json.dumps(target()))])])) )
            originals = copy.deepcopy(samples)
            sets = [[target(),target('No','A'),target('No','A'),target('No','B'),{}],
                    [target()]*5, [target('No',str(i)) for i in range(5)]]
            records = []
            for sample, values in zip(samples,sets):
                records.append(dict(sample_id=sample['sample_id'], expected='Yes', adapter_sha256=adapter_hash,
                    prompt_sha256=hashlib.sha256(json.dumps(sample['row']['messages'][0],sort_keys=True).encode()).hexdigest(),
                    candidates=[dict(text=json.dumps(t),parsed=t,valid=pipeline.valid_target(t),
                                     at_generation_limit=False) for t in values]))
            pipeline.write_rows(run/'candidate_responses.jsonl',records)
            plan = dict(candidates=dict(per_sample=5), dpo=dict(max_pairs_per_source=2))
            with patch.object(pipeline,'accepted_rows',return_value=(plan,samples)):
                pipeline.build_pairs(run)
            report = json.loads((run/'completion_report.json').read_text())
            self.assertEqual((report['pairs'],report['unique_pair_sources'],report['invalid_responses']), (4,2,1))
            self.assertEqual(report['all_five_correct_sources'],1)
            self.assertEqual(samples, originals)
            for pair in pipeline.rows(run/'dpo_pairs.jsonl'):
                self.assertEqual(json.loads(pair['chosen'][0]['content'])['answer'],'Yes')
                self.assertEqual(json.loads(pair['rejected'][0]['content'])['answer'],'No')
                self.assertEqual(len(pair['image_paths']),2)
                self.assertTrue(all(Path(p).is_absolute() for p in pair['image_paths']))


if __name__ == '__main__': unittest.main()
