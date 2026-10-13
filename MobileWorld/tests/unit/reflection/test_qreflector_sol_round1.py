"""Geometry, teacher-only privilege and annotation quarantine protections."""
import copy
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import qreflector_sol_round1 as round1
from PIL import Image


def sample(root):
    Image.new('RGB',(1440,3120),'red').save(root/'before.png')
    Image.new('RGB',(1440,3120),'blue').save(root/'after.png')
    context={'goal':'Enable Data Saver','history':[],'action':{'type':'Tap',
             'arguments':{'x':494,'y':947},'description':'Tap Location'}}
    user={'role':'user','content':[{'type':'text','text':'=== Context ===\n'+json.dumps(context)+'\n=== Image Reference ==='},
                                {'type':'image','image':'before.png'},{'type':'image','image':'after.png'}]}
    target={'precondition':'REFERENCE_PRE','postcondition':'REFERENCE_POST',
            'reason':'Precondition check: before. Postcondition check: after. Task progress: useful.',
            'answer':'Yes'}
    return {'messages':[user,{'role':'assistant','content':[{'type':'text','text':json.dumps(target)}]}]}


class RoundTests(unittest.TestCase):
    def test_incomplete_http_body_retries_same_request(self):
        with tempfile.TemporaryDirectory() as d:
            client=Mock(ledger=Path(d)/'api_requests.jsonl')
            expected=({'parsed':'complete'},False)
            client.correct.side_effect=[round1.http.client.IncompleteRead(b'private',9473),expected]
            row={'messages':[]}
            with patch.object(round1.time,'sleep') as sleep:
                actual=round1.correct_with_transport_retries(client,row,'original draft')
            self.assertEqual(actual,expected)
            self.assertEqual(client.correct.call_count,2)
            self.assertEqual(client.correct.call_args_list[0],client.correct.call_args_list[1])
            sleep.assert_called_once_with(2)
            self.assertNotIn('private',client.ledger.read_text())
            self.assertEqual(json.loads(client.ledger.read_text())['error_type'],'IncompleteRead')

    def test_interrupted_response_retries_are_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            client=Mock(ledger=Path(d)/'api_requests.jsonl')
            client.correct.side_effect=round1.http.client.IncompleteRead(b'',10)
            with patch.object(round1.time,'sleep'), self.assertRaisesRegex(RuntimeError,'three attempts'):
                round1.correct_with_transport_retries(client,{},'draft')
            self.assertEqual(client.correct.call_count,3)

    def test_call_cap_and_nontransport_errors_are_not_retried(self):
        client=Mock()
        client.correct.side_effect=RuntimeError('API call limit reached')
        with self.assertRaisesRegex(RuntimeError,'API call limit'):
            round1.correct_with_transport_retries(client,{},'draft')
        self.assertEqual(client.correct.call_count,1)

    def test_worker_failure_is_reported_immediately(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            with patch.object(round1,'teacher_worker',side_effect=RuntimeError('failure')):
                with self.assertRaisesRegex(RuntimeError,'failure'):
                    round1.monitored_teacher_worker(run,{},[],0)
            self.assertEqual(json.loads((run/'teacher_0_status.json').read_text())['status'],'failed')

    def test_pipeline_status_can_record_its_child_stage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            round1.status(root,'pipeline',status='running',stage='teachers',child_pid=123)
            saved=json.loads((root/'pipeline_status.json').read_text())
            self.assertEqual(saved['stage'],'teachers')
            self.assertEqual(saved['child_pid'],123)

    def test_original_coordinates_are_transformed_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row=sample(root);info=round1.geometry(row,root)
            point=info['action_points'][0]
            self.assertEqual(info['images'][0]['sent_width'],236)
            self.assertEqual(info['images'][0]['sent_height'],512)
            self.assertAlmostEqual(point['normalized_xy'][1],947/3120)
            self.assertAlmostEqual(point['sent_image_pixels'][1],947/3120*512)
            self.assertTrue(point['inside_original_image'])
            self.assertEqual(point['original_pixels'],[494,947])

    def test_geometry_and_reference_are_teacher_only_and_images_stay_ordered(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row=sample(root);original=copy.deepcopy(row)
            cfg={'data':str(root),'image_max_side':512,'teacher_model':'gpt-6.1-sol',
                 'teacher_reasoning':'medium','teacher_max_output_tokens':4096}
            request=round1.teacher_request(row,'STUDENT_DRAFT',cfg)
            student=round1.correction.student_messages(row)
            self.assertEqual(row,original)
            self.assertEqual(student,[original['messages'][0]])
            self.assertNotIn('REFERENCE_PRE',json.dumps(student))
            self.assertNotIn('IMAGE GEOMETRY',json.dumps(student))
            self.assertIn('IMAGE GEOMETRY',json.dumps(request))
            self.assertIn('REFERENCE_PRE',json.dumps(request))
            self.assertEqual([c['type'] for c in request['input'][0]['content']],
                             ['input_text','input_text','input_image','input_image','input_text'])

    def test_disagreements_and_known_annotation_issues_cannot_enter_training(self):
        corrected={'precondition':'P','postcondition':'Q','reason':
            'Precondition check: P. Postcondition check: Q. Task progress: yes.','answer':'Yes'}
        record={'sample_id':'train-1','expected':'Yes','status':'completed','parsed':{
                'training_usable':True,'review_note':'ok','corrected_response':corrected}}
        self.assertEqual(round1.eligible(record),(True,'accepted'))
        record['sample_id']='train-90'
        self.assertEqual(round1.eligible(record),(False,'predeclared_annotation_hold'))
        record['sample_id']='train-1';corrected['answer']='No'
        self.assertFalse(round1.eligible(record)[0])
        corrected['answer']='Yes';record['parsed']['training_usable']=False
        self.assertFalse(round1.eligible(record)[0])
        record['parsed']['training_usable']=True;corrected['reason']='No checks'
        self.assertEqual(round1.eligible(record),(False,'missing_reason_checks'))

    def test_stream_reader_ignores_only_incomplete_trailing_record(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'stream.jsonl';path.write_text('{"sample_id":"train-1"}\n{"sample_id":')
            self.assertEqual(round1.rows_if_exists(path),[{'sample_id':'train-1'}])
            path.write_text('{"sample_id":')
            self.assertEqual(round1.rows_if_exists(path),[])


if __name__=='__main__':unittest.main()
