import importlib.util
import json
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / '../GRPO/reward_v9.py'
spec = importlib.util.spec_from_file_location('reward_v9', PATH)
reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reward)


def response(answer='Yes', **changes):
    obj = dict(precondition='An editable search field is available.',
               postcondition='The supplied query appears in that field.',
               reason='Precondition check: Search is focused. Postcondition check: The query is visible. Task progress: It searches the requested item.',
               answer=answer)
    obj.update(changes)
    return json.dumps(obj)


class RewardTests(unittest.TestCase):
    def test_symmetric_ablation_is_preserved(self):
        for answer, other in [('Yes', 'No'), ('No', 'Yes')]:
            self.assertEqual(reward.score(response(answer), answer, mode='symmetric')['reward'], 1.1)
            self.assertEqual(reward.score(response(other), answer, mode='symmetric')['reward'], -0.9)

    def test_default_uses_legacy_asymmetric_confusion_matrix(self):
        for gold,pred,expected in [('Yes','Yes',2.5), ('No','No',2.0),
                                    ('Yes','No',-2.5), ('No','Yes',-2.0)]:
            result = reward.score(response(pred),gold)
            self.assertEqual(result['judgement_reward'],expected)
            self.assertAlmostEqual(result['reward'],expected+.1)
            self.assertEqual(result['mode'],'asymmetric')

    def test_correctness_dominates_format_in_both_modes(self):
        for mode in ('symmetric','asymmetric'):
            for gold,wrong in [('Yes','No'),('No','Yes')]:
                for weight in (0,.1,.25):
                    self.assertGreater(reward.score(json.dumps({'answer':gold}),gold,weight,mode)['reward'],
                                       reward.score(response(wrong),gold,weight,mode)['reward'])
                    self.assertLessEqual(reward.score('malformed',gold,weight,mode)['reward'],
                                         reward.score(json.dumps({'answer':wrong}),gold,weight,mode)['reward'])

    def test_order_and_reason_sections(self):
        obj = json.loads(response())
        reordered = json.dumps({k:obj[k] for k in reversed(obj)})
        self.assertEqual(reward.score(reordered, 'Yes')['format_reward'], 0)
        self.assertEqual(reward.score(response(reason='Unstructured prose.'), 'Yes')['format_reward'], 0)
        self.assertEqual(reward.score(response(precondition=''), 'Yes')['format_reward'], 0)

    def test_no_length_or_keyword_penalty(self):
        long = response(reason='Precondition check: '+ 'no visible change '*100 +
                        'Postcondition check: The result is visible. Task progress: Useful.')
        self.assertEqual(reward.score(long, 'Yes')['reward'], 2.6)

    def test_invalid_duplicate_or_nested_answer_is_not_a_verdict(self):
        for value in ['{"answer":"Yes","answer":"No"}', '[{"answer":"Yes"}]',
                      '{"reason":"answer: Yes"}', 'Yes', '{"answer":true}']:
            self.assertIsNone(reward.inspect_completion(value)['answer'])
            self.assertEqual(reward.score(value, 'Yes')['reward'], -2.5)
            self.assertEqual(reward.score(value, 'Yes', mode='symmetric')['reward'], -1)

    def test_trl_chat_and_ground_truth_alignment(self):
        fn = reward.V9RewardFunction()
        self.assertEqual(fn([[{'role':'assistant','content':response()}]], expected_answer=['Yes']), [2.6])
        symmetric = reward.V9RewardFunction(mode='symmetric')
        self.assertEqual(symmetric([response()],expected_answer=['Yes']),[1.1])
        for expected in [None, [], ['unknown']]:
            with self.assertRaises(ValueError): fn([response()], expected_answer=expected)

    def test_invalid_configuration_fails_closed(self):
        for weight in (-.1,.3,float('nan')):
            with self.assertRaises(ValueError): reward.V9RewardFunction(format_weight=weight)
        with self.assertRaises(ValueError): reward.V9RewardFunction(mode='typo')

    def test_group_normalization_cancels_class_weight_but_centering_preserves_it(self):
        # Same correctness pattern and format for two input groups, one per label.
        # With no invalid outputs, different class weights are only a rescaling.
        import math
        def advantages(gold, normalize):
            other = 'No' if gold == 'Yes' else 'Yes'
            values = [reward.score(response(p),gold)['reward'] for p in [gold,gold,other,other]]
            mean = sum(values)/len(values)
            centered = [v-mean for v in values]
            std = math.sqrt(sum(v*v for v in centered)/(len(centered)-1))
            return [v/std for v in centered] if normalize else centered
        for yes,no in zip(advantages('Yes',True),advantages('No',True)):
            self.assertAlmostEqual(yes,no)
        for yes,no in zip(advantages('Yes',False),advantages('No',False)):
            self.assertAlmostEqual(yes/no,1.25)


if __name__ == '__main__':
    unittest.main()
