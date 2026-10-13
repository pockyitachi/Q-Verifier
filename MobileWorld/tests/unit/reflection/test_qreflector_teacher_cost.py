import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from summarize_qreflector_teacher_pilot import cost


class CostTests(unittest.TestCase):
    def test_cache_writes_replace_input_rate_and_reasoning_is_already_output(self):
        events = [{'usage': {'input_tokens': 1000, 'input_tokens_details': {
            'cached_tokens': 200, 'cache_write_tokens': 700}, 'output_tokens': 300,
            'output_tokens_details': {'reasoning_tokens': 200}}}]
        result = cost(events, 'gpt-6.1-sol')
        self.assertAlmostEqual(result['estimated_standard_api_cost_usd'],
                               (100*2 + 200*0.1 + 700*2.5 + 300*10)/1e6)
        self.assertEqual(result['reasoning_tokens'], 200)

    def test_usage_of_failed_or_incomplete_attempts_is_counted(self):
        event = {'event': 'response_received', 'response_status': 'incomplete',
                 'usage': {'input_tokens': 100, 'output_tokens': 100}}
        self.assertAlmostEqual(cost([event,event], 'gpt-6-luna')['estimated_standard_api_cost_usd'], .00012)


if __name__ == '__main__':
    unittest.main()
