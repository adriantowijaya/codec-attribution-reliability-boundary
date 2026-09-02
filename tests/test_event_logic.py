import json
import subprocess
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class EventLogic(unittest.TestCase):
    def test_48_non_event(self):
        crit = json.loads((ROOT / 'config' / 'expected_results.json').read_text(encoding='utf-8'))['event_logic']['48']
        self.assertFalse(crit['dominant_error_share'] >= crit['threshold'])
        self.assertFalse(crit['full_event'])
    def test_positive_events(self):
        data = json.loads((ROOT / 'config' / 'expected_results.json').read_text(encoding='utf-8'))
        for rate in ['80', '96']:
            crit = data['event_logic'][rate]
            self.assertEqual(crit['dominant_error_share'] >= crit['threshold'], crit['full_event'])
if __name__ == '__main__':
    unittest.main()
