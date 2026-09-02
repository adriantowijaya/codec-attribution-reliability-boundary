import json
import subprocess
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class ExpectedResults(unittest.TestCase):
    def test_expected_events(self):
        data = json.loads((ROOT / 'config' / 'expected_results.json').read_text(encoding='utf-8'))
        self.assertFalse(data['events']['48_full_event'])
        self.assertTrue(data['events']['64_discovery_rate_replication'])
        self.assertTrue(data['events']['80_full_event'])
        self.assertTrue(data['events']['96_full_event'])
    def test_headline_values(self):
        data = json.loads((ROOT / 'config' / 'expected_results.json').read_text(encoding='utf-8'))
        self.assertEqual(data['discovery']['parent_count'], 82)
        self.assertEqual(data['confirmation']['parent_count'], 60)
        self.assertAlmostEqual(data['confirmation']['endpoint_BA'], 0.8416666666666667)
if __name__ == '__main__':
    unittest.main()
