import json
import unittest
from collections import Counter
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-12

def rows():
    with open(ROOT / "frozen_outputs" / "predictions" / "AQ2C1B_all_predictions.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def count_rate(rate):
    items = [r for r in rows() if r["true_codec"] == "AAC" and r["nominal_rate_kbps"] == str(rate)]
    counts = Counter(r["predicted_codec"] for r in items)
    errors = len(items) - counts.get("AAC", 0)
    wrong = {"MP3": counts.get("MP3", 0), "Opus": counts.get("Opus", 0)}
    dominant, n = max(wrong.items(), key=lambda kv: kv[1])
    return len(items), counts.get("AAC", 0), errors, wrong, dominant, n / errors

class DominantErrorShare(unittest.TestCase):
    def test_exact_values(self):
        expected = {48: 0.5882352941176471, 64: 0.8, 80: 0.6785714285714286, 96: 0.7272727272727273}
        for rate, value in expected.items():
            self.assertAlmostEqual(count_rate(rate)[5], value, delta=TOL)

    def test_denominator_semantics(self):
        total, correct, errors, wrong, dominant, share = count_rate(80)
        self.assertEqual(total, 60)
        self.assertEqual(correct, 4)
        self.assertEqual(errors, 56)
        self.assertNotEqual(38 / total, share)
        self.assertEqual(38 / errors, share)

    def test_dominant_family(self):
        expected = {48: "MP3", 64: "MP3", 80: "MP3", 96: "Opus"}
        for rate, family in expected.items():
            self.assertEqual(count_rate(rate)[4], family)

    def test_threshold(self):
        expected = {48: False, 80: True, 96: True}
        for rate, decision in expected.items():
            self.assertEqual(count_rate(rate)[5] >= 0.60, decision)

    def test_event_decisions_regression(self):
        data = json.loads((ROOT / "config" / "expected_results.json").read_text(encoding="utf-8"))
        self.assertFalse(data["events"]["48_full_event"])
        self.assertTrue(data["events"]["64_discovery_rate_replication"])
        self.assertTrue(data["events"]["80_full_event"])
        self.assertTrue(data["events"]["96_full_event"])

    def test_not_contract_only(self):
        data = json.loads((ROOT / "config" / "expected_results.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(count_rate(96)[5], data["dominant_error_share"]["96"], delta=TOL)

if __name__ == "__main__":
    unittest.main()
