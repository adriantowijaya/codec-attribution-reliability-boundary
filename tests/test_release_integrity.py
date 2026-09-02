import json
import subprocess
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class ReleaseIntegrity(unittest.TestCase):
    def test_verify_release_passes(self):
        result = subprocess.run([sys.executable, 'scripts/verify_release.py'], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
if __name__ == '__main__':
    unittest.main()
