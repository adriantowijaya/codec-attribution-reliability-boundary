import json
import subprocess
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class ManifestIntegrity(unittest.TestCase):
    def test_required_manifests_present(self):
        names = '\n'.join(p.name for p in (ROOT / 'manifests').rglob('*') if p.is_file())
        for token in ['source_parent_manifest', 'split_manifest', 'canonical_manifest', 'derivative_manifest', 'speech_parent_manifest']:
            self.assertIn(token, names)
if __name__ == '__main__':
    unittest.main()
