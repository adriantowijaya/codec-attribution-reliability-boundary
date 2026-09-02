import json
import subprocess
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class Portability(unittest.TestCase):
    def test_no_private_paths(self):
        hits = []
        for path in ROOT.rglob('*'):
            if path.is_dir() or path.stat().st_size > 1024 * 1024:
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith('verification/'):
                continue
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            if 'C:\\Workspace\\' in text or 'C:\\Users\\' in text:
                hits.append(rel)
        self.assertEqual(hits, [])
if __name__ == '__main__':
    unittest.main()
