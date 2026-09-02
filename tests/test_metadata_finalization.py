import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class MetadataFinalization(unittest.TestCase):
    def test_final_files_exist(self):
        for rel in ["AUTHORS.md", "LICENSE", "DATA_LICENSE.md", "CITATION.cff", ".zenodo.json", "metadata/HUMAN_METADATA_GATE.md"]:
            self.assertTrue((ROOT / rel).exists(), rel)

    def test_placeholders_removed(self):
        for rel in ["AUTHORS_PENDING.md", "LICENSE_DECISION_REQUIRED.md", "metadata/CITATION.cff.template", "metadata/zenodo_metadata.template.json"]:
            self.assertFalse((ROOT / rel).exists(), rel)

    def test_creator_order_consistency(self):
        authors = (ROOT / "AUTHORS.md").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
        self.assertLess(authors.index("Tonny Hidayat"), authors.index("Adrianto Mahendra Wijaya"))
        self.assertLess(citation.index('given-names: "Tonny"'), citation.index('given-names: "Adrianto Mahendra"'))
        self.assertEqual(zenodo["creators"][0]["name"], "Hidayat, Tonny")
        self.assertEqual(zenodo["creators"][1]["name"], "Wijaya, Adrianto Mahendra")

if __name__ == "__main__":
    unittest.main()
