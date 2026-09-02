from pathlib import Path
print((Path(__file__).resolve().parents[1] / 'environment' / 'package_versions.txt').read_text())
