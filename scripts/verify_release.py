import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "verification"
TOL = 1e-12

REQUIRED_DIRS = ["metadata", "docs", "environment", "config", "manifests", "src", "scripts", "frozen_outputs", "tests"]
REQUIRED_FILES = ["README.md", "CHANGELOG.md", "AUTHORS.md", "LICENSE", "DATA_LICENSE.md", "CITATION.cff", "metadata/HUMAN_METADATA_GATE.md", "docs/CLAIM_BOUNDARY.md", "docs/THIRD_PARTY_DATA.md", "config/expected_results.json", "manifests/checksums.sha256"]
SECRET_PATTERNS = ["github" + "_pat_", "g" + "hp_", "ZENODO" + "_TOKEN", "ZENODO" + "_SANDBOX_TOKEN", "api" + "_key", "pass" + "word", "PRIVATE" + " KEY", "C:" + "\\Users\\", "C:" + "\\Workspace\\"]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def add(checks, name, passed, detail="", kind="RELEASE_GATE"):
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail, "kind": kind})

def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def dominant_counts():
    rows = read_csv(ROOT / "frozen_outputs" / "predictions" / "AQ2C1B_all_predictions.csv")
    result = {}
    for rate in ["48", "64", "80", "96"]:
        items = [r for r in rows if r["true_codec"] == "AAC" and r["nominal_rate_kbps"] == rate]
        counts = Counter(r["predicted_codec"] for r in items)
        errors = len(items) - counts.get("AAC", 0)
        wrong = {"MP3": counts.get("MP3", 0), "Opus": counts.get("Opus", 0)}
        dominant, count = max(wrong.items(), key=lambda kv: kv[1])
        result[rate] = {"aac_total": len(items), "aac_correct": counts.get("AAC", 0), "aac_errors": errors, "predicted_mp3_errors": wrong["MP3"], "predicted_opus_errors": wrong["Opus"], "dominant_wrong_family": dominant, "dominant_error_share": count / errors}
    return result

def verify_checksums(checks):
    ok = True
    for line in (ROOT / "manifests" / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        target = ROOT / rel
        if not target.exists() or sha256_file(target) != digest:
            ok = False
            add(checks, "checksum " + rel, False, "missing or digest mismatch")
    add(checks, "release checksums match", ok)

def scan_security(checks):
    hits = []
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("verification/"):
            continue
        if path.name.lower() == ".env" or "credential" in path.name.lower():
            hits.append({"file": rel, "pattern": "credential_file", "value": "[REDACTED]"})
        if path.stat().st_size > 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pat in SECRET_PATTERNS:
            if pat in text:
                hits.append({"file": rel, "pattern": pat, "value": "[REDACTED]"})
    add(checks, "static security scan", len(hits) == 0, f"{len(hits)} hits")
    return hits

def expected_checks(checks):
    data = json.loads((ROOT / "config" / "expected_results.json").read_text(encoding="utf-8"))
    add(checks, "Discovery parent count", data["discovery"]["parent_count"] == 82)
    add(checks, "Discovery split 49/16/17", data["discovery"]["split"] == {"train": 49, "validation": 16, "test": 17})
    add(checks, "Discovery represented-endpoint BA", approx(data["discovery"]["represented_endpoint_BA"], 0.8529411764705882))
    add(checks, "Discovery held-out 64 BA", approx(data["discovery"]["heldout_64_BA"], 0.392156862745098))
    add(checks, "Discovery permutation p", approx(data["discovery"]["permutation_p"], 0.20267973202679732))
    add(checks, "Confirmation parent count 60", data["confirmation"]["parent_count"] == 60)
    add(checks, "Confirmation endpoint BA", approx(data["confirmation"]["endpoint_BA"], 0.8416666666666667))
    add(checks, "48 kbps full event = NO", data["events"]["48_full_event"] is False)
    add(checks, "64 kbps replication = YES", data["events"]["64_discovery_rate_replication"] is True)
    add(checks, "80 kbps full event = YES", data["events"]["80_full_event"] is True)
    add(checks, "96 kbps full event = YES", data["events"]["96_full_event"] is True)
    add(checks, "Alternative representation BA", approx(data["generality"]["alternative_representation_endpoint_BA"], 0.6083333333))
    add(checks, "Speech primary BA", approx(data["generality"]["speech_primary_endpoint_BA"], 0.3708333333))
    add(checks, "Speech RF BA", approx(data["generality"]["speech_RF_endpoint_BA"], 0.3541666667))
    add(checks, "Speech MFCC BA", approx(data["generality"]["speech_MFCC_endpoint_BA"], 0.3916666667))
    counts = dominant_counts()
    expected_family = {"48": "MP3", "64": "MP3", "80": "MP3", "96": "Opus"}
    for rate in ["48", "64", "80", "96"]:
        expected = data["dominant_error_share"][rate]
        computed = counts[rate]["dominant_error_share"]
        add(checks, f"{rate} kbps dominant-error share exact", abs(computed - expected) <= TOL, f"computed={computed} expected={expected}", "EXACT_NUMERIC_CHECK")
        add(checks, f"{rate} kbps dominant wrong family", counts[rate]["dominant_wrong_family"] == expected_family[rate], f"computed={counts[rate]['dominant_wrong_family']} expected={expected_family[rate]}", "EXACT_NUMERIC_CHECK")
    for rate in ["48", "80", "96"]:
        crit = data["event_logic"][rate]
        computed = counts[rate]["dominant_error_share"] >= data["dominant_error_threshold"]
        add(checks, f"{rate} kbps dominant-error criterion", computed is crit["full_event"], f"computed={computed} expected={crit['full_event']}", "BOOLEAN_DECISION_CHECK")

def main():
    checks = []
    for d in REQUIRED_DIRS:
        add(checks, "directory " + d, (ROOT / d).is_dir())
    for f in REQUIRED_FILES:
        add(checks, "file " + f, (ROOT / f).is_file())
    for f in ["AUTHORS_PENDING.md", "LICENSE_DECISION_REQUIRED.md", ".zenodo.json", "metadata/CITATION.cff.template", "metadata/zenodo_metadata.template.json"]:
        add(checks, "placeholder absent " + f, not (ROOT / f).exists(), kind="METADATA_RELEASE_CONTRACT")
    raw_ext = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".opus", ".ogg"}
    add(checks, "no unexpected raw-audio payload", len([p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in raw_ext]) == 0)
    add(checks, "no binary installers", len([p for p in ROOT.rglob("*") if p.is_file() and p.suffix.lower() in {".exe", ".msi"}]) == 0)
    security_hits = scan_security(checks)
    names = "\n".join(p.name for p in (ROOT / "manifests").rglob("*") if p.is_file())
    for token in ["source_parent_manifest", "split_manifest", "canonical_manifest", "derivative_manifest", "speech_parent_manifest"]:
        add(checks, token + " exists", token in names)
    all_names = "\n".join(p.name for p in ROOT.rglob("*") if p.is_file())
    for token in ["AQ2B0_feature_spec", "AQ2D0BH_DESIGN_FREEZE", "parent_permutation", "bootstrap", "centroid", "model_config"]:
        add(checks, "method artifact " + token, token in all_names)
    for figure in ["FIG2_SIX_RATE_RELIABILITY_PROFILE.csv", "FIG3_AAC_BOUNDARY_GEOMETRY.csv", "TAB1_CONTROLLED_EVIDENCE_DESIGN.csv", "TAB2_C1B_PROSPECTIVE_GATE_SUMMARY.csv", "TAB3_GENERALITY_AND_CLAIM_BOUNDARY.csv"]:
        add(checks, "publication data " + figure, (ROOT / "frozen_outputs" / "publication_data" / figure).exists())
    expected_checks(checks)
    verify_checksums(checks)
    passed = all(c["status"] == "PASS" for c in checks)
    REPORT_DIR.mkdir(exist_ok=True)
    report = {"verification_passed": passed, "checks": checks, "security_hits": security_hits, "dominant_error_counts": dominant_counts()}
    (REPORT_DIR / "VERIFY_RELEASE_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Verify Release Report", "", f"Overall: {'PASS' if passed else 'FAIL'}", ""]
    for c in checks:
        lines.append(f"- {c['name']}: {c['status']} {c.get('detail', '')}".rstrip())
    (REPORT_DIR / "VERIFY_RELEASE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("VERIFY_RELEASE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
