import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Portable full reproduction orchestrator. RP1B executes dry-run only.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    stages = ["validate third-party corpus acquisition", "generate canonical PCM", "generate codec derivatives", "extract frozen features", "run frozen evaluations", "regenerate publication outputs"]
    print("AQ full reproduction orchestrator")
    print("data_root:", args.data_root)
    print("output_root:", args.output_root)
    print("ffmpeg:", args.ffmpeg)
    for stage in stages:
        print("DRY-RUN stage:" if args.dry_run else "stage:", stage)
    if args.dry_run:
        return 0
    if not Path(args.data_root).exists():
        print("STOP: upstream corpus data are absent or path is invalid")
        return 2
    print("STOP: non-dry-run execution is outside RP1B authorization")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
