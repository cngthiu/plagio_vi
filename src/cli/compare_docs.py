# =========================
# file: src/cli/compare_docs.py
# =========================
import argparse
import json
from src.service.runner import run_compare

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run_compare(args.config, args.hardware, args.a, args.b, args.out)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()