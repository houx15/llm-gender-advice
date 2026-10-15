"""Download public reference files (IPEDS, BLS, O*NET, Scorecard, ISEI, crosswalks, held-out
perplexity text) into data/reference/raw/ and fill the taxonomy attributes. Blocked URLs ->
QUESTIONS.md. Run on the login node / laptop.

  python scripts/online/cache_reference_data.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from advice_audit.reference.build import build_all  # noqa: E402
from advice_audit.reference.sources import fetch_all  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for k, v in fetch_all(a.force).items():
        print(f"{'OK     ' if v else 'MISSING'} {k}: {v}")
    build_all()
