#!/usr/bin/env python3
"""
Download daily VIX data from an open source and store it as CSV.

Default source:
https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import ssl
import urllib.request
from pathlib import Path


DEFAULT_URL = "https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download daily VIX CSV.")
    ap.add_argument("--url", default=DEFAULT_URL, help="Source CSV URL")
    ap.add_argument("--output", default="", help="Output CSV path")
    ap.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (use only if your local cert chain is broken).",
    )
    args = ap.parse_args()
    base_dir = Path(__file__).resolve().parent
    if not args.output:
        args.output = str(base_dir / "data" / "vix_daily.csv")

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    context = None
    if args.insecure:
        context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(args.url, timeout=30, context=context) as r:
            raw = r.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            "Failed to download VIX CSV.\n"
            "If this is an SSL certificate issue, retry with:\n"
            "python download_vix_daily.py --output ./data/vix_daily.csv --insecure\n"
            f"Original error: {e}"
        ) from e

    probe = raw.lstrip().lower()[:300]
    if "<html" in probe or "<!doctype" in probe:
        raise RuntimeError(
            "Downloaded content is HTML, not CSV. This usually means proxy/login/certificate interception.\n"
            "Use one of these options:\n"
            "1) Retry with a different network\n"
            "2) Manually download a VIX CSV and pass it to --vix-csv\n"
            "3) Skip VIX download and run prepare_kaggle_optionsdx.py without --vix-csv (uses VIX proxy)"
        )

    # Normalize schema to: date,vix
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for row in reader:
        # datasets/finance-vix usually provides Date,VIX Close
        date = row.get("Date") or row.get("date")
        vix = (
            row.get("VIX Close")
            or row.get("vix_close")
            or row.get("vix")
            or row.get("Close")
            or row.get("close")
        )
        if date is None or vix is None:
            continue
        rows.append({"date": date, "vix": vix})

    if not rows:
        raise ValueError("Downloaded file did not contain recognizable VIX columns.")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "vix"])
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out}")
    print(f"Rows: {len(rows)}")
    print(f"Source: {args.url}")


if __name__ == "__main__":
    main()
