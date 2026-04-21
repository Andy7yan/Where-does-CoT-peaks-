#!/usr/bin/env python3
"""
Split t1a_overview.csv into per-difficulty CSVs for pgfplots.

Usage:
    python split_t1a.py t1a_overview.csv [output_dir]

Produces:
    {output_dir}/t1a_easy.csv
    {output_dir}/t1a_medium.csv
    {output_dir}/t1a_hard.csv

Each output keeps only bin_status=ok rows, sorted by L.
Empty k_star values are written as 'nan' (pgfplots skips them).
Also prints the L* value for each difficulty (for LaTeX constants).
"""
import csv, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "t1a_overview.csv"
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(src) or "."

with open(src) as f:
    reader = list(csv.DictReader(f))

for diff in ("easy", "medium", "hard"):
    rows = [r for r in reader
            if r["difficulty"].strip() == diff
            and r["bin_status"].strip() == "ok"]
    rows.sort(key=lambda r: int(r["L"]))

    l_star = None
    for r in rows:
        if r["l_star"].strip() == "True":
            l_star = int(r["L"])
        # pgfplots needs 'nan' for missing numeric values
        if r["k_star"].strip() == "" or r["k_star"].strip().lower() in ("null", "none"):
            r["k_star"] = "nan"

    path = os.path.join(out_dir, f"t1a_{diff}.csv")
    fields = ["L", "accuracy", "accuracy_se", "k_star", "mean_tas", "tas_se"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"{diff:8s}  L*={l_star}  rows={len(rows)}  -> {path}")
