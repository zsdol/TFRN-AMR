#!/usr/bin/env python
"""Aggregate runs/<exp>/results.json into the paper's Table 1-style summary
with paired t-test / Wilcoxon p-values against TFRN.

    python report.py --runs-dir runs --out report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from utils import mean_std, paired_tests


def load_all(runs_dir: str):
    rows = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        seeds = sorted(r["seeds"], key=lambda x: int(x))
        avgs = [r["seeds"][s]["avg"] for s in seeds]
        lows = [r["seeds"][s]["low"] for s in seeds]
        peaks = [r["seeds"][s]["peak"] for s in seeds]
        rows[os.path.basename(os.path.dirname(path))] = dict(
            model=r.get("model", "?"), seeds=seeds, avg=avgs, low=lows, peak=peaks)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="report.md")
    p.add_argument("--reference", default=None,
                   help="experiment name used as reference (default: name containing 'tfrn')")
    a = p.parse_args()

    rows = load_all(a.runs_dir)
    if not rows:
        print("No results.json found under", a.runs_dir)
        return
    ref = a.reference or next((k for k in rows if "tfrn" in k.lower()), None)
    if ref is None:
        ref = list(rows)[0]
    ref_avg = rows[ref]["avg"]

    lines = ["| Experiment | Model | Avg (%) mean+/-std | Low (%) | Peak (%) | Delta vs ref (pp) | t-test p | Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in rows.items():
        mu, sd = mean_std(r["avg"])
        lo, _ = mean_std(r["low"])
        pk, _ = mean_std(r["peak"])
        if name == ref or len(r["avg"]) != len(ref_avg):
            delta = tp = wp = "--"
        else:
            st = paired_tests(ref_avg, r["avg"])
            delta, tp, wp = f"{st['delta_pp']:+.2f}", f"{st['ttest_p']:.3g}", f"{st['wilcoxon_p']:.3g}"
        lines.append(f"| {name} | {r['model']} | {mu*100:.2f}+/-{sd*100:.2f} | "
                     f"{lo*100:.2f} | {pk*100:.2f} | {delta} | {tp} | {wp} |")
    text = "\n".join(lines)
    print(text)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# TFRN experiment summary (reference: {ref})\n\n{text}\n")
    print(f"[done] -> {a.out}")


if __name__ == "__main__":
    main()
