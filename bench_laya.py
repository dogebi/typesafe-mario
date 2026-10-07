"""Measure Laya on Mario: prior on/off, and the scripted stall recovery.

    python bench_laya.py [seeds] [max_decisions]

Runs the real CLI (headless) once per seed per configuration, then reports how
far Mario got, whether he died, which actions were taken, the decision latency
and -- from the `source` field of each run log -- who actually picked each
action ("laya", "stall_break", "fallback"). The configurations differ only in
`--laya-prior-t` and `--laya-stall-break`.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["123", "124", "125"])]
MAX_DECISIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
CONFIGS = [
    ("laya alone", 0.0, False),
    ("laya alone + stall-break", 0.0, True),
    ("laya + prior", 0.4, False),
    ("laya + prior + stall-break", 0.4, True),
]


def run(prior_t: float, stall_break: bool, seed: int) -> dict:
    out_dir = Path(f"artifacts/bench-prior{prior_t}-break{int(stall_break)}")
    command = [
        "uv",
        "run",
        "--no-sync",
        "typesafe-mario",
        "play",
        "--policy",
        "laya",
        "--laya-prior-t",
        str(prior_t),
        "--display",
        "none",
        "--max-decisions",
        str(MAX_DECISIONS),
        "--seed",
        str(seed),
        "--artifacts-dir",
        str(out_dir),
    ]
    if stall_break:
        command.append("--laya-stall-break")
    subprocess.run(command, check=True, capture_output=True, text=True)

    log = max(out_dir.glob("run-*.jsonl"), key=lambda path: path.stat().st_mtime)
    rows = [json.loads(line) for line in log.open()]
    max_x = max(r["state"]["player"]["x"] for r in rows)
    return {
        "seed": seed,
        "decisions": len(rows),
        "max_x": max_x,
        "died": bool(rows[-1]["terminated"]),
        "sources": Counter(r.get("source", "?") for r in rows),
        "actions": Counter(r["action"] for r in rows),
        "latency_p50": statistics.median(r["latency_ms"] for r in rows),
        "jump_prob_mean": statistics.mean(
            r["jump_needed_probability"] for r in rows if r["jump_needed_probability"] is not None
        )
        if any(r["jump_needed_probability"] is not None for r in rows)
        else None,
    }


for label, prior_t, stall_break in CONFIGS:
    print(f"\n=== {label} (prior_t={prior_t}, stall_break={stall_break}) ===")
    xs, deaths, breaks = [], 0, 0
    for seed in SEEDS:
        r = run(prior_t, stall_break, seed)
        xs.append(r["max_x"])
        deaths += int(r["died"])
        forced = r["sources"].get("stall_break", 0)
        breaks += forced
        top = ", ".join(f"{a}×{n}" for a, n in r["actions"].most_common(3))
        jump_p = "n/a" if r["jump_prob_mean"] is None else f"{r['jump_prob_mean']:.2f}"
        print(
            f"  seed {r['seed']}: x={r['max_x']:>4}  decisions={r['decisions']:>3}  "
            f"died={r['died']}  p50={r['latency_p50']:.0f}ms  jump_p={jump_p}  "
            f"forced={forced:>2}  sources={dict(r['sources'])}  {top}"
        )
    print(
        f"  mean x {statistics.mean(xs):.0f}   best {max(xs)}   deaths {deaths}/{len(SEEDS)}   "
        f"stall-break decisions {breaks}"
    )
