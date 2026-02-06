#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<val>[0-9.]+)\s+train_time:(?P<tt>[0-9.]+)ms\s+step_avg:(?P<avg>[0-9.]+)ms"
)

TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)(?:\s+train_loss:(?P<train_loss>[0-9.]+))?\s+train_time:(?P<tt>[0-9.]+)ms\s+step_avg:(?P<avg>[0-9.]+)ms"
)


def parse_log(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    val_points = []
    train_points = []

    for line in lines:
        m = VAL_RE.search(line)
        if m:
            val_points.append(
                {
                    "step": int(m.group("step")),
                    "total_steps": int(m.group("total")),
                    "val_loss": float(m.group("val")),
                    "train_time_ms": float(m.group("tt")),
                    "step_avg_ms": float(m.group("avg")),
                }
            )
            continue
        m = TRAIN_RE.search(line)
        if m:
            train_points.append(
                {
                    "step": int(m.group("step")),
                    "total_steps": int(m.group("total")),
                    "train_loss": float(m.group("train_loss")) if m.group("train_loss") is not None else None,
                    "train_time_ms": float(m.group("tt")),
                    "step_avg_ms": float(m.group("avg")),
                }
            )

    if val_points:
        best = min(val_points, key=lambda x: x["val_loss"])
        last_val = val_points[-1]
    else:
        best = None
        last_val = None

    last_train = train_points[-1] if train_points else None

    summary = {
        "log_path": str(path),
        "n_val_points": len(val_points),
        "n_train_points": len(train_points),
        "best_val": best,
        "last_val": last_val,
        "last_train": last_train,
        "val_points": val_points,
        "train_points": train_points,
    }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Parse modded-nanogpt training log")
    p.add_argument("log", type=str, help="Path to log .txt")
    p.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = p.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    res = parse_log(log_path)

    print(f"Parsed: {log_path}")
    print(f"Val points: {res['n_val_points']} | Train points: {res['n_train_points']}")
    if res["best_val"] is not None:
        b = res["best_val"]
        print(f"Best val_loss={b['val_loss']:.4f} at step {b['step']}/{b['total_steps']}")
    if res["last_val"] is not None:
        l = res["last_val"]
        print(f"Last val_loss={l['val_loss']:.4f} at step {l['step']}/{l['total_steps']}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = log_path.with_suffix(".parsed.json")
    out_path.write_text(json.dumps(res, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
