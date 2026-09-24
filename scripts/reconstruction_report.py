#!/usr/bin/env python3
"""Compare filtered fields against separately stored hidden reference fields."""
import argparse
import csv
import json
import math
from pathlib import Path


def fields(path, phase=None):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if phase is not None:
        if rows and "phase" not in rows[0]:
            raise ValueError("phase requested but reconstruction has no phase column")
        rows = [row for row in rows if row["phase"] == phase]
    return {(int(row["step"]), int(row["ix"]), int(row["iy"])): float(row["vorticity"])
            for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("analysis", "forecast"))
    args = parser.parse_args()
    truth, prediction = fields(args.truth), fields(args.reconstruction, args.phase)
    with args.observations.open() as stream:
        next(stream)
        sensors = {(int(row["step"]), int(row["x"]), int(row["y"])) for row in csv.DictReader(stream)}
    if not prediction or not prediction.keys() <= truth.keys():
        parser.error("reference fields must cover every reconstructed point")
    result = {}
    for name, keys in (("all_grid", list(prediction)), ("unobserved_grid", [key for key in prediction if key not in sensors])):
        if not keys:
            parser.error("no points available for " + name)
        squared = sum((prediction[key] - truth[key]) ** 2 for key in keys)
        reference = sum(truth[key] ** 2 for key in keys)
        result[name] = {"points": len(keys), "rmse": math.sqrt(squared / len(keys)),
                        "relative_l2": math.sqrt(squared / reference) if reference else None}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
