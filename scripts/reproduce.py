#!/usr/bin/env python3
"""Reproduce the first end-to-end deterministic calibration and stochastic runs."""
import argparse
import json
import platform
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/reproduced"))
    parser.add_argument("--binary", type=Path, default=Path("build/release/hzl-flow"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = str(args.binary.resolve())
    commands = []

    def run(command, *options):
        invocation = [binary, command, *map(str, options)]
        commands.append(invocation)
        subprocess.run(invocation, check=True)

    data = args.output / "observations"
    run("observe", "--sigma", 0, "--nu", 0.035, "--alpha", 0.12, "--output", data)
    run("calibrate", "--data", data / "training.csv", "--validation", data / "validation.csv",
        "--output", args.output / "calibration")
    run("simulate", "--n", 64, "--steps", 1000, "--dt", 0.005,
        "--output", args.output / "stochastic")
    run("ensemble", "--n", 32, "--count", 24, "--workers", 6,
        "--output", args.output / "ensemble")
    manifest = {"platform": platform.platform(), "commands": commands}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
