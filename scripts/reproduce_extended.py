#!/usr/bin/env python3
"""Run the extended CPU case study and record commands plus source fingerprints."""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/release/hzl-flow"))
    parser.add_argument("--output", type=Path, default=Path("results/extended-case-study"))
    parser.add_argument("--quick", action="store_true", help="Use smoke-sized statistical counts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = str(args.binary.resolve())
    commands = []

    def run(command, name, *options):
        invocation = [binary, command, *map(str, options), "--output", str(args.output / name)]
        commands.append(invocation)
        subprocess.run(invocation, check=True)

    run("observe", "deterministic-data", "--sigma", 0, "--nu", 0.035, "--alpha", 0.12)
    data = args.output / "deterministic-data"
    run("calibrate", "deterministic-fit", "--data", data / "training.csv", "--method", "bfgs",
        "--gradient", "adjoint")
    run("gradient-check", "gradients", "--data", data / "training.csv", "--checkpoint-stride", 20)
    run("convergence", "convergence", "--levels", 3 if args.quick else 4,
        "--reference-n", 64 if args.quick else 128)
    run("coverage", "coverage", "--replicates", 4 if args.quick else 40,
        "--steps", 40 if args.quick else 90, "--every", 10, "--noise", 0.08)
    run("observe", "stochastic-data", "--steps", 90 if args.quick else 150, "--every", 10,
        "--noise", 0.15, "--nu", 0.03, "--alpha", 0.1, "--sigma", 0.2, "--tau", 0.5)
    stochastic = args.output / "stochastic-data"
    run("filter", "enkf-forecast", "--method", "enkf", "--data", stochastic / "training.csv",
        "--particles", 24 if args.quick else 128, "--inflation", 1.02, "--forecast-steps", 20,
        "--forecast-every", 10, "--nu", 0.03, "--alpha", 0.1, "--sigma", 0.2, "--tau", 0.5)
    run("reduced", "rom", "--sigma", 0, "--n", 32, "--training-steps", 100 if args.quick else 200,
        "--validation-steps", 100 if args.quick else 200, "--rank", 8, "--variance", 0.99999999999,
        "--validation-alpha", 0.2)
    run("surrogate", "surrogate", "--data", data / "training.csv", "--surrogate", "rbf",
        "--samples", 100 if args.quick else 1000)
    run("control-variates", "control", "--n", 16, "--steps", 50 if args.quick else 100,
        "--pilot", 8 if args.quick else 32, "--count", 16 if args.quick else 128)
    run("antithetic-parameters", "antithetic", "--n", 16, "--steps", 50 if args.quick else 100,
        "--pairs", 8 if args.quick else 128, "--log-sd", 0.25, "--sigma", 0)
    run("rare-event", "rare-event", "--n", 16, "--segment-steps", 10 if args.quick else 25,
        "--levels", 4 if args.quick else 8, "--particles", 16 if args.quick else 128,
        "--splitting-replicates", 2 if args.quick else 8,
        "--direct-count", 32 if args.quick else 1024,
        "--dissipation-threshold", 0.08 if args.quick else 0.14)

    sources = {}
    for root in (Path("src"), Path("include"), Path("tests"), Path("scripts")):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": 1, "quick": args.quick, "platform": platform.platform(),
        "python": platform.python_version(), "commands": commands, "source_sha256": sources,
        "scope": "CPU workflow; CUDA and MPI require separate hardware/runtime commands.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
