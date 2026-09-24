#!/usr/bin/env python3
"""Launch the C++ engine from a flat JSON experiment configuration."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("build/release/hzl-flow"))
    parser.add_argument("--output", help="Override the configured result directory")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("schema") != 1 or set(config) != {"schema", "command", "options"}:
        parser.error("expected schema 1 with command and options")
    if config["command"] not in {"simulate", "observe", "calibrate", "sample", "ensemble", "filter", "particle-sample", "calibrate-stochastic", "gradient-check", "reduced", "posterior-predict", "control-variates", "stress", "surrogate"}:
        parser.error("unknown command")
    options = config["options"]
    if not isinstance(options, dict):
        parser.error("options must be an object")
    if args.output:
        options["output"] = args.output
    command = [str(args.binary.resolve()), config["command"]]
    for key, value in options.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            parser.error("invalid option name")
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            parser.error("option values must be strings or numbers")
        command.extend(["--" + key, str(value)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
