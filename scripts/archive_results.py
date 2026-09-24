#!/usr/bin/env python3
"""Archive CSV tables, JSON metadata, and checkpoints in a typed HDF5 container."""
import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("source must be an experiment directory")
    with h5py.File(args.output, "x") as archive:
        archive.attrs["schema"] = "HZL_EXPERIMENT_ARCHIVE_V1"
        archive.attrs["source"] = str(args.source.resolve())
        for path in sorted(args.source.rglob("*")):
            if not path.is_file() or path.suffix not in {".csv", ".json", ".txt"}:
                continue
            relative = str(path.relative_to(args.source))
            if path.suffix != ".csv":
                text = path.read_text()
                if path.suffix == ".json":
                    json.loads(text)
                archive.create_dataset(relative, data=text, dtype=h5py.string_dtype("utf-8"))
                continue
            group = archive.create_group(relative)
            with path.open(newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                if header[0].startswith("HZL_"):
                    group.attrs["dataset_header"] = json.dumps(header)
                    header = next(reader)
                rows = list(reader)
            if any(len(row) != len(header) for row in rows):
                raise ValueError(f"Inconsistent row width: {path}")
            for index, name in enumerate(header):
                values = [row[index] for row in rows]
                if values and all(re.fullmatch(r"[+-]?\d+", value) for value in values):
                    integers = list(map(int, values))
                    data = np.array(integers, dtype=np.int64 if min(integers) < 0 else np.uint64)
                else:
                    try:
                        data = np.array(values, dtype=np.float64)
                    except ValueError:
                        data = np.array(values, dtype=h5py.string_dtype("utf-8"))
                group.create_dataset(name, data=data, compression="gzip")
        print(f"Archived {args.source} to {args.output}")


if __name__ == "__main__":
    main()
