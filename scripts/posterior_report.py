#!/usr/bin/env python3
"""Rank-normalized split-chain convergence diagnostics using the standard library."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def quantile(values, probability):
    values = sorted(values)
    position = probability * (len(values) - 1)
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def split_chains(chains):
    half = min(map(len, chains)) // 2
    return [part for chain in chains for part in (chain[:half], chain[-half:])]


def rank_normalize(chains):
    sizes = [len(chain) for chain in chains]
    flat = [value for chain in chains for value in chain]
    order = sorted(range(len(flat)), key=flat.__getitem__)
    ranks = [0.0] * len(flat)
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and flat[order[end]] == flat[order[begin]]:
            end += 1
        rank = 0.5 * (begin + 1 + end)
        for index in order[begin:end]:
            ranks[index] = rank
        begin = end
    normal = statistics.NormalDist()
    transformed = [normal.inv_cdf((rank - 0.375) / (len(flat) + 0.25)) for rank in ranks]
    result, offset = [], 0
    for size in sizes:
        result.append(transformed[offset:offset + size])
        offset += size
    return result


def split_rhat(chains):
    n = len(chains[0])
    within = statistics.mean(statistics.variance(chain) for chain in chains)
    between = n * statistics.variance(statistics.mean(chain) for chain in chains)
    return math.sqrt(((n - 1) * within / n + between / n) / within) if within else None


def multi_chain_ess(chains):
    m, n = len(chains), len(chains[0])
    means = [statistics.mean(chain) for chain in chains]
    within = statistics.mean(statistics.variance(chain) for chain in chains)
    between = n * statistics.variance(means) if m > 1 else 0.0
    variance_plus = (n - 1) * within / n + between / n
    if variance_plus <= 0:
        return 0.0
    correlations = [1.0]
    for lag in range(1, n):
        variogram = statistics.mean(
            sum((chain[i] - chain[i - lag]) ** 2 for i in range(lag, n)) / (n - lag)
            for chain in chains)
        correlations.append(1 - variogram / (2 * variance_plus))
    pairs = []
    for lag in range(1, len(correlations) - 1, 2):
        pair = correlations[lag] + correlations[lag + 1]
        if pair < 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    tau = max(1.0, -1 + 2 * (1 + sum(pairs)))
    return min(float(m * n), m * n / tau)


def summarize(chains):
    split = split_chains(chains)
    pooled = [value for chain in chains for value in chain]
    rank_split = rank_normalize(split)
    median = quantile(pooled, 0.5)
    folded = rank_normalize([[abs(value - median) for value in chain] for chain in split])
    lower, upper = quantile(pooled, 0.05), quantile(pooled, 0.95)
    lower_indicator = [[float(value <= lower) for value in chain] for chain in split]
    upper_indicator = [[float(value >= upper) for value in chain] for chain in split]
    bulk = multi_chain_ess(rank_split)
    tail = min(multi_chain_ess(lower_indicator), multi_chain_ess(upper_indicator))
    sd = statistics.stdev(pooled)
    return {
        "mean": statistics.mean(pooled), "sd": sd,
        "quantile_025": quantile(pooled, 0.025), "median": median,
        "quantile_975": quantile(pooled, 0.975),
        "rank_normalized_split_rhat": max(split_rhat(rank_split), split_rhat(folded)),
        "bulk_ess": bulk, "tail_ess": tail,
        "mean_mcse": sd / math.sqrt(bulk) if bulk else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chains", nargs="+", type=Path)
    parser.add_argument("--burn", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.burn < 0 or len(args.chains) < 2:
        parser.error("provide at least two chains and nonnegative burn-in")
    chains = []
    for path in args.chains:
        with path.open() as file:
            rows = list(csv.DictReader(file))[args.burn:]
        if len(rows) < 20:
            parser.error("at least 20 samples must remain in each chain")
        chains.append(rows)
    report = {
        "method": "rank-normalized folded split R-hat and multi-chain bulk/tail ESS",
        "burn_per_chain": args.burn, "chain_files": list(map(str, args.chains)),
        "note": "Diagnostics follow modern rank/fold/split principles; thresholds are screening rules, not proof of convergence.",
        "acceptance_rates": [statistics.mean(int(row["accepted"]) for row in chain) for chain in chains],
    }
    parameters = ("nu", "alpha", "sigma", "tau") if "sigma" in chains[0][0] else ("nu", "alpha")
    for parameter in parameters:
        report[parameter] = summarize([[float(row[parameter]) for row in chain] for chain in chains])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
