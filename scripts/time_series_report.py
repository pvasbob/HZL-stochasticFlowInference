#!/usr/bin/env python3
"""Finite-record flow statistics; requires the optional analysis environment."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(values, spacing):
    centered = values - values.mean()
    count = len(values)
    transform = np.fft.rfft(centered, n=2 * count)
    covariance = np.fft.irfft(transform * transform.conj(), n=2 * count)[:count] / count
    correlation = covariance / covariance[0] if covariance[0] > 0 else np.ones(count)
    # Initial positive paired autocorrelation sums, with lag zero treated separately.
    total = 0.0
    for lag in range(1, count - 1, 2):
        pair = correlation[lag] + correlation[lag + 1]
        if pair <= 0:
            break
        total += pair
    inefficiency = max(1.0, 1.0 + 2.0 * total)
    spectrum = spacing / count * np.abs(np.fft.rfft(centered)) ** 2
    spectrum[1:-1 if count % 2 == 0 else None] *= 2
    return {
        'mean': float(values.mean()), 'standard_deviation': float(values.std(ddof=1)),
        'estimated_effective_samples': float(count / inefficiency),
        'estimated_mean_standard_error': float(values.std(ddof=1) * np.sqrt(inefficiency / count)),
        'integrated_correlation_time': float(spacing * inefficiency),
    }, correlation, spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('simulation', type=Path)
    parser.add_argument('--burn-time', type=float, default=50)
    parser.add_argument('--block-time', type=float, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = np.genfromtxt(args.simulation / 'diagnostics.csv', delimiter=',', names=True)
    metadata = json.loads((args.simulation / 'metadata.json').read_text())
    data = data[data['time'] >= args.burn_time - 1e-9]
    if len(data) < 8 or args.block_time <= 0 or args.burn_time < 0:
        parser.error('require at least eight retained records, nonnegative burn time and positive block time')
    times = data['time']
    spacing = float(np.median(np.diff(times)))
    if spacing <= 0 or not np.allclose(np.diff(times), spacing, rtol=1e-7, atol=1e-10):
        parser.error('diagnostics must be regularly spaced without a partial final interval')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be empty or new')
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'source': str(args.simulation), 'start_time': float(times[0]),
              'end_time': float(times[-1]), 'records': len(data), 'diagnostic_spacing': spacing,
              'scope': 'Finite-record estimates conditional on approximate stationarity; no stationarity or confidence-interval coverage claim.'}
    for name in ('energy', 'enstrophy', 'dissipation', 'injection'):
        report[name], correlation, spectrum = analyze(data[name], spacing)
        np.savetxt(args.output / f'{name}_autocorrelation.csv', np.column_stack((np.arange(len(data)) * spacing, correlation)), delimiter=',', header='lag,autocorrelation', comments='')
        np.savetxt(args.output / f'{name}_psd.csv', np.column_stack((np.fft.rfftfreq(len(data), spacing), spectrum)), delimiter=',', header='frequency,one_sided_psd', comments='')
    duration = times[-1] - times[0]
    rate = data['injection'] - data['dissipation'] - 2 * metadata['alpha'] * data['energy']
    observed = float((data['energy'][-1] - data['energy'][0]) / duration)
    integrated = float(np.trapezoid(rate, times) / duration)
    report['energy_budget'] = {'observed_mean_derivative': observed, 'trapezoidal_mean_rhs': integrated,
                               'residual': observed - integrated,
                               'scope': 'Uses saved endpoint forcing and diagnostic cadence; includes sampling quadrature and time-discretization errors.'}
    blocks = []
    block_count = int(np.floor((duration + 1e-8) / args.block_time))
    for block in range(block_count):
        start = times[0] + block * args.block_time
        mask = (times >= start - 1e-9) & (times < start + args.block_time - 1e-9)
        if mask.any():
            blocks.append([start, start + args.block_time, int(mask.sum()), *[float(data[name][mask].mean()) for name in ('energy', 'enstrophy', 'dissipation', 'injection')]])
    np.savetxt(args.output / 'block_means.csv', np.asarray(blocks).reshape(-1, 7), delimiter=',', header='start,end,records,energy,enstrophy,dissipation,injection', comments='')
    report['complete_blocks'] = len(blocks)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
