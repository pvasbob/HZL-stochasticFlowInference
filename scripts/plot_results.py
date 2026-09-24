#!/usr/bin/env python3
"""Create scientific figures from measured engine outputs (no simulated stand-ins)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return np.atleast_1d(np.genfromtxt(path, names=True, delimiter=",", dtype=None, encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--chains", nargs="+", type=Path)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--reduced", nargs="+", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--reconstruction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})

    def save(fig, name):
        fig.savefig(args.output / (name + ".png"), dpi=160, bbox_inches="tight")
        fig.savefig(args.output / (name + ".pdf"), bbox_inches="tight")
        plt.close(fig)

    if args.simulation:
        field = read(args.simulation / "vorticity.csv")
        diagnostics = read(args.simulation / "diagnostics.csv")
        n = int(field["ix"].max()) + 1
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        im = axes[0].imshow(field["vorticity"].reshape(n, n), origin="lower", extent=(0, 2*np.pi, 0, 2*np.pi), cmap="RdBu_r")
        axes[0].set(title="Final vorticity", xlabel="x", ylabel="y")
        axes[0].grid(False)
        fig.colorbar(im, ax=axes[0], label="Vorticity")
        axes[1].plot(diagnostics["time"], diagnostics["energy"], label="Energy")
        axes[1].plot(diagnostics["time"], diagnostics["enstrophy"], label="Enstrophy")
        axes[1].set(title="Integrated diagnostics", xlabel="Time", ylabel="Value")
        axes[1].legend()
        save(fig, "flow")

    if args.calibration:
        history = read(args.calibration / "history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].semilogy(history["evaluations"], history["nll"])
        axes[0].set(xlabel="Optimizer evaluations", ylabel="Negative log likelihood", title="Calibration convergence")
        axes[1].plot(history["iteration"], history["nu"], label="Viscosity")
        axes[1].plot(history["iteration"], history["alpha"], label="Friction")
        axes[1].set(xlabel="Iteration", ylabel="Parameter value", title="Parameter history")
        axes[1].legend()
        save(fig, "calibration")

    if args.chains:
        chains = [read(path)[args.burn:] for path in args.chains]
        if any(len(chain) < 2 for chain in chains):
            parser.error("not enough chain samples after burn-in")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for index, chain in enumerate(chains):
            axes[0, 0].plot(chain["iteration"], chain["nu"], alpha=0.6, linewidth=0.7, label=f"Chain {index+1}")
            axes[0, 1].plot(chain["iteration"], chain["alpha"], alpha=0.6, linewidth=0.7)
            axes[1, 0].scatter(chain["nu"][::5], chain["alpha"][::5], s=3, alpha=0.3)
        axes[0, 0].set(xlabel="Iteration", ylabel="Viscosity", title="Post-warmup traces")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].set(xlabel="Iteration", ylabel="Friction", title="Post-warmup traces")
        axes[1, 0].set(xlabel="Viscosity", ylabel="Friction", title="Posterior dependence")
        axes[1, 1].hist(np.concatenate([chain["nu"] for chain in chains]), bins=40, density=True)
        axes[1, 1].set(xlabel="Viscosity", ylabel="Density", title="Marginal posterior")
        save(fig, "posterior")

    if args.prediction:
        data = read(args.prediction / "predictions.csv")
        sensors = sorted(set(zip(data["x"], data["y"])))
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
        for ax, (x, y) in zip(axes.flat, sensors):
            rows = data[(data["x"] == x) & (data["y"] == y)]
            ax.fill_between(rows["step"], rows["observation_q025"], rows["observation_q975"], alpha=0.2, label="95% observation interval")
            ax.fill_between(rows["step"], rows["state_q025"], rows["state_q975"], alpha=0.4, label="95% state interval")
            ax.plot(rows["step"], rows["state_mean"], linewidth=1)
            ax.scatter(rows["step"], rows["observed"], s=12, color="black", label="Held-out observation")
            ax.set(title=f"Sensor ({x}, {y})", xlabel="Time step", ylabel="Vorticity")
        axes.flat[0].legend(fontsize=6)
        save(fig, "posterior_predictive")

    if args.reduced:
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        for path in args.reduced:
            rows = read(path / "validation.csv")
            summary = json.loads((path / "summary.json").read_text())
            ax.semilogy(rows["time"], rows["trajectory_relative_l2"], label=f"Rank {summary['rank']} trajectory")
            ax.semilogy(rows["time"], rows["projection_relative_l2"], "--", label=f"Rank {summary['rank']} projection")
        ax.set(xlabel="Time (held-out interval)", ylabel="Relative vorticity L2 error", title="Reduced-model extrapolation")
        ax.legend()
        save(fig, "reduced_model")

    if args.control:
        rows = read(args.control / "samples.csv")
        summary = json.loads((args.control / "summary.json").read_text())
        count = np.arange(1, len(rows)+1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].plot(count, np.cumsum(rows["full_energy"])/count, label="Direct Monte Carlo")
        axes[0].plot(count, np.cumsum(rows["corrected_energy"])/count, label="Control variate")
        axes[0].set(xlabel="Production samples", ylabel="Mean energy estimate", title="Running estimates")
        axes[0].legend()
        axes[1].bar(["Direct MC", "Control variate"], [summary["raw_standard_error"], summary["corrected_standard_error"]])
        axes[1].set(yscale="log", ylabel="Estimated standard error", title="Independent-pilot control variate")
        save(fig, "control_variate")

    if args.truth and args.reconstruction:
        truth, reconstruction = read(args.truth), read(args.reconstruction)
        step = int(reconstruction["step"].max())
        truth, reconstruction = truth[truth["step"] == step], reconstruction[reconstruction["step"] == step]
        truth = np.sort(truth, order=["iy", "ix"])
        reconstruction = np.sort(reconstruction, order=["iy", "ix"])
        n = int(truth["ix"].max())+1
        ref, estimate = truth["vorticity"].reshape(n,n), reconstruction["vorticity"].reshape(n,n)
        bound = max(abs(ref).max(), abs(estimate).max())
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        for ax, values, title in zip(axes, [ref, estimate, estimate-ref], ["Hidden reference", "EnKF mean", "Reconstruction error"]):
            limit = bound if title != "Reconstruction error" else max(abs(values).max(), 1e-15)
            im = ax.imshow(values, origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set(title=title, xlabel="Grid x", ylabel="Grid y")
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.75)
        save(fig, "state_reconstruction")
    print(f"Figures written to {args.output}")


if __name__ == "__main__":
    main()
