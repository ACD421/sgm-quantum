"""
SGM HARDWARE CALIBRATION -- Quick QPU validation
=================================================
Author: Andrew Dorman (ACD421)
Date: May 6, 2026

Strategy: Run SGM evolution on Aer (free), then validate the survivorship
curve on real IBM hardware. Only 2 qubit sizes (4, 8), one job each.
Minimal QPU burn: ~12 circuits total, ~2 minutes of hardware time.

This tells us: does the survivorship curve from simulation hold on real silicon?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import json
import os
import time
import math
from datetime import datetime, timezone
from scipy.optimize import curve_fit

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError,
)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PI = math.pi
IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "ABF780RcTfC4WTHh-97XGWm7v5UWO2kATufcNZxGcpxS")

# --- Config ---
QUBIT_SIZES = [4, 8]
ANSATZ_LAYERS = 3
SHOTS = 8192
SHOTS_HW = 8192

# Aer evolution config
GENERATIONS = 150
MUTATION_COUNT = 5
LOCK_THRESHOLD = 0.02
LOCK_WINDOW = 12

# Lock checkpoints to validate on hardware
LOCK_CHECKPOINTS = [0.0, 0.20, 0.40, 0.60, 0.80, 0.95]

BACKEND_NAME = "ibm_fez"


def n_params(n_qubits, n_layers):
    return n_qubits * (n_layers + 1)


def build_ansatz(n_qubits, n_layers, angles):
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(angles[idx], q)
            idx += 1
        pairs = [(i, i+1) for i in range(layer % 2, n_qubits - 1, 2)]
        for i, j in pairs:
            qc.cz(i, j)
    for q in range(n_qubits):
        qc.ry(angles[idx], q)
        idx += 1
    qc.measure_all()
    return qc


def generate_target(n_qubits, seed=123):
    rng = np.random.RandomState(seed)
    n_p = n_params(n_qubits, ANSATZ_LAYERS)
    target_angles = rng.uniform(-PI, PI, n_p)
    qc = build_ansatz(n_qubits, ANSATZ_LAYERS, target_angles)
    sim = AerSimulator()
    result = sim.run(qc, shots=SHOTS * 10, seed_simulator=999).result()
    counts = result.get_counts()
    total = sum(counts.values())
    return {k.zfill(n_qubits): v / total for k, v in counts.items()}, target_angles


def fidelity(counts, target_dist, n_qubits, shots):
    all_keys = set(target_dist.keys())
    for k in counts:
        all_keys.add(k.zfill(n_qubits))
    bc = 0.0
    for k in all_keys:
        p = counts.get(k, 0) / shots
        p2 = counts.get(k.lstrip('0') or '0', 0) / shots
        p = max(p, p2)
        q = target_dist.get(k, 0)
        bc += np.sqrt(p * q)
    return bc ** 2


def sgm_evolve_with_snapshots(n_qubits, sim, target_dist, lock_checkpoints, seed=42):
    """
    Run SGM evolution on Aer. Capture circuit snapshots at each lock checkpoint.
    Returns: snapshots dict {lock_pct: (angles, locked_mask, fitness)}
    """
    rng = np.random.RandomState(seed)
    n_p = n_params(n_qubits, ANSATZ_LAYERS)

    best_angles = rng.uniform(-PI, PI, n_p)
    locked = np.zeros(n_p, dtype=bool)
    angle_history = np.zeros((GENERATIONS, n_p))

    # Evaluate initial
    qc = build_ansatz(n_qubits, ANSATZ_LAYERS, best_angles)
    result = sim.run(qc, shots=SHOTS, seed_simulator=seed).result()
    best_fitness = fidelity(result.get_counts(), target_dist, n_qubits, SHOTS)

    snapshots = {}
    # Capture initial state (0% locked)
    snapshots[0.0] = (best_angles.copy(), locked.copy(), best_fitness)

    captured_checkpoints = {0.0}
    fitness_curve = [best_fitness]

    for gen in range(GENERATIONS):
        candidate = best_angles.copy()
        free_indices = np.where(~locked)[0]
        n_free = len(free_indices)

        if n_free > 0:
            n_mut = min(MUTATION_COUNT, n_free)
            mut_idx = rng.choice(free_indices, n_mut, replace=False)
            candidate[mut_idx] += rng.normal(0, 0.3, n_mut)

        qc = build_ansatz(n_qubits, ANSATZ_LAYERS, candidate)
        result = sim.run(qc, shots=SHOTS, seed_simulator=seed * 10000 + gen).result()
        fit = fidelity(result.get_counts(), target_dist, n_qubits, SHOTS)

        if fit >= best_fitness:
            best_fitness = fit
            best_angles = candidate.copy()

        angle_history[gen] = best_angles
        fitness_curve.append(best_fitness)

        # SGM locking
        if gen >= LOCK_WINDOW:
            window = angle_history[gen - LOCK_WINDOW:gen + 1]
            for p in range(n_p):
                if not locked[p]:
                    if np.ptp(window[:, p]) < LOCK_THRESHOLD:
                        locked[p] = True

        lock_pct = np.mean(locked)

        # Capture snapshots at checkpoints
        for cp in lock_checkpoints:
            if cp not in captured_checkpoints and lock_pct >= cp:
                snapshots[cp] = (best_angles.copy(), locked.copy(), best_fitness)
                captured_checkpoints.add(cp)
                print(f"    Snapshot @ {cp*100:.0f}% lock (gen {gen}, fit={best_fitness:.4f})")

        if gen % 50 == 0:
            print(f"    gen={gen:03d} fit={best_fitness:.4f} lock={lock_pct*100:.1f}%")

    # Final snapshot
    final_lock = np.mean(locked)
    snapshots[final_lock] = (best_angles.copy(), locked.copy(), best_fitness)

    return snapshots, fitness_curve


def run_calibration():
    print("=" * 70)
    print("  SGM HARDWARE CALIBRATION")
    print("  Aer evolution -> IBM hardware validation")
    print("=" * 70)
    print(f"  Sizes: {QUBIT_SIZES}")
    print(f"  Lock checkpoints: {LOCK_CHECKPOINTS}")
    print(f"  Backend: {BACKEND_NAME}")
    print("=" * 70)

    # Connect to IBM
    print("\n  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
    backend = service.backend(BACKEND_NAME)
    print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")

    all_results = {}

    for n_q in QUBIT_SIZES:
        n_p = n_params(n_q, ANSATZ_LAYERS)
        print(f"\n{'='*60}")
        print(f"  {n_q} QUBITS | {n_p} parameters | Hilbert dim = {2**n_q}")
        print(f"{'='*60}")

        # Phase A: Aer evolution with snapshots
        print(f"\n  Phase A: Aer evolution ({GENERATIONS} gens)...")
        sim = AerSimulator()
        target_dist, target_angles = generate_target(n_q, seed=123 + n_q)

        snapshots, fitness_curve = sgm_evolve_with_snapshots(
            n_q, sim, target_dist, LOCK_CHECKPOINTS, seed=42
        )

        print(f"\n  Captured {len(snapshots)} snapshots:")
        for lp in sorted(snapshots.keys()):
            angles, locked, fit = snapshots[lp]
            print(f"    {lp*100:5.1f}% locked: fitness={fit:.4f}, "
                  f"n_free={np.sum(~locked)}/{n_p}")

        # Phase B: Aer fidelity at each snapshot (baseline)
        print(f"\n  Phase B: Aer baseline fidelity...")
        aer_fidelities = {}
        for lp in sorted(snapshots.keys()):
            angles, locked, _ = snapshots[lp]
            qc = build_ansatz(n_q, ANSATZ_LAYERS, angles)
            result = sim.run(qc, shots=SHOTS, seed_simulator=777).result()
            fid = fidelity(result.get_counts(), target_dist, n_q, SHOTS)
            aer_fidelities[lp] = fid
            print(f"    {lp*100:5.1f}% lock: Aer fidelity = {fid:.4f}")

        # Phase C: Hardware fidelity at each snapshot
        print(f"\n  Phase C: Hardware validation on {BACKEND_NAME}...")
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

        hw_circuits = []
        hw_lock_pcts = []
        for lp in sorted(snapshots.keys()):
            angles, locked, _ = snapshots[lp]
            qc = build_ansatz(n_q, ANSATZ_LAYERS, angles)
            transpiled = pm.run(qc)
            hw_circuits.append(transpiled)
            hw_lock_pcts.append(lp)

        # Submit as one batch
        sampler = SamplerV2(mode=backend)
        pubs = [(circ, None, SHOTS_HW) for circ in hw_circuits]

        print(f"  Submitting {len(pubs)} circuits...")
        t0 = time.time()
        job = sampler.run(pubs)
        print(f"  Job: {job.job_id()}")
        result = job.result()
        hw_elapsed = time.time() - t0
        print(f"  Hardware done in {hw_elapsed:.1f}s")

        hw_fidelities = {}
        for i, lp in enumerate(hw_lock_pcts):
            pub_result = result[i]
            creg = pub_result.data.meas
            counts = creg.get_counts()
            fid = fidelity(counts, target_dist, n_q, SHOTS_HW)
            hw_fidelities[lp] = fid

        # Print comparison
        print(f"\n  {'Lock%':>6} {'Aer':>8} {'Hardware':>10} {'Delta':>8} {'Signal':>8}")
        print(f"  {'-'*45}")
        for lp in sorted(snapshots.keys()):
            aer_f = aer_fidelities.get(lp, 0)
            hw_f = hw_fidelities.get(lp, 0)
            delta = hw_f - aer_f
            print(f"  {lp*100:>5.1f}% {aer_f:>8.4f} {hw_f:>10.4f} {delta:>+8.4f}")

        # Compute survivorship ratios
        baseline_aer = aer_fidelities.get(0.0, aer_fidelities[min(aer_fidelities.keys())])
        baseline_hw = hw_fidelities.get(0.0, hw_fidelities[min(hw_fidelities.keys())])

        print(f"\n  Survivorship amplification (fidelity per free param, normalized):")
        print(f"  {'Lock%':>6} {'Aer_ratio':>10} {'HW_ratio':>10}")
        print(f"  {'-'*30}")
        aer_ratios = {}
        hw_ratios = {}
        for lp in sorted(snapshots.keys()):
            _, locked, _ = snapshots[lp]
            n_free = max(1, np.sum(~locked))
            aer_r = (aer_fidelities[lp] / n_free) * n_p / baseline_aer if baseline_aer > 0 else 0
            hw_r = (hw_fidelities[lp] / n_free) * n_p / baseline_hw if baseline_hw > 0 else 0
            aer_ratios[lp] = aer_r
            hw_ratios[lp] = hw_r
            print(f"  {lp*100:>5.1f}% {aer_r:>10.3f} {hw_r:>10.3f}")

        # Alpha extraction
        def extract_alpha(lock_pcts, ratios):
            lp_arr = np.array(lock_pcts)
            r_arr = np.array(ratios)
            mask = (lp_arr > 0.05) & (lp_arr < 0.99) & (r_arr > 0)
            if np.sum(mask) < 3:
                return 0.0, 0.0
            try:
                def exp_model(x, a, c):
                    return c * np.exp(a * x)
                popt, _ = curve_fit(exp_model, lp_arr[mask], r_arr[mask], p0=[1, 1], maxfev=5000)
                y_pred = exp_model(lp_arr[mask], *popt)
                ss_res = np.sum((r_arr[mask] - y_pred) ** 2)
                ss_tot = np.sum((r_arr[mask] - np.mean(r_arr[mask])) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-12)
                return popt[0], r2
            except:
                return 0.0, 0.0

        lps = sorted(aer_ratios.keys())
        aer_alpha, aer_r2 = extract_alpha(lps, [aer_ratios[l] for l in lps])
        hw_alpha, hw_r2 = extract_alpha(lps, [hw_ratios[l] for l in lps])

        print(f"\n  Aer alpha = {aer_alpha:.4f} (R^2 = {aer_r2:.4f})")
        print(f"  HW  alpha = {hw_alpha:.4f} (R^2 = {hw_r2:.4f})")

        all_results[f"q{n_q}"] = {
            "n_qubits": n_q,
            "n_params": n_p,
            "hilbert_dim": 2 ** n_q,
            "job_id": job.job_id(),
            "hw_elapsed_s": hw_elapsed,
            "aer_fidelities": {str(k): v for k, v in aer_fidelities.items()},
            "hw_fidelities": {str(k): v for k, v in hw_fidelities.items()},
            "aer_ratios": {str(k): v for k, v in aer_ratios.items()},
            "hw_ratios": {str(k): v for k, v in hw_ratios.items()},
            "aer_alpha": aer_alpha,
            "aer_r2": aer_r2,
            "hw_alpha": hw_alpha,
            "hw_r2": hw_r2,
            "fitness_curve": fitness_curve,
            "lock_checkpoints": {str(k): {
                "n_free": int(np.sum(~snapshots[k][1])),
                "aer_fitness": float(snapshots[k][2]),
            } for k in snapshots},
        }

    # Summary
    print(f"\n{'='*70}")
    print("  CALIBRATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Qubits':>7} {'Aer_alpha':>10} {'HW_alpha':>10} {'Aer_R2':>8} {'HW_R2':>8}")
    print(f"  {'-'*50}")
    for n_q in QUBIT_SIZES:
        r = all_results[f"q{n_q}"]
        print(f"  {n_q:>7} {r['aer_alpha']:>10.4f} {r['hw_alpha']:>10.4f} "
              f"{r['aer_r2']:>8.4f} {r['hw_r2']:>8.4f}")

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_DIR, f"hw_calibration_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "SGM Hardware Calibration",
            "timestamp": ts,
            "backend": BACKEND_NAME,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n  Data: {out_path}")

    return all_results


def plot_calibration(results):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')

    fig, axes = plt.subplots(1, len(QUBIT_SIZES), figsize=(8 * len(QUBIT_SIZES), 6))
    fig.patch.set_facecolor('#0a0a0a')
    if len(QUBIT_SIZES) == 1:
        axes = [axes]

    for idx, n_q in enumerate(QUBIT_SIZES):
        ax = axes[idx]
        r = results[f"q{n_q}"]

        # Aer curve
        aer_lps = sorted(r['aer_fidelities'].keys(), key=float)
        aer_fids = [r['aer_fidelities'][k] for k in aer_lps]
        hw_fids = [r['hw_fidelities'][k] for k in aer_lps]
        lps_pct = [float(k) * 100 for k in aer_lps]

        ax.plot(lps_pct, aer_fids, 'o-', color='#00ff00', linewidth=2, markersize=8, label='Aer (sim)')
        ax.plot(lps_pct, hw_fids, 's-', color='#ff00ff', linewidth=2, markersize=8, label=f'Hardware ({BACKEND_NAME})')

        ax.set_xlabel('Lock %', fontsize=12, color='#ffffff')
        ax.set_ylabel('Fidelity', fontsize=12, color='#ffffff')
        ax.set_title(f'{n_q} Qubits (Hilbert dim {2**n_q})\n'
                     f'Aer alpha={r["aer_alpha"]:.3f} | HW alpha={r["hw_alpha"]:.3f}',
                     fontsize=11, color='#ffffff')
        ax.legend(fontsize=10, facecolor='#1a1a1a')
        ax.tick_params(colors='#ffffff')
        ax.grid(True, alpha=0.2)

    fig.suptitle('SGM HARDWARE CALIBRATION: Aer vs IBM Quantum',
                fontsize=14, color='#ffffff', fontweight='bold')
    plt.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, "hw_calibration.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
    print(f"  Figure: {fig_path}")
    plt.close()


if __name__ == "__main__":
    results = run_calibration()
    try:
        plot_calibration(results)
    except Exception as e:
        print(f"  Plot error: {e}")
        import traceback; traceback.print_exc()
    print("\n  Calibration complete.")
