"""
SGM 156-QUBIT FULL CHIP v2 -- Marginal-Based Fitness
=====================================================
Author: Andrew Dorman (ACD421)
Date: May 6, 2026

v1 failed: Hellinger fidelity = 0 at 156 qubits because 4096 shots
across 2^156 possible bitstrings produces zero bitstring overlap.

v2 fix: Use PER-QUBIT MARGINALS as the fitness signal.
- Target = 156 marginal probabilities P(q_i=1) from hardware
- Fitness = cosine similarity of marginal vectors
- Measurable with 4096 shots at ANY qubit count
- SGM evolution + locking proceeds on these 156-dimensional vectors
- Survivorship = marginal accuracy per free parameter

Batched evolution: 5 candidates per job, best-of-5 selection.
40 generations x 5 candidates = 200 evaluations in 40 hardware jobs.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stdout.flush()

import numpy as np
import json
import os
import time
import math
from datetime import datetime, timezone
from scipy.optimize import curve_fit

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PI = math.pi
IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "ABF780RcTfC4WTHh-97XGWm7v5UWO2kATufcNZxGcpxS")

N_QUBITS = 156
ANSATZ_LAYERS = 2
SHOTS_EVOLVE = 4096
SHOTS_VALIDATE = 16384
SHOTS_TARGET = 16384
GENERATIONS = 40
CANDIDATES_PER_GEN = 5     # batched: 5 candidates per hardware job
MUTATION_COUNT = 8          # more mutations for 468 params
LOCK_THRESHOLD = 0.025
LOCK_WINDOW = 8
BACKEND_NAME = "ibm_fez"

CHECKPOINTS = [0.0, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95]


def flush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def n_params(n_qubits, n_layers):
    return n_qubits * (n_layers + 1)


def build_ansatz(n_qubits, n_layers, angles):
    """RY + nearest-neighbor CZ ansatz."""
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(angles[idx]), q)
            idx += 1
        pairs = [(i, i+1) for i in range(layer % 2, n_qubits - 1, 2)]
        for i, j in pairs:
            qc.cz(i, j)
    for q in range(n_qubits):
        qc.ry(float(angles[idx]), q)
        idx += 1
    qc.measure_all()
    return qc


def counts_to_marginals(counts, n_qubits, shots):
    """Extract per-qubit P(1) from measurement counts."""
    marginals = np.zeros(n_qubits)
    for bitstring, count in counts.items():
        bits = bitstring.zfill(n_qubits)
        for q in range(n_qubits):
            if bits[q] == '1':
                marginals[q] += count
    return marginals / shots


def marginal_fitness(measured_marginals, target_marginals):
    """
    Cosine similarity of marginal vectors.
    Range: -1 to 1, higher = better match.
    Shift to 0-1 range.
    """
    dot = np.dot(measured_marginals, target_marginals)
    norm_m = np.linalg.norm(measured_marginals)
    norm_t = np.linalg.norm(target_marginals)
    if norm_m < 1e-10 or norm_t < 1e-10:
        return 0.0
    cos_sim = dot / (norm_m * norm_t)
    return (cos_sim + 1.0) / 2.0  # map [-1,1] to [0,1]


def marginal_mse(measured, target):
    """Mean squared error of marginals. Lower = better."""
    return np.mean((measured - target) ** 2)


def combined_fitness(measured, target):
    """Combined: high cosine sim + low MSE. Range 0-1."""
    cos = marginal_fitness(measured, target)
    mse = marginal_mse(measured, target)
    # MSE of 0 = perfect, MSE of 0.25 = random (all 0.5 vs all 0/1)
    mse_score = max(0, 1.0 - mse / 0.25)
    return 0.5 * cos + 0.5 * mse_score


def run_156q():
    n_p = n_params(N_QUBITS, ANSATZ_LAYERS)

    flush("=" * 70)
    flush("  SGM 156-QUBIT v2 -- MARGINAL-BASED FITNESS")
    flush("=" * 70)
    flush(f"  Qubits: {N_QUBITS} | Parameters: {n_p}")
    flush(f"  Hilbert dimension: 2^{N_QUBITS}")
    flush(f"  Generations: {GENERATIONS} x {CANDIDATES_PER_GEN} candidates")
    flush(f"  Shots: evolve={SHOTS_EVOLVE} validate={SHOTS_VALIDATE}")
    flush(f"  Mutations: {MUTATION_COUNT} fixed")
    flush("=" * 70)

    flush("\n  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
    backend = service.backend(BACKEND_NAME)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    flush(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")

    rng = np.random.RandomState(42)

    # ============================================================
    # Step 1: Generate target marginals on hardware
    # ============================================================
    flush(f"\n  Step 1: Target generation ({SHOTS_TARGET} shots)...")
    target_angles = rng.uniform(-PI, PI, n_p)
    target_qc = build_ansatz(N_QUBITS, ANSATZ_LAYERS, target_angles)
    target_transpiled = pm.run(target_qc)
    flush(f"  Target circuit depth: {target_transpiled.depth()}")

    sampler = SamplerV2(mode=backend)
    t0 = time.time()
    target_job = sampler.run([(target_transpiled, None, SHOTS_TARGET)])
    flush(f"  Target job: {target_job.job_id()}")
    target_result = target_job.result()
    target_counts = target_result[0].data.meas.get_counts()
    target_marginals = counts_to_marginals(target_counts, N_QUBITS, SHOTS_TARGET)
    flush(f"  Target acquired: {time.time()-t0:.1f}s")
    flush(f"  Target marginal range: [{target_marginals.min():.3f}, {target_marginals.max():.3f}]")
    flush(f"  Target marginal mean: {target_marginals.mean():.3f}")
    flush(f"  Target marginal std: {target_marginals.std():.3f}")

    # ============================================================
    # Step 2: SGM Evolution with batched candidates
    # ============================================================
    flush(f"\n  Step 2: SGM Evolution ({GENERATIONS} gens x {CANDIDATES_PER_GEN} candidates)...")

    best_angles = rng.uniform(-PI, PI, n_p)
    locked = np.zeros(n_p, dtype=bool)
    angle_history = np.zeros((GENERATIONS, n_p))

    # Evaluate initial
    init_qc = build_ansatz(N_QUBITS, ANSATZ_LAYERS, best_angles)
    init_transpiled = pm.run(init_qc)
    init_job = sampler.run([(init_transpiled, None, SHOTS_EVOLVE)])
    init_result = init_job.result()
    init_counts = init_result[0].data.meas.get_counts()
    init_marginals = counts_to_marginals(init_counts, N_QUBITS, SHOTS_EVOLVE)
    best_fitness = combined_fitness(init_marginals, target_marginals)
    best_marginals = init_marginals.copy()

    flush(f"  Initial fitness: {best_fitness:.6f}")
    flush(f"  Initial MSE: {marginal_mse(init_marginals, target_marginals):.6f}")

    snapshots = {}
    snapshots[0.0] = (best_angles.copy(), locked.copy(), best_fitness, best_marginals.copy())
    captured = {0.0}
    job_ids = [init_job.job_id()]

    t_evo_start = time.time()

    for gen in range(GENERATIONS):
        free_idx = np.where(~locked)[0]
        n_free = len(free_idx)

        # Generate CANDIDATES_PER_GEN mutants
        candidates = []
        for c in range(CANDIDATES_PER_GEN):
            cand = best_angles.copy()
            if n_free > 0:
                n_mut = min(MUTATION_COUNT, n_free)
                mut_idx = rng.choice(free_idx, n_mut, replace=False)
                cand[mut_idx] += rng.normal(0, 0.3, n_mut)
            candidates.append(cand)

        # Build and transpile all candidates
        transpiled_list = []
        for cand in candidates:
            qc = build_ansatz(N_QUBITS, ANSATZ_LAYERS, cand)
            transpiled_list.append(pm.run(qc))

        # Submit ALL candidates as one batch job
        pubs = [(tc, None, SHOTS_EVOLVE) for tc in transpiled_list]
        job = sampler.run(pubs)
        result = job.result()
        job_ids.append(job.job_id())

        # Evaluate each candidate
        best_gen_fitness = best_fitness
        best_gen_angles = best_angles.copy()
        best_gen_marginals = best_marginals.copy()

        for c in range(CANDIDATES_PER_GEN):
            counts = result[c].data.meas.get_counts()
            marg = counts_to_marginals(counts, N_QUBITS, SHOTS_EVOLVE)
            fit = combined_fitness(marg, target_marginals)

            if fit > best_gen_fitness:
                best_gen_fitness = fit
                best_gen_angles = candidates[c].copy()
                best_gen_marginals = marg.copy()

        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_angles = best_gen_angles.copy()
            best_marginals = best_gen_marginals.copy()

        angle_history[gen] = best_angles

        # SGM locking
        if gen >= LOCK_WINDOW:
            window = angle_history[gen - LOCK_WINDOW:gen + 1]
            for p in range(n_p):
                if not locked[p]:
                    if np.ptp(window[:, p]) < LOCK_THRESHOLD:
                        locked[p] = True

        lock_pct = np.mean(locked)
        mse = marginal_mse(best_marginals, target_marginals)

        # Snapshots
        for cp in CHECKPOINTS:
            if cp not in captured and lock_pct >= cp:
                snapshots[cp] = (best_angles.copy(), locked.copy(), best_fitness, best_marginals.copy())
                captured.add(cp)
                flush(f"  >> Snapshot @ {cp*100:.0f}% lock (gen {gen}, fit={best_fitness:.6f})")

        elapsed = time.time() - t_evo_start
        if gen % 5 == 0 or gen == GENERATIONS - 1:
            flush(f"  gen={gen:03d} fit={best_fitness:.6f} mse={mse:.6f} "
                  f"lock={lock_pct*100:5.1f}% free={n_free:4d}/{n_p} [{elapsed:.0f}s]")

    evo_time = time.time() - t_evo_start
    final_lock = np.mean(locked)
    flush(f"\n  Evolution: {evo_time:.0f}s, {len(job_ids)} hardware jobs")
    flush(f"  Final fitness: {best_fitness:.6f}, lock: {final_lock*100:.1f}%")

    # Final snapshot
    snapshots[final_lock] = (best_angles.copy(), locked.copy(), best_fitness, best_marginals.copy())

    # ============================================================
    # Step 3: High-shot validation
    # ============================================================
    flush(f"\n  Step 3: Validation ({len(snapshots)} snapshots, {SHOTS_VALIDATE} shots each)...")

    # Batch all validation circuits in one job
    val_lock_pcts = sorted(snapshots.keys())
    val_transpiled = []
    for lp in val_lock_pcts:
        angles, _, _, _ = snapshots[lp]
        qc = build_ansatz(N_QUBITS, ANSATZ_LAYERS, angles)
        val_transpiled.append(pm.run(qc))

    val_pubs = [(tc, None, SHOTS_VALIDATE) for tc in val_transpiled]
    flush(f"  Submitting {len(val_pubs)} validation circuits...")
    val_job = sampler.run(val_pubs)
    flush(f"  Validation job: {val_job.job_id()}")
    val_result = val_job.result()

    hw_fidelities = {}
    hw_mses = {}
    hw_ratios = {}
    baseline_fit = None

    flush(f"\n  {'Lock%':>6} {'Fitness':>8} {'MSE':>8} {'Ratio':>8} {'Free':>6}")
    flush(f"  {'-'*45}")

    for i, lp in enumerate(val_lock_pcts):
        _, lock_mask, _, _ = snapshots[lp]
        counts = val_result[i].data.meas.get_counts()
        marg = counts_to_marginals(counts, N_QUBITS, SHOTS_VALIDATE)
        fit = combined_fitness(marg, target_marginals)
        mse = marginal_mse(marg, target_marginals)
        n_free = max(1, int(np.sum(~lock_mask)))

        hw_fidelities[lp] = fit
        hw_mses[lp] = mse

        if baseline_fit is None:
            baseline_fit = fit

        ratio = (fit / n_free) * n_p / baseline_fit if baseline_fit > 0 else 0
        hw_ratios[lp] = ratio

        flush(f"  {lp*100:>5.1f}% {fit:>8.6f} {mse:>8.6f} {ratio:>8.2f} {n_free:>6}")

    # ============================================================
    # Step 4: Alpha extraction
    # ============================================================
    lps = sorted([k for k in hw_ratios if 0.05 < k < 0.99])
    if len(lps) >= 3:
        lp_arr = np.array(lps)
        r_arr = np.array([hw_ratios[k] for k in lps])
        try:
            def exp_model(x, a, c):
                return c * np.exp(a * x)
            popt, _ = curve_fit(exp_model, lp_arr, r_arr, p0=[1, 1], maxfev=5000)
            alpha = popt[0]
            y_pred = exp_model(lp_arr, *popt)
            ss_res = np.sum((r_arr - y_pred) ** 2)
            ss_tot = np.sum((r_arr - np.mean(r_arr)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
        except Exception as e:
            flush(f"  Curve fit failed: {e}")
            alpha, r2 = 0.0, 0.0
    else:
        alpha, r2 = 0.0, 0.0

    flush(f"\n{'='*70}")
    flush(f"  156-QUBIT RESULT (PURE HARDWARE)")
    flush(f"{'='*70}")
    flush(f"  Alpha = {alpha:.4f}")
    flush(f"  R^2   = {r2:.4f}")
    flush(f"  Hilbert dimension: 2^{N_QUBITS}")
    flush(f"  Parameters: {n_p}")
    flush(f"  Final fitness: {best_fitness:.6f}")
    flush(f"  Final lock: {final_lock*100:.1f}%")
    flush(f"  Hardware jobs: {len(job_ids) + 2}")
    flush(f"  Total QPU time: ~{evo_time:.0f}s evolution")
    flush(f"  Target job: {target_job.job_id()}")
    flush(f"  Validation job: {val_job.job_id()}")
    flush(f"{'='*70}")

    # Compare with scaling data
    flush(f"\n  SCALING CONTEXT:")
    flush(f"  {'Qubits':>7} {'HW Alpha':>10}")
    flush(f"  {'-'*20}")
    scaling = [(4, 3.376), (6, 3.511), (8, 12.751), (10, 16.713),
               (12, 15.712), (14, 11.604), (16, 13.556)]
    for q, a in scaling:
        flush(f"  {q:>7} {a:>10.3f}")
    flush(f"  {N_QUBITS:>7} {alpha:>10.3f}  << THIS RUN")

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = {
        "experiment": "SGM 156-Qubit Full Chip v2 (marginal fitness)",
        "timestamp": ts,
        "backend": BACKEND_NAME,
        "n_qubits": N_QUBITS,
        "n_params": n_p,
        "hilbert_dim_log2": N_QUBITS,
        "ansatz_layers": ANSATZ_LAYERS,
        "generations": GENERATIONS,
        "candidates_per_gen": CANDIDATES_PER_GEN,
        "mutation_count": MUTATION_COUNT,
        "alpha": float(alpha),
        "r2": float(r2),
        "final_fitness": float(best_fitness),
        "final_lock_pct": float(final_lock),
        "hw_fidelities": {str(k): float(v) for k, v in hw_fidelities.items()},
        "hw_mses": {str(k): float(v) for k, v in hw_mses.items()},
        "hw_ratios": {str(k): float(v) for k, v in hw_ratios.items()},
        "target_marginals": target_marginals.tolist(),
        "final_marginals": best_marginals.tolist(),
        "target_job": target_job.job_id(),
        "validation_job": val_job.job_id(),
        "evolution_job_ids": job_ids[:15],
        "evo_time_s": evo_time,
        "scaling_context": scaling + [(N_QUBITS, float(alpha))],
    }
    out_path = os.path.join(DATA_DIR, f"fullchip_156q_v2_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    flush(f"\n  Data: {out_path}")

    return out


if __name__ == "__main__":
    result = run_156q()
    flush("\n  156-qubit experiment complete.")
