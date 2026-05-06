"""
DEPTH SCALING: SGM vs Decoherence on 156 Qubits
================================================
Author: Andrew Dorman (ACD421)
Date: May 6, 2026

Push circuit depth until quantum dies. Then lock with SGM. Does it survive?
Classical control at matched param counts. One run answers:
  - Does quantum alpha exceed classical at high param counts?
  - Does SGM extend useful circuit depth past the decoherence wall?
  - What is the maximum dimension SGM can ride on quantum hardware?

Depths: 2, 5, 10, 20, 30, 50 layers on 156 qubits.
Params: 468 to 7,956. All on ibm_fez.
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

PI = math.pi
IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "ABF780RcTfC4WTHh-97XGWm7v5UWO2kATufcNZxGcpxS")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_QUBITS = 156
DEPTHS = [2, 5, 10, 20, 30, 50]
LOCK_PCTS = [0.0, 0.50, 0.90, 0.95, 0.99]
SHOTS = 4096
NOISE_STD = 0.8
BACKEND_NAME = "ibm_fez"


def flush(*a, **k):
    print(*a, **k, flush=True)


def n_params(nq, nl):
    return nq * (nl + 1)


def build_ansatz(nq, nl, angles):
    from qiskit.circuit import QuantumCircuit
    qc = QuantumCircuit(nq)
    idx = 0
    for layer in range(nl):
        for q in range(nq):
            qc.ry(float(angles[idx]), q)
            idx += 1
        for i in range(layer % 2, nq - 1, 2):
            qc.cz(i, i + 1)
    for q in range(nq):
        qc.ry(float(angles[idx]), q)
        idx += 1
    qc.measure_all()
    return qc


def counts_to_marginals(counts, nq, shots):
    m = np.zeros(nq)
    for bs, cnt in counts.items():
        bits = bs.zfill(nq)
        for q in range(nq):
            if bits[q] == '1':
                m[q] += cnt
    return m / shots


def marginal_fitness(meas, targ):
    dot = np.dot(meas, targ)
    nm, nt = np.linalg.norm(meas), np.linalg.norm(targ)
    if nm < 1e-10 or nt < 1e-10:
        return 0.0
    cos = dot / (nm * nt)
    mse = np.mean((meas - targ) ** 2)
    return 0.5 * ((cos + 1) / 2) + 0.5 * max(0, 1 - mse / 0.25)


def classical_sgm_forced(n_p, lock_pcts, noise_std=0.8, seed=42):
    """Classical forced-lock experiment at given param count."""
    rng_t = np.random.RandomState(seed + 123)
    target = rng_t.uniform(-PI, PI, n_p)
    target_vec = target  # in classical, "marginals" = the vector itself

    rng_c = np.random.RandomState(seed + 999)
    results = {}
    baseline = None

    for lp in lock_pcts:
        n_lock = int(n_p * lp)
        corrupted = target.copy()
        if n_lock < n_p:
            free_idx = np.arange(n_lock, n_p)
            corrupted[free_idx] += rng_c.normal(0, noise_std, len(free_idx))

        # Fitness = cosine similarity + MSE score (same as quantum)
        dot = np.dot(corrupted, target_vec)
        nm = np.linalg.norm(corrupted)
        nt = np.linalg.norm(target_vec)
        cos = dot / (nm * nt) if nm > 0 and nt > 0 else 0
        mse = np.mean((corrupted - target_vec) ** 2)
        fit = 0.5 * ((cos + 1) / 2) + 0.5 * max(0, 1 - mse / 0.25)

        n_free = max(1, n_p - n_lock)
        if baseline is None:
            baseline = fit
        ratio = (fit / n_free) * n_p / baseline if baseline > 0 else 0
        results[lp] = {"fitness": fit, "ratio": ratio, "n_free": n_free}

    # Alpha
    lps = sorted([k for k in results if 0.05 < k < 0.99])
    if len(lps) >= 3:
        la = np.array(lps)
        ra = np.array([results[k]["ratio"] for k in lps])
        try:
            def em(x, a, c):
                return c * np.exp(a * x)
            po, _ = curve_fit(em, la, ra, p0=[1, 1], maxfev=5000)
            yp = em(la, *po)
            ss_r = np.sum((ra - yp) ** 2)
            ss_t = np.sum((ra - np.mean(ra)) ** 2)
            alpha, r2 = po[0], max(0, 1 - ss_r / (ss_t + 1e-12))
        except:
            alpha, r2 = 0, 0
    else:
        alpha, r2 = 0, 0

    return alpha, r2, results


def run():
    flush("=" * 70)
    flush("  DEPTH SCALING: SGM vs DECOHERENCE, 156 QUBITS")
    flush("=" * 70)
    flush(f"  Depths: {DEPTHS}")
    flush(f"  Lock%: {LOCK_PCTS}")
    flush(f"  Backend: {BACKEND_NAME}")
    flush("=" * 70)

    # Classical control first (free, fast)
    flush("\n  CLASSICAL CONTROL:")
    flush(f"  {'Depth':>6} {'Params':>7} {'Alpha':>8} {'R^2':>6}")
    flush(f"  {'-'*35}")
    classical_results = {}
    for nl in DEPTHS:
        np_ = n_params(N_QUBITS, nl)
        a, r2, res = classical_sgm_forced(np_, LOCK_PCTS)
        classical_results[nl] = {"alpha": a, "r2": r2, "n_params": np_, "results": res}
        flush(f"  {nl:>6} {np_:>7} {a:>8.3f} {r2:>6.3f}")

    # Hardware
    flush("\n  CONNECTING TO IBM...")
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
    backend = service.backend(BACKEND_NAME)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = SamplerV2(mode=backend)
    flush(f"  Backend: {backend.name}")

    quantum_results = {}

    for nl in DEPTHS:
        np_ = n_params(N_QUBITS, nl)
        flush(f"\n  {'='*55}")
        flush(f"  DEPTH {nl} | {np_} params | {N_QUBITS}q")
        flush(f"  {'='*55}")

        # Generate target on hardware
        rng = np.random.RandomState(42 + nl)
        target_angles = rng.uniform(-PI, PI, np_)
        tqc = build_ansatz(N_QUBITS, nl, target_angles)
        ttc = pm.run(tqc)
        flush(f"  Target circuit depth after transpile: {ttc.depth()}")

        tjob = sampler.run([(ttc, None, SHOTS * 2)])
        flush(f"  Target job: {tjob.job_id()}")
        tres = tjob.result()
        target_marg = counts_to_marginals(tres[0].data.meas.get_counts(), N_QUBITS, SHOTS * 2)

        # Check: is the target itself scrambled? (marginals all ~0.5 = noise)
        target_spread = np.std(target_marg)
        flush(f"  Target marginal std: {target_spread:.4f} (>0.05 = structured, <0.01 = noise)")

        # Build forced-lock circuits
        rng2 = np.random.RandomState(999 + nl)
        circuits = []
        for lp in LOCK_PCTS:
            n_lock = int(np_ * lp)
            angles = target_angles.copy()
            if n_lock < np_:
                free_idx = np.arange(n_lock, np_)
                angles[free_idx] += rng2.normal(0, NOISE_STD, len(free_idx))
            qc = build_ansatz(N_QUBITS, nl, angles)
            circuits.append(pm.run(qc))

        pubs = [(c, None, SHOTS) for c in circuits]
        flush(f"  Submitting {len(pubs)} circuits...")
        t0 = time.time()
        job = sampler.run(pubs)
        flush(f"  Job: {job.job_id()}")
        result = job.result()
        elapsed = time.time() - t0
        flush(f"  Done: {elapsed:.1f}s")

        baseline = None
        hw_results = {}
        flush(f"  {'Lock%':>6} {'Fitness':>9} {'Free':>6} {'Ratio':>8}")
        for i, lp in enumerate(LOCK_PCTS):
            counts = result[i].data.meas.get_counts()
            marg = counts_to_marginals(counts, N_QUBITS, SHOTS)
            fit = marginal_fitness(marg, target_marg)
            n_lock = int(np_ * lp)
            n_free = max(1, np_ - n_lock)
            if baseline is None:
                baseline = fit
            ratio = (fit / n_free) * np_ / baseline if baseline > 0 else 0
            hw_results[lp] = {"fitness": float(fit), "ratio": float(ratio), "n_free": n_free}
            flush(f"  {lp*100:>5.0f}% {fit:>9.6f} {n_free:>6} {ratio:>8.2f}x")

        # Alpha
        lps = sorted([k for k in hw_results if 0.05 < k < 0.99])
        if len(lps) >= 3:
            la = np.array(lps)
            ra = np.array([hw_results[k]["ratio"] for k in lps])
            try:
                def em(x, a, c):
                    return c * np.exp(a * x)
                po, _ = curve_fit(em, la, ra, p0=[1, 1], maxfev=5000)
                yp = em(la, *po)
                ss_r = np.sum((ra - yp) ** 2)
                ss_t = np.sum((ra - np.mean(ra)) ** 2)
                q_alpha, q_r2 = po[0], max(0, 1 - ss_r / (ss_t + 1e-12))
            except:
                q_alpha, q_r2 = 0, 0
        else:
            q_alpha, q_r2 = 0, 0

        c_alpha = classical_results[nl]["alpha"]
        ratio_qc = q_alpha / c_alpha if c_alpha > 0 else 0

        flush(f"\n  Q alpha={q_alpha:.3f} R2={q_r2:.3f} | C alpha={c_alpha:.3f} | Q/C={ratio_qc:.2f}x")

        # Does SGM rescue a dead circuit?
        unlocked_fit = hw_results[0.0]["fitness"]
        locked99_fit = hw_results[0.99]["fitness"]
        rescue = locked99_fit - unlocked_fit
        flush(f"  Unlocked fitness: {unlocked_fit:.4f} | 99% locked: {locked99_fit:.4f} | Rescue: {rescue:+.4f}")

        quantum_results[nl] = {
            "depth": nl,
            "n_params": np_,
            "q_alpha": float(q_alpha),
            "q_r2": float(q_r2),
            "c_alpha": float(c_alpha),
            "ratio_qc": float(ratio_qc),
            "target_marginal_std": float(target_spread),
            "unlocked_fitness": float(unlocked_fit),
            "locked99_fitness": float(locked99_fit),
            "rescue": float(rescue),
            "job_id": job.job_id(),
            "target_job": tjob.job_id(),
            "elapsed_s": elapsed,
            "hw_results": {str(k): v for k, v in hw_results.items()},
        }

    # SUMMARY
    flush(f"\n{'='*70}")
    flush("  DEPTH SCALING RESULTS")
    flush(f"{'='*70}")
    flush(f"  {'Depth':>6} {'Params':>7} {'Q_Alpha':>9} {'C_Alpha':>9} {'Q/C':>6} {'Rescue':>8} {'Signal':>10}")
    flush(f"  {'-'*65}")

    for nl in DEPTHS:
        q = quantum_results[nl]
        signal = "Q>C" if q["ratio_qc"] > 1.5 else ("Q~C" if q["ratio_qc"] > 0.7 else "C>Q")
        rescued = "RESCUED" if q["rescue"] > 0.02 else "n/a"
        flush(f"  {nl:>6} {q['n_params']:>7} {q['q_alpha']:>9.3f} {q['c_alpha']:>9.3f} "
              f"{q['ratio_qc']:>5.2f}x {q['rescue']:>+7.4f} {signal:>5} {rescued:>8}")

    # THE QUESTION
    flush(f"\n  DECOHERENCE WALL:")
    for nl in DEPTHS:
        q = quantum_results[nl]
        state = "ALIVE" if q["unlocked_fitness"] > 0.6 else ("DYING" if q["unlocked_fitness"] > 0.52 else "DEAD")
        flush(f"    depth={nl:>3}: unlocked={q['unlocked_fitness']:.4f} ({state}) "
              f"| 99%locked={q['locked99_fitness']:.4f} | rescue={q['rescue']:+.4f}")

    any_rescue = any(quantum_results[nl]["rescue"] > 0.02 for nl in DEPTHS)
    any_qc_win = any(quantum_results[nl]["ratio_qc"] > 1.5 for nl in DEPTHS)

    flush(f"\n{'='*70}")
    if any_rescue:
        flush("  SGM EXTENDS CIRCUIT DEPTH PAST DECOHERENCE: YES")
    else:
        flush("  SGM EXTENDS CIRCUIT DEPTH: NOT OBSERVED")
    if any_qc_win:
        flush("  QUANTUM ALPHA > CLASSICAL AT DEPTH: YES")
    else:
        flush("  QUANTUM ALPHA > CLASSICAL: NO (classical matches or exceeds)")
    flush(f"{'='*70}")

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_DIR, f"depth_scaling_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "Depth Scaling: SGM vs Decoherence",
            "timestamp": ts,
            "n_qubits": N_QUBITS,
            "depths": DEPTHS,
            "classical": {str(k): {"alpha": v["alpha"], "r2": v["r2"], "n_params": v["n_params"]}
                         for k, v in classical_results.items()},
            "quantum": quantum_results,
        }, f, indent=2, default=str)
    flush(f"\n  Data: {out_path}")


if __name__ == "__main__":
    run()
    flush("\n  Depth scaling complete.")
