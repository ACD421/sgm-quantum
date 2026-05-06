"""
DEFINITIVE SGM QUANTUM EXPERIMENT
==================================
Author: Andrew Dorman (ACD421)
Date: May 6, 2026

Answers every open question in one run:
  1. Classical alpha at matched parameter counts (CPU, free)
  2. Evolutionary SGM on 8q hardware (true convergence, angle tracking)
  3. Evolutionary SGM on 156q hardware (full chip, marginal fitness)
  4. Stabilizer analysis: do converged angles cluster toward Clifford?
  5. Logical qubit test: if Clifford structure exists, how many logical qubits?

The script that settles it.
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
from collections import Counter

PI = math.pi

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
FIGURES_DIR = os.path.join(REPO_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

IBM_TOKEN = os.environ.get("IBM_QUANTUM_TOKEN", "ABF780RcTfC4WTHh-97XGWm7v5UWO2kATufcNZxGcpxS")
BACKEND_NAME = "ibm_fez"


def flush(*a, **k):
    print(*a, **k, flush=True)


# ================================================================
# UTILITIES
# ================================================================

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


def extract_alpha(lock_pcts, ratios):
    la = np.array(lock_pcts)
    ra = np.array(ratios)
    mask = (la > 0.05) & (la < 0.99) & (ra > 0)
    if np.sum(mask) < 3:
        return 0.0, 0.0
    try:
        def em(x, a, c):
            return c * np.exp(a * x)
        po, _ = curve_fit(em, la[mask], ra[mask], p0=[1, 1], maxfev=5000)
        yp = em(la[mask], *po)
        ss_r = np.sum((ra[mask] - yp) ** 2)
        ss_t = np.sum((ra[mask] - np.mean(ra[mask])) ** 2)
        return po[0], max(0, 1 - ss_r / (ss_t + 1e-12))
    except:
        return 0.0, 0.0


def clifford_distance(angle):
    """Distance from angle to nearest multiple of pi/2."""
    a = angle % (2 * PI)
    cliffs = np.array([0, PI / 2, PI, 3 * PI / 2])
    dists = np.minimum(np.abs(a - cliffs), 2 * PI - np.abs(a - cliffs))
    return np.min(dists)


def nearest_clifford(angle):
    """Round angle to nearest multiple of pi/2."""
    a = angle % (2 * PI)
    cliffs = np.array([0, PI / 2, PI, 3 * PI / 2])
    dists = np.minimum(np.abs(a - cliffs), 2 * PI - np.abs(a - cliffs))
    return cliffs[np.argmin(dists)]


# ================================================================
# PHASE 1: CLASSICAL CONTROL (matched parameter counts)
# ================================================================

def classical_sgm(n_params, target, generations=200, mutation_count=5,
                  lock_threshold=0.02, lock_window=15, seed=42):
    """Classical SGM on a vector optimization problem. Returns alpha, R^2."""
    rng = np.random.RandomState(seed)
    best = rng.uniform(-PI, PI, n_params)
    locked = np.zeros(n_params, dtype=bool)
    history = np.zeros((generations, n_params))
    best_fit = -np.sum((best - target) ** 2) / n_params  # negative MSE

    sgm_curve = []
    for gen in range(generations):
        cand = best.copy()
        free = np.where(~locked)[0]
        nf = len(free)
        if nf > 0:
            nm = min(mutation_count, nf)
            mi = rng.choice(free, nm, replace=False)
            cand[mi] += rng.normal(0, 0.3, nm)
        fit = -np.sum((cand - target) ** 2) / n_params
        if fit > best_fit:
            best_fit = fit
            best = cand.copy()
        history[gen] = best
        if gen >= lock_window:
            w = history[gen - lock_window:gen + 1]
            for p in range(n_params):
                if not locked[p] and np.ptp(w[:, p]) < lock_threshold:
                    locked[p] = True
        lp = np.mean(locked)
        nfree = max(1, np.sum(~locked))
        fpf = (-best_fit / nfree) * n_params
        sgm_curve.append((lp, -best_fit, fpf, int(nfree)))

    # Extract alpha
    lps = [c[0] for c in sgm_curve]
    rats = [c[2] / (sgm_curve[0][2] + 1e-12) for c in sgm_curve]
    return extract_alpha(lps, rats)


def run_phase1():
    flush("\n" + "=" * 70)
    flush("  PHASE 1: CLASSICAL CONTROL (matched parameter counts)")
    flush("=" * 70)

    param_counts = [16, 24, 32, 40, 56, 68, 468]
    results = []

    for np_ in param_counts:
        rng = np.random.RandomState(123)
        target = rng.uniform(-PI, PI, np_)
        alpha, r2 = classical_sgm(np_, target, generations=200, seed=42)
        results.append({"n_params": np_, "alpha": alpha, "r2": r2})
        flush(f"  {np_:>4} params: alpha={alpha:.4f}, R^2={r2:.4f}")

    flush(f"\n  Classical alpha range: {min(r['alpha'] for r in results):.4f} "
          f"to {max(r['alpha'] for r in results):.4f}")
    return results


# ================================================================
# PHASE 2+3: EVOLUTIONARY SGM ON HARDWARE
# ================================================================

def run_evolutionary_hardware(nq, nl, generations, candidates_per_gen,
                              mutation_count, lock_threshold, lock_window,
                              backend, pm, sampler, target_marg, seed=42):
    """
    Full evolutionary SGM on hardware. Returns:
    - sgm_curve: list of (lock_pct, fitness, fit_per_free, n_free)
    - angle_history: full angle trajectory
    - final_locked: boolean mask
    - final_angles: best angles
    """
    from qiskit.circuit import QuantumCircuit

    np_ = n_params(nq, nl)
    rng = np.random.RandomState(seed)
    best_angles = rng.uniform(-PI, PI, np_)
    locked = np.zeros(np_, dtype=bool)
    angle_history = []

    # Initial evaluation
    qc = build_ansatz(nq, nl, best_angles)
    tc = pm.run(qc)
    job = sampler.run([(tc, None, 4096)])
    res = job.result()
    counts = res[0].data.meas.get_counts()
    best_marg = counts_to_marginals(counts, nq, 4096)
    best_fitness = marginal_fitness(best_marg, target_marg)

    sgm_curve = [(0.0, best_fitness, best_fitness, np_)]
    angle_history.append(best_angles.copy())

    t0 = time.time()
    for gen in range(generations):
        free = np.where(~locked)[0]
        nf = len(free)

        # Generate candidates
        cands = []
        for _ in range(candidates_per_gen):
            c = best_angles.copy()
            if nf > 0:
                nm = min(mutation_count, nf)
                mi = rng.choice(free, nm, replace=False)
                c[mi] += rng.normal(0, 0.3, nm)
            cands.append(c)

        # Batch evaluate
        tcs = [pm.run(build_ansatz(nq, nl, c)) for c in cands]
        pubs = [(t, None, 4096) for t in tcs]
        job = sampler.run(pubs)
        res = job.result()

        for i, c in enumerate(cands):
            cts = res[i].data.meas.get_counts()
            mg = counts_to_marginals(cts, nq, 4096)
            ft = marginal_fitness(mg, target_marg)
            if ft > best_fitness:
                best_fitness = ft
                best_angles = c.copy()
                best_marg = mg.copy()

        angle_history.append(best_angles.copy())

        # SGM locking (SLOW threshold for granularity)
        if gen >= lock_window and len(angle_history) > lock_window:
            window = np.array(angle_history[-lock_window:])
            for p in range(np_):
                if not locked[p] and np.ptp(window[:, p]) < lock_threshold:
                    locked[p] = True

        lp = np.mean(locked)
        nfree = max(1, np.sum(~locked))
        fpf = (best_fitness / nfree) * np_
        sgm_curve.append((lp, best_fitness, fpf, int(nfree)))

        if gen % 10 == 0 or gen == generations - 1:
            flush(f"    gen={gen:03d} fit={best_fitness:.6f} lock={lp*100:5.1f}% "
                  f"free={nfree:>4}/{np_} [{time.time()-t0:.0f}s]")

    return sgm_curve, np.array(angle_history), locked, best_angles


# ================================================================
# PHASE 4: STABILIZER ANALYSIS
# ================================================================

def analyze_stabilizer_structure(angles, locked, nq, nl):
    """
    Check if converged angles cluster toward Clifford points.
    Returns analysis dict.
    """
    np_ = len(angles)
    locked_angles = angles[locked]
    n_locked = len(locked_angles)

    if n_locked == 0:
        return {"has_structure": False, "reason": "no locked angles"}

    # Per-angle Clifford distance
    dists = np.array([clifford_distance(a) for a in locked_angles])
    expected_random = PI / 8  # 0.3927

    # Effective per-qubit rotation (sum across layers)
    effective_ry = np.zeros(nq)
    for layer in range(nl + 1):
        for q in range(nq):
            idx = layer * nq + q
            if idx < np_ and locked[idx]:
                effective_ry[q] += angles[idx]

    eff_dists = np.array([clifford_distance(a) for a in effective_ry])

    # Statistical test: are distances significantly smaller than random?
    from scipy.stats import ttest_1samp
    if len(dists) > 5:
        t_stat, p_value = ttest_1samp(dists, expected_random)
    else:
        t_stat, p_value = 0, 1

    # Count near-Clifford
    near_cliff = {
        "0.05": int(np.sum(dists < 0.05)),
        "0.10": int(np.sum(dists < 0.10)),
        "0.20": int(np.sum(dists < 0.20)),
        "pi/8": int(np.sum(dists < PI / 8)),
    }

    eff_near = {
        "0.10": int(np.sum(eff_dists < 0.10)),
        "0.20": int(np.sum(eff_dists < 0.20)),
        "pi/8": int(np.sum(eff_dists < PI / 8)),
    }

    has_structure = p_value < 0.05 and dists.mean() < expected_random * 0.8

    return {
        "n_locked": n_locked,
        "mean_clifford_distance": float(dists.mean()),
        "expected_random_distance": float(expected_random),
        "ratio": float(dists.mean() / expected_random),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "near_clifford_counts": near_cliff,
        "effective_per_qubit_near": eff_near,
        "effective_mean_distance": float(eff_dists.mean()),
        "has_structure": has_structure,
    }


# ================================================================
# PHASE 5: LOGICAL QUBIT TEST
# ================================================================

def test_logical_qubits(angles, locked, nq, nl):
    """
    If locked angles are near Clifford, round them, build Clifford circuit,
    extract stabilizer state, count logical qubits and code distance.
    """
    try:
        from qiskit.quantum_info import Clifford, StabilizerState
        from qiskit.circuit import QuantumCircuit
    except ImportError:
        return {"tested": False, "reason": "qiskit.quantum_info not available"}

    np_ = len(angles)

    # Round locked angles to nearest Clifford
    rounded = angles.copy()
    n_rounded = 0
    for i in range(np_):
        if locked[i]:
            rounded[i] = nearest_clifford(angles[i])
            n_rounded += 1

    # Build circuit with rounded angles (no measurement)
    qc = QuantumCircuit(nq)
    idx = 0
    for layer in range(nl):
        for q in range(nq):
            a = rounded[idx]
            idx += 1
            # Map pi/2 multiples to Clifford gates
            a_mod = a % (2 * PI)
            if abs(a_mod) < 0.01 or abs(a_mod - 2 * PI) < 0.01:
                pass  # identity
            elif abs(a_mod - PI / 2) < 0.01:
                qc.sx(q)
                qc.s(q)  # RY(pi/2) = S.SX in Clifford
            elif abs(a_mod - PI) < 0.01:
                qc.y(q)
            elif abs(a_mod - 3 * PI / 2) < 0.01:
                qc.sdg(q)
                qc.sx(q)  # RY(3pi/2)
            else:
                # Non-Clifford angle on a locked param -- skip this test
                return {
                    "tested": False,
                    "reason": f"Locked angle {i} = {a:.4f} not near Clifford",
                    "n_non_clifford": int(np.sum(np.array([clifford_distance(rounded[j])
                                     for j in range(np_) if locked[j]]) > 0.1)),
                }
        for i_cz in range(layer % 2, nq - 1, 2):
            qc.cz(i_cz, i_cz + 1)

    # Final layer
    for q in range(nq):
        a = rounded[idx]
        idx += 1
        a_mod = a % (2 * PI)
        if abs(a_mod) < 0.01 or abs(a_mod - 2 * PI) < 0.01:
            pass
        elif abs(a_mod - PI / 2) < 0.01:
            qc.sx(q)
            qc.s(q)
        elif abs(a_mod - PI) < 0.01:
            qc.y(q)
        elif abs(a_mod - 3 * PI / 2) < 0.01:
            qc.sdg(q)
            qc.sx(q)
        else:
            return {"tested": False, "reason": "Non-Clifford in final layer"}

    # Try to convert to Clifford
    try:
        cliff = Clifford(qc)
        stab = StabilizerState(cliff)

        # The stabilizer state is a [[n, 0, d]] code (encodes 0 logical qubits)
        # because it's a pure state
        # To find logical qubits, we'd need a SUBSPACE, not a state
        # A stabilizer state has n independent stabilizers for n qubits = 0 logical qubits

        # But we CAN measure code distance: minimum weight of destabilizer
        # For now, just report that it IS a valid stabilizer state
        return {
            "tested": True,
            "is_stabilizer": True,
            "n_qubits": nq,
            "logical_qubits": 0,  # pure state = 0 logical qubits
            "reason": "Stabilizer state confirmed. Pure state encodes 0 logical qubits. "
                      "Logical qubits require a CODE (subspace), not a single state. "
                      "The locked circuit defines a state, not a code.",
            "n_rounded": n_rounded,
            "circuit_depth": qc.depth(),
        }
    except Exception as e:
        return {
            "tested": True,
            "is_stabilizer": False,
            "reason": f"Clifford conversion failed: {str(e)}",
        }


# ================================================================
# MAIN
# ================================================================

def run_definitive():
    flush("=" * 70)
    flush("  DEFINITIVE SGM QUANTUM EXPERIMENT")
    flush("  Every question. One run.")
    flush("=" * 70)

    all_results = {}

    # ---- PHASE 1 ----
    classical_results = run_phase1()
    all_results["phase1_classical"] = classical_results

    # ---- PHASE 2+3: HARDWARE ----
    flush("\n" + "=" * 70)
    flush("  PHASE 2+3: EVOLUTIONARY SGM ON IBM HERON")
    flush("=" * 70)

    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    flush("  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=IBM_TOKEN)
    backend = service.backend(BACKEND_NAME)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    sampler = SamplerV2(mode=backend)
    flush(f"  Backend: {backend.name}")

    hw_sizes = [8, 156]
    nl = 2

    for nq in hw_sizes:
        np_ = n_params(nq, nl)
        flush(f"\n  {'='*55}")
        flush(f"  {nq}q | {np_} params | Hilbert 2^{nq}")
        flush(f"  {'='*55}")

        # Generate target on hardware
        rng = np.random.RandomState(123 + nq)
        target_angles = rng.uniform(-PI, PI, np_)
        tqc = build_ansatz(nq, nl, target_angles)
        ttc = pm.run(tqc)
        flush(f"  Generating target on hardware...")
        tjob = sampler.run([(ttc, None, 16384)])
        tres = tjob.result()
        tcounts = tres[0].data.meas.get_counts()
        target_marg = counts_to_marginals(tcounts, nq, 16384)
        flush(f"  Target job: {tjob.job_id()}")

        # Evolutionary SGM
        gens = 60 if nq <= 16 else 40
        cands = 5
        muts = 5 if nq <= 16 else 8
        lt = 0.005  # VERY slow locking for granularity
        lw = 12

        flush(f"  Running evolutionary SGM ({gens} gens, {cands} candidates)...")
        curve, hist, locked, best_angles = run_evolutionary_hardware(
            nq, nl, gens, cands, muts, lt, lw, backend, pm, sampler, target_marg
        )

        # Extract alpha
        baseline = curve[0][2] if curve[0][2] > 0 else 1
        lps = [c[0] for c in curve]
        rats = [c[2] / baseline for c in curve]
        alpha, r2 = extract_alpha(lps, rats)
        final_lock = np.mean(locked)
        final_fit = curve[-1][1]

        flush(f"\n  Alpha = {alpha:.4f}, R^2 = {r2:.4f}")
        flush(f"  Final lock: {final_lock*100:.1f}%, fitness: {final_fit:.6f}")

        # ---- PHASE 4: STABILIZER ANALYSIS ----
        flush(f"\n  Stabilizer analysis...")
        stab_analysis = analyze_stabilizer_structure(best_angles, locked, nq, nl)
        flush(f"  Mean Clifford distance: {stab_analysis['mean_clifford_distance']:.4f} "
              f"(expected random: {stab_analysis['expected_random_distance']:.4f})")
        flush(f"  Ratio: {stab_analysis['ratio']:.3f}")
        flush(f"  p-value: {stab_analysis['p_value']:.4f}")
        flush(f"  Has Clifford structure: {stab_analysis['has_structure']}")

        # ---- PHASE 5: LOGICAL QUBIT TEST ----
        if stab_analysis['has_structure']:
            flush(f"\n  Logical qubit test...")
            lq_result = test_logical_qubits(best_angles, locked, nq, nl)
            flush(f"  Result: {lq_result.get('reason', 'unknown')}")
        else:
            lq_result = {
                "tested": False,
                "reason": "No Clifford clustering. Stabilizer test not applicable."
            }
            flush(f"  Skipping logical qubit test (no Clifford structure).")

        all_results[f"q{nq}"] = {
            "n_qubits": nq,
            "n_params": np_,
            "alpha": float(alpha),
            "r2": float(r2),
            "final_lock_pct": float(final_lock),
            "final_fitness": float(final_fit),
            "target_job": tjob.job_id(),
            "stabilizer_analysis": stab_analysis,
            "logical_qubit_test": lq_result,
            "sgm_curve_summary": {
                "n_points": len(curve),
                "lock_range": [float(min(lps)), float(max(lps))],
                "fitness_range": [float(min(c[1] for c in curve)), float(max(c[1] for c in curve))],
            },
        }

    # ---- SUMMARY ----
    flush(f"\n{'='*70}")
    flush("  DEFINITIVE RESULTS")
    flush(f"{'='*70}")

    # Classical vs Quantum
    flush(f"\n  CLASSICAL CONTROL (matched params):")
    for r in classical_results:
        flush(f"    {r['n_params']:>4} params: alpha={r['alpha']:.4f}")

    flush(f"\n  QUANTUM HARDWARE (IBM Heron):")
    for key in sorted(all_results):
        if key.startswith("q"):
            r = all_results[key]
            flush(f"    {r['n_qubits']:>4}q ({r['n_params']} params): "
                  f"alpha={r['alpha']:.4f}, R^2={r['r2']:.4f}")

    # Stabilizer verdict
    flush(f"\n  STABILIZER STRUCTURE:")
    for key in sorted(all_results):
        if key.startswith("q"):
            r = all_results[key]
            sa = r['stabilizer_analysis']
            flush(f"    {r['n_qubits']}q: ratio={sa['ratio']:.3f}, "
                  f"p={sa['p_value']:.4f}, structure={'YES' if sa['has_structure'] else 'NO'}")

    # Logical qubit verdict
    flush(f"\n  LOGICAL QUBITS:")
    for key in sorted(all_results):
        if key.startswith("q"):
            r = all_results[key]
            lq = r['logical_qubit_test']
            flush(f"    {r['n_qubits']}q: {lq.get('reason', 'not tested')}")

    # THE ANSWER
    flush(f"\n{'='*70}")
    any_structure = any(
        all_results[k]['stabilizer_analysis']['has_structure']
        for k in all_results if k.startswith("q")
    )
    if any_structure:
        flush("  VERDICT: Evolutionary convergence produces Clifford clustering.")
        flush("  SGM MAY discover stabilizer codes on existing Heron hardware.")
        flush("  Further investigation required for code distance measurement.")
    else:
        flush("  VERDICT: Evolutionary convergence does NOT produce Clifford clustering.")
        flush("  Survivorship amplification operates through parameter redundancy,")
        flush("  not quantum error correction. No logical qubits discovered.")
        flush("  The effect is real but the mechanism is classical-like.")
    flush(f"{'='*70}")

    # Save
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(DATA_DIR, f"definitive_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    flush(f"\n  Data: {out_path}")

    return all_results


if __name__ == "__main__":
    run_definitive()
    flush("\n  Definitive experiment complete.")
