#!/usr/bin/env python3
"""
random_baseline.py

Random-baseline experiment for comparisons with proposed significant set.

Baselines (same total size as T_sig^d, same split between X and Z errors):
  B1  weight-matched random: same number of patterns of each weight as
      T_sig^d (including weight 0, 1 and m+1), supports drawn uniformly
  B2  within-guarantee random: weights drawn uniformly from 1..m

Random sets are drawn from a large pool of pre-decoded random patterns
(sampling with replacement), so many random sets can be evaluated cheaply.

Usage:  python3 random_baseline.py            (all distances)
        python3 random_baseline.py 5 7        (selected distances)
"""

import sys
import time
import json
import math
import random

import numpy as np

DISTANCES = [5, 7, 9, 11, 13]
POOL_SIZE = 2000          # random patterns per (Pauli type, weight)
N_TRIALS = 5000           # random test sets per baseline and distance
SEED = 2026

FAULTS = [
    "no_boundary_neutral",
    "full_edge_growth",
    "grow_even_clusters",
    "no_merge",
    "peel_arbitrary_root",
    "peel_invert",
    "peel_no_propagate",
    "drop_defect",
    "syndrome_shift",
    "truncate_correction",
]
ALL_DECODERS = ["reference"] + FAULTS
# faults whose T_sig counts reproduce Table 8 of the manuscript exactly at
# d = 5..13; the four growth-stage faults are plausible variants of the
# manuscript's faults but do not reproduce its counts
VALIDATED = {"peel_arbitrary_root", "peel_invert", "peel_no_propagate",
             "drop_defect", "syndrome_shift", "truncate_correction"}


# --------------------------------------------------------------------------
# 1. Lattice: which stabilizers touch each data qubit
# --------------------------------------------------------------------------
def build_lattice(d):
    n = d * d
    nb = (d - 1) // 2                  # stabilizers of one type per bulk row
    p = (d * d - 1) // 2               # stabilizers of each type
    z_of = [[] for _ in range(n)]      # Z stabilizers of qubit q (detect X errors)
    x_of = [[] for _ in range(n)]      # X stabilizers of qubit q (detect Z errors)

    # bulk plaquettes: plaquette (i, j) covers rows i, i+1 and columns j, j+1
    for i in range(d - 1):
        for j in range(d - 1):
            idx = i * nb + j // 2
            qubits = [i * d + j, i * d + j + 1, (i + 1) * d + j, (i + 1) * d + j + 1]
            for q in qubits:
                if (i + j) % 2 == 0:
                    x_of[q].append(idx)
                else:
                    z_of[q].append(idx)

    base = (d - 1) * nb
    # Z boundary stabilizers: top row first, then bottom row
    for k in range(nb):
        for q in (2 * k, 2 * k + 1):
            z_of[q].append(base + k)
        for q in ((d - 1) * d + 2 * k + 1, (d - 1) * d + 2 * k + 2):
            z_of[q].append(base + nb + k)
    # X boundary stabilizers: ordered by row, right side on even rows, left on odd
    for i in range(d - 1):
        col = d - 1 if i % 2 == 0 else 0
        for q in (i * d + col, (i + 1) * d + col):
            x_of[q].append(base + i)

    return n, p, z_of, x_of


class Sector:
    """Decoding graph for one Pauli type.
    sector 'Z': Z stabilizers, X errors.  sector 'X': X stabilizers, Z errors."""

    def __init__(self, d, name):
        n, p, z_of, x_of = build_lattice(d)
        self.d, self.n, self.p, self.name = d, n, p, name
        self.stabs_of = z_of if name == "Z" else x_of
        self.B = p                                   # single boundary vertex
        self.edges = []                              # edge index = qubit index
        self.adj = [[] for _ in range(p + 1)]
        for q in range(n):
            s = self.stabs_of[q]
            u = s[0]
            v = s[1] if len(s) == 2 else self.B
            self.edges.append((u, v))
            self.adj[u].append(q)
            self.adj[v].append(q)
        # logical test operator: column 0 for X errors, row 0 for Z errors
        if name == "Z":
            self.logical = set(r * d for r in range(d))
        else:
            self.logical = set(range(d))

    def syndrome(self, error):
        defects = set()
        for q in error:
            for s in self.stabs_of[q]:
                defects ^= {s}
        return sorted(defects)


# --------------------------------------------------------------------------
# 2. Union-Find decoder with optional injected fault
# --------------------------------------------------------------------------
def uf_decode(G, defects_in, fault="reference"):
    p, B, edges, adj = G.p, G.B, G.edges, G.adj
    defects = sorted(defects_in)

    # syndrome-processing faults
    if fault == "drop_defect" and len(defects) > 0:
        defects = defects[:-1]                       # ignore the last defect
    if fault == "syndrome_shift":
        defects = sorted((v + 1) % p for v in defects)
    if len(defects) == 0:
        return []

    parent = list(range(p + 1))
    members = {v: [v] for v in range(p + 1)}
    parity = [0] * (p + 1)
    touches_boundary = [False] * (p + 1)
    touches_boundary[B] = True
    has_defect = [0] * (p + 1)
    for v in defects:
        parity[v] = 1
        has_defect[v] = 1
    support = [0] * len(edges)
    step = 2 if fault == "full_edge_growth" else 1

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def is_odd(r):
        if fault == "no_boundary_neutral":
            return parity[r] == 1
        return parity[r] == 1 and not touches_boundary[r]

    max_rounds = 4 * (p + 1)
    rounds = 0
    while True:
        roots = sorted(set(find(v) for v in defects))
        odd_roots = [r for r in roots if is_odd(r)]
        if len(odd_roots) == 0:
            break
        if fault == "grow_even_clusters":
            grow_roots = roots
        else:
            grow_roots = odd_roots
        rounds += 1
        if rounds > max_rounds:
            break

        fusion = []
        grew = False
        for r in grow_roots:
            seen = set()
            for v in members[r]:
                for q in adj[v]:
                    if support[q] < 2 and q not in seen:
                        seen.add(q)
                        support[q] = min(2, support[q] + step)
                        grew = True
                        if support[q] == 2:
                            fusion.append(q)
        if not grew:
            break

        for q in fusion:
            u, v = edges[q]
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            if len(members[ru]) < len(members[rv]):
                ru, rv = rv, ru
            parent[rv] = ru
            members[ru].extend(members[rv])
            del members[rv]
            if fault != "no_merge":          # no_merge: parity/boundary labels go stale
                parity[ru] ^= parity[rv]
                touches_boundary[ru] = touches_boundary[ru] or touches_boundary[rv]

    # peeling on the connected components of the erasure (fully grown edges)
    er_adj = {}
    for q in range(len(edges)):
        if support[q] == 2:
            u, v = edges[q]
            er_adj.setdefault(u, []).append((q, v))
            er_adj.setdefault(v, []).append((q, u))

    flag = has_defect[:]
    visited = set()
    correction = []
    starts = sorted(set(list(er_adj.keys()) + defects))
    for start in starts:
        if start in visited:
            continue
        comp = [start]
        comp_set = {start}
        k = 0
        while k < len(comp):
            v = comp[k]
            k += 1
            for (q, w) in er_adj.get(v, []):
                if w not in comp_set:
                    comp_set.add(w)
                    comp.append(w)
        visited |= comp_set

        if fault == "peel_arbitrary_root":
            root = min(comp_set)
        else:
            root = B if B in comp_set else min(comp_set)

        order = [root]
        par_edge = {root: None}
        par_vertex = {}
        k = 0
        while k < len(order):
            v = order[k]
            k += 1
            for (q, w) in er_adj.get(v, []):
                if w not in par_edge:
                    par_edge[w] = q
                    par_vertex[w] = v
                    order.append(w)

        for v in reversed(order[1:]):
            take = flag[v] == 1
            if fault == "peel_invert":
                take = not take
            if take:
                correction.append(par_edge[v])
                flag[v] ^= 1
                if fault != "peel_no_propagate":
                    flag[par_vertex[v]] ^= 1

    if fault == "truncate_correction" and len(correction) > 0:
        correction.pop()
    return correction


def verdict(G, error, correction):
    residual = set(error) ^ set(correction)
    if len(G.syndrome(residual)) > 0:
        return "RESIDUAL"
    if len(residual) == 0:
        return "EXACT"
    if len(residual & G.logical) % 2 == 1:
        return "LOGICAL"
    return "DEGENERATE_OK"


def success(v):
    return v in ("EXACT", "DEGENERATE_OK")


def evaluate_pattern(G, error):
    """Return (reference_success, array of regression flags per fault)."""
    defects = G.syndrome(error)
    ref_ok = success(verdict(G, error, uf_decode(G, defects, "reference")))
    flags = np.zeros(len(FAULTS), dtype=np.int8)
    if ref_ok:
        for k, f in enumerate(FAULTS):
            if not success(verdict(G, error, uf_decode(G, defects, f))):
                flags[k] = 1
    return ref_ok, flags


# --------------------------------------------------------------------------
# 3. T_sig generation (Algorithms 2, 3, 4 of the manuscript)
# --------------------------------------------------------------------------
def consecutive_lists(d, s):
    out = []
    c_hat = 1
    for r in range(d):                                   # horizontal
        if r == 0 or r == d - 1:
            for c in range(0, d - s + 1):
                out.append([r * d + j for j in range(c, c + s)])
        else:
            if c_hat > d - s:
                c_hat = 1
            out.append([r * d + j for j in range(c_hat, c_hat + s)])
            c_hat += 1
    r_hat = 1
    for c in range(d):                                   # vertical
        if c == 0 or c == d - 1:
            for r in range(0, d - s + 1):
                out.append([j * d + c for j in range(r, r + s)])
        else:
            if r_hat > d - s:
                r_hat = 1
            out.append([j * d + c for j in range(r_hat, r_hat + s)])
            r_hat += 1
    for r in range(0, d - s + 1):                        # principal diagonal
        out.append([j * d + j for j in range(r, r + s)])
    for r in range(0, d - s + 1):                        # anti-diagonal
        c = d - 1 - r
        out.append([(r + j) * d + (c - j) for j in range(s)])
    return out


def nonconsecutive_lists(d, s, delta):
    out = []
    sigma = (s - 1) * delta + 1
    L = d - sigma
    if L < 0:
        return out
    a0 = min(1, L)
    c_hat = a0
    for r in range(d):
        if r == 0 or r == d - 1:
            for c in range(0, L + 1):
                out.append([r * d + c + k * delta for k in range(s)])
        else:
            if c_hat > L:
                c_hat = a0
            out.append([r * d + c_hat + k * delta for k in range(s)])
            c_hat += 1
    r_hat = a0
    for c in range(d):
        if c == 0 or c == d - 1:
            for r in range(0, L + 1):
                out.append([(r + k * delta) * d + c for k in range(s)])
        else:
            if r_hat > L:
                r_hat = a0
            out.append([(r_hat + k * delta) * d + c for k in range(s)])
            r_hat += 1
    for a in range(0, L + 1):
        out.append([(a + k * delta) * d + (a + k * delta) for k in range(s)])
    for a in range(0, L + 1):
        out.append([(a + k * delta) * d + (d - 1 - a - k * delta) for k in range(s)])
    return out


def tsig_test_set(d, delta_set=(2,)):
    """List of (sector, error tuple, group) for both Pauli types."""
    m = (d - 1) // 2
    multi = []
    for s in range(2, m + 1):
        multi += consecutive_lists(d, s)
    for s in range(2, m + 1):
        for delta in delta_set:
            multi += nonconsecutive_lists(d, s, delta)
    multi += consecutive_lists(d, m + 1)
    tests = []
    for sector in ("Z", "X"):
        tests.append((sector, (), "T0"))
        for q in range(d * d):
            tests.append((sector, (q,), "T1"))
        for e in multi:
            tests.append((sector, tuple(e), "multi"))
    return tests, multi


# --------------------------------------------------------------------------
# 4. Experiment
# --------------------------------------------------------------------------
def check_layout_d5():
    """Spot-check syndromes against Table 2 of the manuscript (d = 5)."""
    Gz, Gx = Sector(5, "Z"), Sector(5, "X")

    def bits(defects, p=12):
        b = ["0"] * p
        for v in defects:
            b[v] = "1"
        s = "".join(b)
        return s[0:4] + "_" + s[4:8] + "_" + s[8:12]

    table2 = {  # q : (S_z for X error, S_x for Z error)
        0: ("0000_0000_1000", "1000_0000_0000"),
        1: ("1000_0000_1000", "1000_0000_0000"),
        4: ("0100_0000_0000", "0000_0000_1000"),
        5: ("0010_0000_0000", "1000_0000_0100"),
        7: ("1001_0000_0000", "0110_0000_0000"),
        14: ("0000_0100_0000", "0001_0000_0010"),
        20: ("0000_0010_0000", "0000_0000_0001"),
        22: ("0000_0001_0010", "0000_0010_0000"),
        24: ("0000_0000_0001", "0000_0001_0000"),
    }
    for q, (sz, sx) in table2.items():
        assert bits(Gz.syndrome([q])) == sz, (q, bits(Gz.syndrome([q])), sz)
        assert bits(Gx.syndrome([q])) == sx, (q, bits(Gx.syndrome([q])), sx)


def run_distance(d, rng, nprng, log):
    m = (d - 1) // 2
    G = {"Z": Sector(d, "Z"), "X": Sector(d, "X")}
    for name in ("Z", "X"):   # logical operator commutes with opposite-type stabilizers
        other = Sector(d, "X" if name == "Z" else "Z")
        for st in range(other.p):
            sup = [q for q in range(other.n) if st in other.stabs_of[q]]
            assert len(set(sup) & G[name].logical) % 2 == 0

    # ---- T_sig ----
    tests, multi = tsig_test_set(d)
    N = len(tests)
    tsig_full = np.zeros(len(FAULTS), dtype=int)
    tsig_multi = np.zeros(len(FAULTS), dtype=int)
    ref_fail_by_weight = {}
    zero_syn = 0
    weight_hist = {"Z": {}, "X": {}}
    t0 = time.time()
    for sector, err, group in tests:
        w = len(err)
        weight_hist[sector][w] = weight_hist[sector].get(w, 0) + 1
        if group == "multi" and len(G[sector].syndrome(err)) == 0:
            zero_syn += 1
        ref_ok, flags = evaluate_pattern(G[sector], err)
        if not ref_ok:
            ref_fail_by_weight[w] = ref_fail_by_weight.get(w, 0) + 1
            assert w > m, f"reference failed a guaranteed-correctable pattern {sector} {err}"
        tsig_full += flags
        if group == "multi":
            tsig_multi += flags
    log(f"  d={d}: |T_sig|={N}, multi-qubit lists={len(multi)}, "
        f"zero-syndrome (multi, both types)={zero_syn}, "
        f"reference failures by weight={ref_fail_by_weight}  [{time.time()-t0:.1f}s]")

    # ---- random pools: per (sector, weight) regression flags ----
    pool = {}
    weights_needed = set()
    for sector in ("Z", "X"):
        for w in weight_hist[sector]:
            weights_needed.add(w)
    for w in range(1, m + 1):
        weights_needed.add(w)
    t0 = time.time()
    n = d * d
    for sector in ("Z", "X"):
        for w in sorted(weights_needed):
            if w == 0:
                errors = [()]
            elif w == 1:
                errors = [(q,) for q in range(n)]          # exhaustive
            else:
                errors = [tuple(sorted(rng.sample(range(n), w))) for _ in range(POOL_SIZE)]
            flags = np.zeros((len(errors), len(FAULTS)), dtype=np.int8)
            for i, e in enumerate(errors):
                _, flags[i] = evaluate_pattern(G[sector], e)
            pool[(sector, w)] = flags
    log(f"  d={d}: random pools decoded [{time.time()-t0:.1f}s]")

    def draw(sector, w, count):
        f = pool[(sector, w)]
        idx = nprng.integers(0, f.shape[0], size=(N_TRIALS, count))
        return f[idx].sum(axis=1)                       # (N_TRIALS, F)

    # ---- B1: weight-matched random sets ----
    b1 = np.zeros((N_TRIALS, len(FAULTS)), dtype=int)
    for sector in ("Z", "X"):
        for w, count in weight_hist[sector].items():
            b1 += draw(sector, w, count)

    # ---- B2: within-guarantee random sets (weights uniform in 1..m) ----
    b2 = np.zeros((N_TRIALS, len(FAULTS)), dtype=int)
    half = N // 2
    for sector in ("Z", "X"):
        w_choice = nprng.integers(1, m + 1, size=(N_TRIALS, half))
        for w in range(1, m + 1):
            f = pool[(sector, w)]
            mask = (w_choice == w)
            idx = nprng.integers(0, f.shape[0], size=(N_TRIALS, half))
            b2 += (f[idx] * mask[:, :, None]).sum(axis=1)

    # ---- per-pattern regression rates and budget for 95 % detection (B1 mix) ----
    rate = {}
    for (sector, w), f in pool.items():
        rate[(sector, w)] = f.mean(axis=0)
    n95 = []
    for k in range(len(FAULTS)):
        # P(no detection | size = x*N) = prod over (sector,w) of (1-r)^(x*count)
        log_miss_full = 0.0
        for sector in ("Z", "X"):
            for w, count in weight_hist[sector].items():
                r = rate[(sector, w)][k]
                if r >= 1.0:
                    log_miss_full = -math.inf
                elif r > 0:
                    log_miss_full += count * math.log(1.0 - r)
        if log_miss_full == 0.0:
            n95.append(None)                             # never detected
        elif log_miss_full == -math.inf:
            n95.append(1)
        else:
            # need x*N patterns with x*log_miss_full <= log(0.05)
            x = math.log(0.05) / log_miss_full
            n95.append(max(1, math.ceil(x * N)))

    res = {
        "d": d, "m": m, "size": N,
        "tsig_full": tsig_full.tolist(),
        "tsig_multi": tsig_multi.tolist(),
        "B1_detect_prob": (b1 > 0).mean(axis=0).tolist(),
        "B1_mean": b1.mean(axis=0).tolist(),
        "B1_p05": np.percentile(b1, 5, axis=0).tolist(),
        "B1_min": b1.min(axis=0).tolist(),
        "B1_max": b1.max(axis=0).tolist(),
        "B2_detect_prob": (b2 > 0).mean(axis=0).tolist(),
        "B2_mean": b2.mean(axis=0).tolist(),
        "B2_min": b2.min(axis=0).tolist(),
        "n95_B1": n95,
        "ref_fail_by_weight": ref_fail_by_weight,
        "zero_syndrome_multi": zero_syn,
    }
    return res


def main():
    dists = [int(a) for a in sys.argv[1:]] or DISTANCES
    rng = random.Random(SEED)
    nprng = np.random.default_rng(SEED)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    check_layout_d5()
    log("Layout check against Table 2 (d=5): OK")
    results = []
    for d in dists:
        results.append(run_distance(d, rng, nprng, log))

    with open("random_baseline_results.json", "w") as fh:
        json.dump({"faults": FAULTS, "pool_size": POOL_SIZE, "trials": N_TRIALS,
                   "seed": SEED, "results": results}, fh, indent=1)

    # summary table
    log("")
    for r in results:
        log(f"d = {r['d']}   |T_sig| = {r['size']}")
        log(f"  {'fault':22s} {'Tsig(multi)':>11s} {'Tsig(full)':>10s} "
            f"{'B1 P(det)':>9s} {'B1 mean':>8s} {'B1 min':>6s} {'B2 P(det)':>9s} {'B2 mean':>8s} {'n95(B1)':>8s}")
        for k, f in enumerate(FAULTS):
            n95 = r["n95_B1"][k]
            log(f"  {f:22s} {r['tsig_multi'][k]:11d} {r['tsig_full'][k]:10d} "
                f"{r['B1_detect_prob'][k]:9.3f} {r['B1_mean'][k]:8.1f} {r['B1_min'][k]:6d} "
                f"{r['B2_detect_prob'][k]:9.3f} {r['B2_mean'][k]:8.1f} {str(n95):>8s}")
        log("")
    with open("random_baseline_results.csv", "w") as fh:
        fh.write("d,size,fault,matches_table8,tsig_multi,tsig_full,"
                 "B1_detect_prob,B1_mean,B1_min,B1_max,B2_detect_prob,B2_mean,n95_B1\n")
        for r in results:
            for k, f in enumerate(FAULTS):
                fh.write(f"{r['d']},{r['size']},{f},{f in VALIDATED},{r['tsig_multi'][k]},"
                         f"{r['tsig_full'][k]},{r['B1_detect_prob'][k]:.4f},{r['B1_mean'][k]:.1f},"
                         f"{r['B1_min'][k]},{r['B1_max'][k]},{r['B2_detect_prob'][k]:.4f},"
                         f"{r['B2_mean'][k]:.1f},{r['n95_B1'][k]}\n")
    with open("random_baseline_log.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
