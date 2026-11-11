#!/usr/bin/env python3
"""
Algorithm 1 : Single data qubit error test set T_1
==================================================

For a distance-d rotated surface code with n = d^2 data qubits and
p = (d^2 - 1)/2 stabilizers of each type, this program builds

    T_1^z = { S_z^(0), S_z^(1), ..., S_z^(n-1) }   syndrome of the Z-stabilizers
                                                   for a single X-error on qubit q
    T_1^x = { S_x^(0), S_x^(1), ..., S_x^(n-1) }   syndrome of the X-stabilizers
                                                   for a single Z-error on qubit q

    T_1   = { T_1^z , T_1^x }

Input files (one line per data qubit):

    q<index> <stab> [<stab>]
    e.g.   q0 z8
           q1 z0 z8

A qubit sitting on a boundary touches only ONE stabilizer of that type; the
missing "second" field is treated as -1, exactly as in the pseudocode.

Reported for each distance:
    a) time to generate T_1
    b) memory size of the stored test set
    c) two output files (Z-error test set and X-error test set)
    d) test set size for X and for Z errors
    e) number of degenerate cases
    f) the data qubit indices that give rise to those degenerate cases

Usage
-----
    python3 gen_T1.py                          # auto-detect every d in ./
    python3 gen_T1.py --dir data --out results # look in ./data, write to ./results
    python3 gen_T1.py -d 5 7                   # only these distances
    python3 gen_T1.py --z Qbit_and_Z_d5.txt --x Qbit_and_X_d5.txt -d 5
"""

import argparse
import os
import re
import sys
import time

NO_SECOND = -1          # sentinel used by the pseudocode for a boundary qubit


# --------------------------------------------------------------------------
# 1.  Reading an input file
# --------------------------------------------------------------------------
def read_qubit_stabilizer_file(path):
    """
    Read a 'Qbit_and_Z_d*.txt' / 'Qbit_and_X_d*.txt' file.

    Returns a list `table` where table[q] = (first, second) and
    second == -1 when the qubit touches only one stabilizer.
    """
    entries = {}                       # q -> (first, second)
    max_stab_index = -1

    with open(path, "r") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue                              # skip blanks / comments

            tokens = line.split()
            if len(tokens) < 2 or len(tokens) > 3:
                raise ValueError(
                    f"{path}: line {line_no}: expected 'q<i> <stab> [<stab>]', got '{line}'")

            # ---- qubit index : strip the leading letter(s) of 'q12' ----------
            q = int(re.sub(r"^[A-Za-z]+", "", tokens[0]))

            # ---- stabilizer indices : strip the leading letter of 'z8'/'x3' --
            stabs = [int(re.sub(r"^[A-Za-z]+", "", t)) for t in tokens[1:]]

            first = stabs[0]
            second = stabs[1] if len(stabs) == 2 else NO_SECOND

            if q in entries:
                raise ValueError(f"{path}: data qubit q{q} listed more than once")
            entries[q] = (first, second)
            max_stab_index = max(max_stab_index, first, second)

    if not entries:
        raise ValueError(f"{path}: file contains no data")

    n = len(entries)
    missing = [q for q in range(n) if q not in entries]
    if missing:
        raise ValueError(f"{path}: missing data qubit indices {missing}")

    table = [entries[q] for q in range(n)]
    return table, max_stab_index


def check_dimensions(qz_table, qx_table, d):
    """Sanity check n = d^2 and p = (d^2 - 1)/2 against the parsed files."""
    n = d * d
    p = (d * d - 1) // 2
    if len(qz_table) != n or len(qx_table) != n:
        raise ValueError(f"d={d}: expected n = {n} data qubits, "
                         f"found {len(qz_table)} (Z file) and {len(qx_table)} (X file)")
    return n, p


# --------------------------------------------------------------------------
# 2.  Algorithm 1 itself
# --------------------------------------------------------------------------
def build_T1(qbit_and_z, qbit_and_x, n, p):
    """
    Direct transcription of Algorithm 1.

    A p-bit syndrome word is stored as a Python integer:
    bit i (value 1 << i) is the syndrome bit of stabilizer i,
    i.e. an LSB-first packing  s_0 s_1 ... s_{p-1}.

    Returns (T1_z, T1_x), each a list of n integers.
    """
    T1_z = [0] * n                                     # T_1^z[q]
    T1_x = [0] * n                                     # T_1^x[q]

    for q in range(n):                                 # line 2
        S_z = 0                                        # line 3 : clear Z-syndrome word
        S_x = 0                                        # line 4 : clear X-syndrome word

        # ---------------- Z-syndrome of an X-error on qubit q ----------------
        zidx = qbit_and_z[q][0]                        # line 5
        S_z ^= (1 << zidx)                             # line 6 : flip that bit
        if qbit_and_z[q][1] != NO_SECOND:              # line 7
            zidx = qbit_and_z[q][1]                    # line 8
            S_z ^= (1 << zidx)                         # line 9
        T1_z[q] = S_z                                  # line 10

        # ---------------- X-syndrome of a Z-error on qubit q ----------------
        xidx = qbit_and_x[q][0]                        # line 11
        S_x ^= (1 << xidx)                             # line 12
        if qbit_and_x[q][1] != NO_SECOND:              # line 13
            xidx = qbit_and_x[q][1]                    # line 14
            S_x ^= (1 << xidx)                         # line 15
        T1_x[q] = S_x                                  # line 16

    return T1_z, T1_x                                  # lines 17-18


# --------------------------------------------------------------------------
# 3.  Helpers : printing, memory, degeneracy
# --------------------------------------------------------------------------
def syndrome_to_bitstring(word, p):
    """Return s_0 s_1 ... s_{p-1}  (stabilizer 0 on the LEFT, LSB-first packing)."""
    return "".join("1" if (word >> i) & 1 else "0" for i in range(p))


def fired_stabilizers(word, p):
    """Indices of the stabilizers whose syndrome bit is 1."""
    return [i for i in range(p) if (word >> i) & 1]


def measure_memory(T1_z, T1_x, n, p):
    """
    Memory occupied by the stored test set.

    'python_bytes'  : what the Python objects actually cost (list + int objects)
    'packed_bytes'  : the information-theoretic cost of 2*n words of p bits each,
                      i.e. what the same table costs in packed form (RTL / C).
    """
    python_bytes = 0
    for lst in (T1_z, T1_x):
        python_bytes += sys.getsizeof(lst)
        python_bytes += sum(sys.getsizeof(w) for w in lst)

    packed_bits = 2 * n * p
    packed_bytes = (packed_bits + 7) // 8
    return python_bytes, packed_bytes, packed_bits


def analyse_degeneracy(T1, n):
    """
    Two data qubits are *degenerate* when a single error on either of them
    produces exactly the same syndrome; the decoder cannot tell them apart
    (the two errors differ by a stabilizer of the opposite type).

    Returns a dict with
        classes        : list of qubit-index lists, one per colliding syndrome
        n_classes      : how many syndromes are shared by more than one qubit
        n_qubits       : how many data qubits take part in a collision
        n_pairs        : number of unordered degenerate qubit pairs
        distinct       : number of distinct syndrome patterns in the test set
        qubit_indices  : sorted flat list of the participating qubit indices
    """
    buckets = {}
    for q in range(n):
        buckets.setdefault(T1[q], []).append(q)

    classes = [sorted(qs) for w, qs in sorted(buckets.items()) if len(qs) > 1]
    qubit_indices = sorted(q for cls in classes for q in cls)
    n_pairs = sum(len(cls) * (len(cls) - 1) // 2 for cls in classes)

    return {
        "classes": classes,
        "n_classes": len(classes),
        "n_qubits": len(qubit_indices),
        "n_pairs": n_pairs,
        "distinct": len(buckets),
        "qubit_indices": qubit_indices,
    }


def write_test_set(path, T1, n, p, d, err_type, stab_type, degen):
    """Write one half of the test set to a text file."""
    with open(path, "w") as f:
        f.write(f"# Single data qubit error test set  T_1^{stab_type}   (distance d = {d})\n")
        f.write(f"# n = {n} data qubits, p = {p} {stab_type.upper()}-stabilizers\n")
        f.write(f"# Row q : syndrome of the {stab_type.upper()}-stabilizers produced by a "
                f"single {err_type}-error on data qubit q\n")
        f.write(f"# Bit string is LSB-first: leftmost character is stabilizer "
                f"{stab_type}0, rightmost is {stab_type}{p-1}\n")
        f.write("#\n")
        f.write(f"# {'qubit':>6}  {'syndrome bits':<{max(p,13)}}  fired stabilizers\n")
        for q in range(n):
            bits = syndrome_to_bitstring(T1[q], p)
            fired = " ".join(f"{stab_type}{i}" for i in fired_stabilizers(T1[q], p))
            f.write(f"  q{q:<5d}  {bits:<{max(p,13)}}  {fired}\n")

        f.write("#\n# Degenerate classes (qubits sharing an identical syndrome):\n")
        if degen["classes"]:
            for cls in degen["classes"]:
                bits = syndrome_to_bitstring(T1[cls[0]], p)
                f.write("#   {" + ", ".join(f"q{q}" for q in cls) + "}  ->  " + bits + "\n")
        else:
            f.write("#   none\n")


# --------------------------------------------------------------------------
# 4.  Driver for one distance
# --------------------------------------------------------------------------
def run_one_distance(d, z_path, x_path, out_dir, repeats=1):
    qz_table, max_z = read_qubit_stabilizer_file(z_path)
    qx_table, max_x = read_qubit_stabilizer_file(x_path)
    n, p = check_dimensions(qz_table, qx_table, d)

    if max_z >= p or max_x >= p:
        raise ValueError(f"d={d}: stabilizer index out of range "
                         f"(max z{max_z}, x{max_x}, but p = {p})")

    # ---------------- a) timing --------------------------------------------
    #
    # A single call to build_T1 takes only tens of microseconds, which is the
    # same order as the clock granularity and the OS scheduling jitter.  Timing
    # it directly makes the FIRST distance measured look artificially slow,
    # because it also pays the one-off interpreter warm-up (bytecode
    # specialisation, allocator free lists, cold cache).
    #
    # So: (1) run once untimed to warm up, then (2) time a BATCH of `inner`
    # repetitions and divide, so the measured interval is milliseconds rather
    # than microseconds, and (3) keep the fastest batch out of `repeats`.
    #
    build_T1(qz_table, qx_table, n, p)                 # warm-up, not timed

    # choose the batch size so that one batch lasts at least ~50 ms
    t0 = time.perf_counter()
    build_T1(qz_table, qx_table, n, p)
    single = max(time.perf_counter() - t0, 1e-9)
    inner = max(1, min(100000, int(0.05 / single)))

    best = float("inf")
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        for _ in range(inner):
            T1_z, T1_x = build_T1(qz_table, qx_table, n, p)
        best = min(best, (time.perf_counter() - t0) / inner)
    elapsed = best

    # ---------------- b) memory --------------------------------------------
    py_bytes, packed_bytes, packed_bits = measure_memory(T1_z, T1_x, n, p)

    # ---------------- e/f) degeneracy --------------------------------------
    degen_z = analyse_degeneracy(T1_z, n)
    degen_x = analyse_degeneracy(T1_x, n)

    # ---------------- c) output files --------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    z_out = os.path.join(out_dir, f"T1_Zsyndrome_d{d}.txt")
    x_out = os.path.join(out_dir, f"T1_Xsyndrome_d{d}.txt")
    write_test_set(z_out, T1_z, n, p, d, err_type="X", stab_type="z", degen=degen_z)
    write_test_set(x_out, T1_x, n, p, d, err_type="Z", stab_type="x", degen=degen_x)

    # ---------------- report -----------------------------------------------
    print("=" * 78)
    print(f"  DISTANCE d = {d}     n = d^2 = {n} data qubits     "
          f"p = (d^2-1)/2 = {p} stabilizers per type")
    print("=" * 78)
    print(f"a) Generation time for T_1            : {elapsed*1e6:12.3f} us "
          f"({elapsed:.9f} s)")
    print(f"   per data qubit                     : {elapsed*1e9/n:12.1f} ns/qubit"
          f"   [best of {max(1, repeats)} x {inner} runs, after warm-up]")
    print(f"b) Memory to store T_1 (Python objs)  : {py_bytes:12d} bytes "
          f"({py_bytes/1024:.2f} KiB)")
    print(f"   Memory to store T_1 (packed bits)  : {packed_bytes:12d} bytes "
          f"({packed_bits} bits = 2 x {n} x {p})")
    print(f"c) Z-error test set written to        : {z_out}")
    print(f"   X-error test set written to        : {x_out}")
    print(f"d) Test set size, X errors (|T_1^z|)  : {n:12d} syndrome patterns")
    print(f"   Test set size, Z errors (|T_1^x|)  : {n:12d} syndrome patterns")
    print(f"   Total |T_1|                        : {2*n:12d} syndrome patterns")

    for label, degen, T1, stab in (
            ("X errors -> Z-syndromes", degen_z, T1_z, "z"),
            ("Z errors -> X-syndromes", degen_x, T1_x, "x")):
        print(f"\n   --- {label} ---")
        print(f"   distinct syndrome patterns         : {degen['distinct']}")
        print(f"e) degenerate classes                 : {degen['n_classes']}")
        print(f"   degenerate qubit pairs             : {degen['n_pairs']}")
        print(f"   data qubits involved               : {degen['n_qubits']}")
        print(f"f) degenerate qubit indices           : {degen['qubit_indices']}")
        if degen["classes"]:
            print("   degenerate classes in detail       :")
            for cls in degen["classes"]:
                fired = " ".join(f"{stab}{i}" for i in fired_stabilizers(T1[cls[0]], p))
                print("       {" + ", ".join(f"q{q}" for q in cls) + "}"
                      + f"  ->  syndrome {{{fired}}}")
    print()

    return {
        "d": d, "n": n, "p": p, "time_s": elapsed,
        "python_bytes": py_bytes, "packed_bytes": packed_bytes,
        "degen_z": degen_z, "degen_x": degen_x,
    }


# --------------------------------------------------------------------------
# 5.  Command line
# --------------------------------------------------------------------------
def discover_distances(directory):
    """Find every d for which BOTH Qbit_and_Z_d*.txt and Qbit_and_X_d*.txt exist."""
    found = []
    for name in os.listdir(directory):
        m = re.fullmatch(r"Qbit_and_Z_d(\d+)\.txt", name)
        if m:
            d = int(m.group(1))
            if os.path.exists(os.path.join(directory, f"Qbit_and_X_d{d}.txt")):
                found.append(d)
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(
        description="Algorithm 1: build the single data qubit error test set T_1.")
    ap.add_argument("--dir", default=".", help="directory holding the input files")
    ap.add_argument("--out", default=".", help="directory for the output files")
    ap.add_argument("-d", "--distances", type=int, nargs="+",
                    help="distances to process (default: every pair found in --dir)")
    ap.add_argument("--z", help="explicit path to the Qbit_and_Z file (needs -d)")
    ap.add_argument("--x", help="explicit path to the Qbit_and_X file (needs -d)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat the generation and report the best time")
    args = ap.parse_args()

    # explicit single pair
    if args.z or args.x:
        if not (args.z and args.x and args.distances and len(args.distances) == 1):
            ap.error("--z and --x must be given together with exactly one -d value")
        run_one_distance(args.distances[0], args.z, args.x, args.out, args.repeats)
        return

    distances = args.distances or discover_distances(args.dir)
    if not distances:
        ap.error(f"no Qbit_and_Z_d*.txt / Qbit_and_X_d*.txt pair found in '{args.dir}'")

    summary = []
    for d in distances:
        z_path = os.path.join(args.dir, f"Qbit_and_Z_d{d}.txt")
        x_path = os.path.join(args.dir, f"Qbit_and_X_d{d}.txt")
        for pth in (z_path, x_path):
            if not os.path.exists(pth):
                ap.error(f"missing input file: {pth}")
        summary.append(run_one_distance(d, z_path, x_path, args.out, args.repeats))

    if len(summary) > 1:
        print("=" * 78)
        print("  SUMMARY")
        print("=" * 78)
        print(f"{'d':>4} {'n':>7} {'p':>7} {'|T_1|':>8} {'time (us)':>12} "
              f"{'packed (B)':>12} {'degen Z':>9} {'degen X':>9}")
        for s in summary:
            print(f"{s['d']:>4} {s['n']:>7} {s['p']:>7} {2*s['n']:>8} "
                  f"{s['time_s']*1e6:>12.3f} {s['packed_bytes']:>12} "
                  f"{s['degen_z']['n_classes']:>9} {s['degen_x']['n_classes']:>9}")
        print()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(1)
