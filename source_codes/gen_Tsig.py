#!/usr/bin/env python3
"""
=============================================================================
 Algorithm 5 : Generation of the error syndrome pattern test set  T_sig
=============================================================================

 One self-contained program.  Algorithms 1-4 are re-implemented here with the
 file-writing parts removed, so that they can be used purely as subroutines:

   Algorithm 1 : single-qubit syndrome tables  T_1^z  and  T_1^x
                 (built from the Qbit_and_Z_d*.txt / Qbit_and_X_d*.txt
                  stabilizer-incidence files)
   Algorithm 2 : significant consecutive multi-qubit error list, 2 <= s <= m
   Algorithm 3 : significant non-consecutive multi-qubit error list, 2 <= s <= m
   Algorithm 4 : significant consecutive (m+1)-qubit error list, s = m+1

   Algorithm 5 : for every error configuration E in
                     L = Significant_Error_List = Alg2 + Alg3 + Alg4
                 XOR together the single-qubit syndrome columns of the qubits
                 in E to obtain the Z-syndrome sigma^z and the X-syndrome
                 sigma^x of that configuration, and file the pair (sigma, E)
                 under its weight s = |E|.

                     T_sig^z = U_{s=2..m+1} T_s^z
                     T_sig^x = U_{s=2..m+1} T_s^x
                     T_sig   = T_sig^z  U  T_sig^x

 Conventions
 -----------
   * n = d^2 data qubits, labelled row-major q[r][c] = r*d + c
   * p = (d^2 - 1)/2 stabilizers of each Pauli type
   * m = floor((d-1)/2)
   * A p-bit syndrome word is stored as a Python integer; bit i (value 1 << i)
     is the syndrome bit of stabilizer i.  XOR of syndrome words is therefore
     just the integer '^' operator, which is exactly the GF(2) addition of
     the parity-check columns.
   * sigma^z is the Z-stabilizer syndrome of a bit-flip (X) error pattern.
     sigma^x is the X-stabilizer syndrome of a phase-flip (Z) error pattern.

 Reported for each distance d:
   1. Time to generate the complete set of error syndrome patterns
      (file writing excluded)
   2. Size of the complete test set
   3. Number of degenerate cases
   4. List of degenerate cases
   5. The syndrome patterns written to file

 Usage
 -----
     python3 gen_Tsig.py                     # inputs and outputs in ./
     python3 gen_Tsig.py --dir data --out results
     python3 gen_Tsig.py -d 5 7
=============================================================================
"""

import argparse
import os
import re
import sys
import time

NO_SECOND = -1                  # sentinel for a boundary qubit (single stabilizer)

DISTANCES = [5, 7, 9, 11, 13]   # default distances
TRAVERSALS = ['H', 'V', 'D']    # traversal modes for Algorithms 2-4
DELTA_SET = [2]                 # separation set Delta for Algorithm 3
DEDUPLICATE = True              # remove repeated error configurations from L
MAX_CLASSES_PRINTED = 25        # console cap; the file always holds the full list


# ==========================================================================
#  ALGORITHM 1 : single data-qubit error test set  T_1 = { T_1^z , T_1^x }
# ==========================================================================

def read_qubit_stabilizer_file(path):
    """Read a 'Qbit_and_Z_d*.txt' / 'Qbit_and_X_d*.txt' incidence file.

    Each line is   q<index> <stab> [<stab>]
    A qubit on a boundary touches only one stabilizer of that type; the
    missing second field is stored as -1, exactly as in the pseudocode.

    Returns (table, max_stab_index) with table[q] = (first, second).
    """
    entries = {}
    max_stab_index = -1

    with open(path, 'r') as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            tokens = line.split()
            if len(tokens) < 2 or len(tokens) > 3:
                raise ValueError("%s: line %d: expected 'q<i> <stab> [<stab>]', "
                                 "got '%s'" % (path, line_no, line))

            q = int(re.sub(r'^[A-Za-z]+', '', tokens[0]))
            stabs = [int(re.sub(r'^[A-Za-z]+', '', t)) for t in tokens[1:]]

            first = stabs[0]
            second = stabs[1] if len(stabs) == 2 else NO_SECOND

            if q in entries:
                raise ValueError("%s: data qubit q%d listed more than once"
                                 % (path, q))
            entries[q] = (first, second)
            max_stab_index = max(max_stab_index, first, second)

    if not entries:
        raise ValueError("%s: file contains no data" % path)

    n = len(entries)
    missing = [q for q in range(n) if q not in entries]
    if missing:
        raise ValueError("%s: missing data qubit indices %s" % (path, missing))

    return [entries[q] for q in range(n)], max_stab_index


def build_T1(qbit_and_z, qbit_and_x, n):
    """Algorithm 1.

    T_1^z[q] = Z-stabilizer syndrome of a single X error on data qubit q
    T_1^x[q] = X-stabilizer syndrome of a single Z error on data qubit q
    """
    T1_z = [0] * n                                     # line 1
    T1_x = [0] * n                                     # line 1

    for q in range(n):                                 # line 2
        S_z = 0                                        # line 3
        S_x = 0                                        # line 4

        zidx = qbit_and_z[q][0]                        # line 5
        S_z ^= (1 << zidx)                             # line 6
        if qbit_and_z[q][1] != NO_SECOND:              # line 7
            zidx = qbit_and_z[q][1]                    # line 8
            S_z ^= (1 << zidx)                         # line 9
        T1_z[q] = S_z                                  # line 10

        xidx = qbit_and_x[q][0]                        # line 11
        S_x ^= (1 << xidx)                             # line 12
        if qbit_and_x[q][1] != NO_SECOND:              # line 13
            xidx = qbit_and_x[q][1]                    # line 14
            S_x ^= (1 << xidx)                         # line 15
        T1_x[q] = S_x                                  # line 16

    return T1_z, T1_x                                  # lines 17-18


# ==========================================================================
#  ALGORITHM 2 : significant consecutive multiple data-qubit error list
# ==========================================================================

def consecutive_error_list(d, tau):
    """Algorithm 2 -- all consecutive patterns of weight 2 <= s <= m."""
    Consecutive_Error_List = []                 # line 1
    m = (d - 1) // 2                            # line 1

    for s in range(2, m + 1):                   # line 2

        if tau == 'H':                                           # line 3
            c_hat = 1                                            # line 4
            for r in range(0, d):                                # line 5
                if r == 0 or r == d - 1:                         # line 6
                    for c in range(0, d - s + 1):                # line 7
                        temp_list = []                           # line 8
                        for j in range(c, c + s):                # line 9
                            temp_list.append(r * d + j)
                        Consecutive_Error_List.append(temp_list)  # line 10
                else:                                            # line 11
                    if c_hat > d - s:                            # line 12
                        c_hat = 1
                    temp_list = []                               # line 13
                    for j in range(c_hat, c_hat + s):            # line 14
                        temp_list.append(r * d + j)
                    Consecutive_Error_List.append(temp_list)     # line 15
                    c_hat = c_hat + 1

        elif tau == 'V':                                         # line 16
            r_hat = 1                                            # line 17
            for c in range(0, d):                                # line 18
                if c == 0 or c == d - 1:                         # line 19
                    for r in range(0, d - s + 1):                # line 20
                        temp_list = []                           # line 21
                        for j in range(r, r + s):                # line 22
                            temp_list.append(j * d + c)
                        Consecutive_Error_List.append(temp_list)  # line 23
                else:                                            # line 24
                    if r_hat > d - s:                            # line 25
                        r_hat = 1
                    temp_list = []                               # line 26
                    for j in range(r_hat, r_hat + s):            # line 27
                        temp_list.append(j * d + c)
                    Consecutive_Error_List.append(temp_list)     # line 28
                    r_hat = r_hat + 1

        else:                                                    # line 29
            for r in range(0, d - s + 1):                        # line 30
                temp_list = []                                   # line 31
                for j in range(r, r + s):                        # line 32
                    temp_list.append(j * d + j)
                Consecutive_Error_List.append(temp_list)         # line 33
            for r in range(0, d - s + 1):                        # line 34
                c = d - 1 - r                                    # line 35
                temp_list = []
                for j in range(0, s):                            # line 36
                    temp_list.append((r + j) * d + (c - j))
                Consecutive_Error_List.append(temp_list)         # line 37

    return Consecutive_Error_List                                # line 38


# ==========================================================================
#  ALGORITHM 3 : significant non-consecutive multiple data-qubit error list
# ==========================================================================

def nc_error_list(d, tau, Delta):
    """Algorithm 3 -- patterns of weight 2 <= s <= m spaced by delta in Delta."""
    NC_Error_List = []                          # line 1
    m = (d - 1) // 2                            # line 1

    for s in range(2, m + 1):                   # line 2
        for delta in Delta:                     # line 3

            sigma = (s - 1) * delta + 1         # line 4  span of the pattern
            L = d - sigma                       # line 4  last valid start
            a_0 = min(1, L)                     # line 4  interior start point

            if L < 0:                           # line 5
                continue

            if tau == 'H':                                       # line 6
                c_hat = a_0                                      # line 7
                for r in range(0, d):                            # line 8
                    if r == 0 or r == d - 1:                     # line 9
                        for c in range(0, L + 1):                # line 10
                            temp_list = []                       # line 11
                            for k in range(0, s):                # line 12
                                temp_list.append(r * d + (c + k * delta))
                            NC_Error_List.append(temp_list)      # line 13
                    else:                                        # line 14
                        if c_hat > L:                            # line 15
                            c_hat = a_0
                        temp_list = []                           # line 16
                        for k in range(0, s):                    # line 17
                            temp_list.append(r * d + (c_hat + k * delta))
                        NC_Error_List.append(temp_list)          # line 18
                        c_hat = c_hat + 1

            elif tau == 'V':                                     # line 19
                r_hat = a_0                                      # line 20
                for c in range(0, d):                            # line 21
                    if c == 0 or c == d - 1:                     # line 22
                        for r in range(0, L + 1):                # line 23
                            temp_list = []                       # line 24
                            for k in range(0, s):                # line 25
                                temp_list.append((r + k * delta) * d + c)
                            NC_Error_List.append(temp_list)      # line 26
                    else:                                        # line 27
                        if r_hat > L:                            # line 28
                            r_hat = a_0
                        temp_list = []                           # line 29
                        for k in range(0, s):                    # line 30
                            temp_list.append((r_hat + k * delta) * d + c)
                        NC_Error_List.append(temp_list)          # line 31
                        r_hat = r_hat + 1

            else:                                                # line 32
                for a in range(0, L + 1):                        # line 33
                    temp_list = []                               # line 34
                    for k in range(0, s):                        # line 35
                        temp_list.append((a + k * delta) * d + (a + k * delta))
                    NC_Error_List.append(temp_list)              # line 36
                for a in range(0, L + 1):                        # line 37
                    temp_list = []                               # line 38
                    for k in range(0, s):                        # line 39
                        temp_list.append((a + k * delta) * d +
                                         (d - 1 - a - k * delta))
                    NC_Error_List.append(temp_list)              # line 40

    return NC_Error_List                                         # line 41


# ==========================================================================
#  ALGORITHM 4 : significant consecutive (m + 1) data-qubit error list
# ==========================================================================

def extended_error_list(d, tau):
    """Algorithm 4 -- consecutive patterns of the single weight s = m + 1."""
    Extended_Error_List = []                    # line 1
    m = (d - 1) // 2                            # line 1
    s = m + 1                                   # line 1

    if tau == 'H':                                           # line 2
        c_hat = 1                                            # line 3
        for r in range(0, d):                                # line 4
            if r == 0 or r == d - 1:                         # line 5
                for c in range(0, d - s + 1):                # line 6
                    temp_list = []                           # line 7
                    for j in range(c, c + s):                # line 8
                        temp_list.append(r * d + j)
                    Extended_Error_List.append(temp_list)    # line 9
            else:                                            # line 10
                if c_hat > d - s:                            # line 11
                    c_hat = 1
                temp_list = []                               # line 12
                for j in range(c_hat, c_hat + s):            # line 13
                    temp_list.append(r * d + j)
                Extended_Error_List.append(temp_list)        # line 14
                c_hat = c_hat + 1

    elif tau == 'V':                                         # line 15
        r_hat = 1                                            # line 16
        for c in range(0, d):                                # line 17
            if c == 0 or c == d - 1:                         # line 18
                for r in range(0, d - s + 1):                # line 19
                    temp_list = []                           # line 20
                    for j in range(r, r + s):                # line 21
                        temp_list.append(j * d + c)
                    Extended_Error_List.append(temp_list)    # line 22
            else:                                            # line 23
                if r_hat > d - s:                            # line 24
                    r_hat = 1
                temp_list = []                               # line 25
                for j in range(r_hat, r_hat + s):            # line 26
                    temp_list.append(j * d + c)
                Extended_Error_List.append(temp_list)        # line 27
                r_hat = r_hat + 1

    else:                                                    # line 28
        for r in range(0, d - s + 1):                        # line 29
            temp_list = []                                   # line 30
            for j in range(r, r + s):                        # line 31
                temp_list.append(j * d + j)
            Extended_Error_List.append(temp_list)            # line 32
        for r in range(0, d - s + 1):                        # line 33
            c = d - 1 - r                                    # line 34
            temp_list = []
            for j in range(0, s):                            # line 35
                temp_list.append((r + j) * d + (c - j))
            Extended_Error_List.append(temp_list)            # line 36

    return Extended_Error_List                               # line 37


def build_significant_error_list(d, Delta, deduplicate):
    """L = Significant_Error_List = Algorithm 2 + Algorithm 3 + Algorithm 4,
    taken over every traversal mode.  This is the input list of Algorithm 5."""
    L = []
    for tau in TRAVERSALS:
        L.extend(consecutive_error_list(d, tau))        # Algorithm 2
        L.extend(nc_error_list(d, tau, Delta))          # Algorithm 3
        L.extend(extended_error_list(d, tau))           # Algorithm 4

    if deduplicate:
        seen = set()
        unique = []
        for E in L:
            key = tuple(E)          # patterns are already in ascending order
            if key not in seen:
                seen.add(key)
                unique.append(E)
        L = unique
    return L


# ==========================================================================
#  ALGORITHM 5 : generation of the error syndrome pattern test set  T_sig
# ==========================================================================

def build_T_sig(T1_z, T1_x, L, d):
    """Algorithm 5.

    Returns (T_z, T_x) where T_z[s] and T_x[s] are lists of (sigma, E) pairs
    and sigma is a p-bit syndrome word packed into a Python integer.
    """
    m = (d - 1) // 2
    p = (d * d - 1) // 2                                        # line 1

    T_z = {}                                                    # line 2
    T_x = {}                                                    # line 2
    for s in range(2, m + 2):                                   # line 2
        T_z[s] = []
        T_x[s] = []

    k = len(L)                                                  # line 3

    for i in range(0, k):                                       # line 4
        E = L[i]                                                # line 5
        s = len(E)                                              # line 6
        sigma_z = 0                                             # line 7  0^p
        sigma_x = 0                                             # line 8  0^p

        for q in E:                                             # line 9
            sigma_z ^= T1_z[q]                                  # line 10
            sigma_x ^= T1_x[q]                                  # line 11

        T_z[s].append((sigma_z, E))                             # line 12
        T_x[s].append((sigma_x, E))                             # line 13

    return T_z, T_x, p                                          # lines 14-15


# ==========================================================================
#  Helpers : syndrome formatting, degeneracy analysis, memory
# ==========================================================================

def syndrome_to_bitstring(word, p):
    """s_0 s_1 ... s_{p-1}: stabilizer 0 leftmost (LSB-first packing)."""
    return ''.join('1' if (word >> i) & 1 else '0' for i in range(p))


def fired_stabilizers(word, p):
    """Indices of the stabilizers whose syndrome bit is set."""
    return [i for i in range(p) if (word >> i) & 1]


def flatten(T):
    """Flatten {s: [(sigma, E), ...]} into a single list of (sigma, E, s)."""
    out = []
    for s in sorted(T):
        for sigma, E in T[s]:
            out.append((sigma, E, s))
    return out


def analyse_degeneracy(T):
    """Locate the degenerate cases in one Pauli sector of T_sig.

    Two DISTINCT error configurations are degenerate when they produce the
    SAME syndrome: the decoder receives one word and cannot tell which
    configuration caused it.  The two supports then differ by an element of
    the stabilizer group (harmless) or by a logical operator (a real failure).

    The all-zero syndrome is reported separately: those configurations are
    themselves stabilizer or logical elements and are silent to the decoder,
    so they are a distinct phenomenon from a syndrome collision between two
    non-trivial patterns.
    """
    entries = flatten(T)

    buckets = {}
    for sigma, E, s in entries:
        buckets.setdefault(sigma, []).append((E, s))

    classes = []            # collisions between two or more distinct patterns
    zero_syndrome = []      # configurations whose syndrome is 0^p

    for sigma in sorted(buckets):
        members = buckets[sigma]
        if sigma == 0:
            zero_syndrome = members
            continue
        if len(members) > 1:
            classes.append((sigma, members))

    n_pairs = sum(len(mem) * (len(mem) - 1) // 2 for sigma, mem in classes)
    n_patterns = sum(len(mem) for sigma, mem in classes)

    return {
        'total': len(entries),
        'distinct': len(buckets),
        'classes': classes,
        'n_classes': len(classes),
        'n_pairs': n_pairs,
        'n_patterns': n_patterns,
        'zero_syndrome': zero_syndrome,
        'n_zero': len(zero_syndrome),
    }


def deep_size(obj, seen=None):
    """Resident memory of a nested structure; shared objects counted once."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))

    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, val in obj.items():
            size += deep_size(key, seen) + deep_size(val, seen)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            size += deep_size(item, seen)
    return size


def human_bytes(n):
    value = float(n)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if value < 1024.0 or unit == 'GB':
            return '%.2f %s' % (value, unit)
        value = value / 1024.0


# ==========================================================================
#  File output
# ==========================================================================

def write_syndrome_patterns(path, T, p, d, sector, degen):
    """Write one Pauli sector of T_sig.

    One line per syndrome pattern:

        s=<weight> | E: <q ...> | S: <p-bit string> | fired: <stab ...>

    The bit string is LSB-first, i.e. its leftmost character is stabilizer 0.
    """
    stab = 'z' if sector == 'z' else 'x'
    err = 'X' if sector == 'z' else 'Z'
    m = (d - 1) // 2

    with open(path, 'w') as f:
        f.write("# Error syndrome pattern test set  T_sig^%s   (distance d = %d)\n"
                % (stab, d))
        f.write("# n = %d data qubits, p = %d %s-stabilizers, m = %d\n"
                % (d * d, p, stab.upper(), m))
        f.write("# Each row: the %s-stabilizer syndrome produced by an %s-error\n"
                % (stab.upper(), err))
        f.write("# on the listed set E of data qubits.\n")
        f.write("# Bit string is LSB-first: leftmost char = %s0, "
                "rightmost = %s%d\n" % (stab, stab, p - 1))
        f.write("# Format: s=<weight> | E: <qubits> | S: <bits> | fired: <stabilizers>\n")
        f.write("#\n")

        for s in sorted(T):
            f.write("# ---- weight s = %d : %d syndrome patterns ----\n"
                    % (s, len(T[s])))
            for sigma, E in T[s]:
                bits = syndrome_to_bitstring(sigma, p)
                fired = ' '.join('%s%d' % (stab, i)
                                 for i in fired_stabilizers(sigma, p))
                f.write("s=%d | E: %s | S: %s | fired: %s\n"
                        % (s, ' '.join(str(q) for q in E), bits, fired))

        f.write("#\n# Degenerate cases in T_sig^%s "
                "(distinct supports sharing one syndrome): %d class(es), "
                "%d pattern(s), %d pair(s)\n"
                % (stab, degen['n_classes'], degen['n_patterns'],
                   degen['n_pairs']))
        if degen['classes']:
            for sigma, members in degen['classes']:
                fired = ' '.join('%s%d' % (stab, i)
                                 for i in fired_stabilizers(sigma, p))
                f.write("#   syndrome {%s}\n" % fired)
                for E, s in members:
                    f.write("#       s=%d  E = {%s}\n"
                            % (s, ', '.join(str(q) for q in E)))
        else:
            f.write("#   none\n")

        f.write("#\n# Zero-syndrome configurations (silent to the decoder): %d\n"
                % degen['n_zero'])
        for E, s in degen['zero_syndrome']:
            f.write("#       s=%d  E = {%s}\n"
                    % (s, ', '.join(str(q) for q in E)))


# ==========================================================================
#  Driver for one distance
# ==========================================================================

def run_one_distance(d, z_path, x_path, out_dir, repeats=5):
    n = d * d
    p = (n - 1) // 2
    m = (d - 1) // 2

    # ---- input files (parsing is I/O, kept outside the timed region) ------
    qz_table, max_z = read_qubit_stabilizer_file(z_path)
    qx_table, max_x = read_qubit_stabilizer_file(x_path)
    if len(qz_table) != n or len(qx_table) != n:
        raise ValueError("d=%d: expected n = %d data qubits, found %d (Z) / %d (X)"
                         % (d, n, len(qz_table), len(qx_table)))
    if max_z >= p or max_x >= p:
        raise ValueError("d=%d: stabilizer index out of range "
                         "(max z%d, x%d, but p = %d)" % (d, max_z, max_x, p))

    # ================= 1. TIMED GENERATION ================================
    #
    # The whole pipeline takes well under a millisecond, which is the same
    # order as the clock granularity and OS scheduling jitter.  Timing a
    # single run makes the numbers non-monotone in d.  So: run once untimed
    # to warm up the interpreter, then keep the FASTEST of `repeats` runs.
    #
    def timed_pipeline():
        t0 = time.perf_counter()
        T1_z, T1_x = build_T1(qz_table, qx_table, n)                 # Alg 1
        t1 = time.perf_counter()
        L = build_significant_error_list(d, DELTA_SET, DEDUPLICATE)  # Alg 2,3,4
        t2 = time.perf_counter()
        T_z, T_x, p_local = build_T_sig(T1_z, T1_x, L, d)            # Alg 5
        t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2, t3 - t0), (L, T_z, T_x, p_local)

    timed_pipeline()                                   # warm-up, not measured

    best = None
    for _ in range(max(1, repeats)):
        times, result = timed_pipeline()
        if best is None or times[3] < best[3]:
            best = times
            keep = result

    time_alg1, time_alg234, time_alg5, time_total = best
    L, T_z, T_x, p = keep
    # ======================================================================

    # ---- 2. size of the complete test set --------------------------------
    size_z = sum(len(T_z[s]) for s in T_z)
    size_x = sum(len(T_x[s]) for s in T_x)
    size_total = size_z + size_x

    # ---- 3./4. degenerate cases ------------------------------------------
    degen_z = analyse_degeneracy(T_z)
    degen_x = analyse_degeneracy(T_x)

    mem_bytes = deep_size(T_z) + deep_size(T_x)

    # ---- report -----------------------------------------------------------
    print("=" * 78)
    print("  DISTANCE d = %-3d   n = d^2 = %-4d data qubits   "
          "p = (d^2-1)/2 = %-3d   m = %d" % (d, n, p, m))
    print("=" * 78)
    print("1. Generation time (file writing excluded)")
    print("     Algorithm 1   (T_1^z, T_1^x)        : %12.6f ms"
          % (time_alg1 * 1e3))
    print("     Algorithms 2-4 (error list L)       : %12.6f ms"
          % (time_alg234 * 1e3))
    print("     Algorithm 5   (syndrome patterns)   : %12.6f ms"
          % (time_alg5 * 1e3))
    print("     ---------------------------------------------------")
    print("     TOTAL, end to end                   : %12.6f ms  (%.6f s)"
          % (time_total * 1e3, time_total))
    print("     [fastest of %d runs, after warm-up]" % max(1, repeats))

    print("\n2. Size of the complete test set")
    print("     |L|  significant error configs       : %8d" % len(L))
    print("     |T_sig^z|  (X errors -> Z syndromes) : %8d" % size_z)
    print("     |T_sig^x|  (Z errors -> X syndromes) : %8d" % size_x)
    print("     |T_sig| = |T_sig^z| + |T_sig^x|      : %8d syndrome patterns"
          % size_total)
    print("     per weight s:")
    print("        %-5s %12s %12s %12s" % ("s", "|T_s^z|", "|T_s^x|", "|T_s|"))
    for s in sorted(T_z):
        tag = '   <- s = m+1' if s == m + 1 else ''
        print("        %-5d %12d %12d %12d%s"
              % (s, len(T_z[s]), len(T_x[s]), len(T_z[s]) + len(T_x[s]), tag))
    print("     memory of stored test set            : %s (%d bytes)"
          % (human_bytes(mem_bytes), mem_bytes))

    for label, degen, stab in (("T_sig^z  (X errors -> Z syndromes)", degen_z, 'z'),
                               ("T_sig^x  (Z errors -> X syndromes)", degen_x, 'x')):
        print("\n3. Degenerate cases in %s" % label)
        print("     distinct syndrome words              : %8d of %d"
              % (degen['distinct'], degen['total']))
        print("     degenerate classes                   : %8d"
              % degen['n_classes'])
        print("     error configurations involved        : %8d"
              % degen['n_patterns'])
        print("     unordered degenerate pairs           : %8d"
              % degen['n_pairs'])
        print("     zero-syndrome (silent) configs       : %8d"
              % degen['n_zero'])

        print("\n4. List of degenerate cases in %s" % label)
        if not degen['classes']:
            print("     none")
        else:
            shown = degen['classes'][:MAX_CLASSES_PRINTED]
            for idx, (sigma, members) in enumerate(shown, start=1):
                fired = ' '.join('%s%d' % (stab, i)
                                 for i in fired_stabilizers(sigma, p))
                print("     [%d] syndrome {%s}" % (idx, fired))
                for E, s in members:
                    print("           s=%d  E = {%s}"
                          % (s, ', '.join(str(q) for q in E)))
            if len(degen['classes']) > len(shown):
                print("     ... %d further class(es); the complete list is in "
                      "the output file"
                      % (len(degen['classes']) - len(shown)))

        if degen['zero_syndrome']:
            print("     zero-syndrome configurations (decoder sees nothing):")
            for E, s in degen['zero_syndrome'][:MAX_CLASSES_PRINTED]:
                print("           s=%d  E = {%s}"
                      % (s, ', '.join(str(q) for q in E)))
            if degen['n_zero'] > MAX_CLASSES_PRINTED:
                print("     ... %d further configuration(s)"
                      % (degen['n_zero'] - MAX_CLASSES_PRINTED))

    # ---- 5. file output (NOT part of the timed region) --------------------
    os.makedirs(out_dir, exist_ok=True)
    z_out = os.path.join(out_dir, "Tsig_Zsyndrome_d%d.txt" % d)
    x_out = os.path.join(out_dir, "Tsig_Xsyndrome_d%d.txt" % d)
    write_syndrome_patterns(z_out, T_z, p, d, 'z', degen_z)
    write_syndrome_patterns(x_out, T_x, p, d, 'x', degen_x)

    print("\n5. Syndrome patterns written to")
    print("     %s   (%s)" % (z_out, human_bytes(os.path.getsize(z_out))))
    print("     %s   (%s)" % (x_out, human_bytes(os.path.getsize(x_out))))
    print()

    return {
        'd': d, 'n': n, 'p': p,
        'time_total': time_total, 'time_alg5': time_alg5,
        'size': size_total, 'mem': mem_bytes,
        'degen_z': degen_z, 'degen_x': degen_x,
    }


# ==========================================================================
#  Command line
# ==========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Algorithm 5: build the error syndrome pattern test set T_sig.")
    ap.add_argument('--dir', default='.', help="directory holding the input files")
    ap.add_argument('--out', default='.', help="directory for the output files")
    ap.add_argument('-d', '--distances', type=int, nargs='+', default=DISTANCES,
                    help="distances to process (default: 5 7 9 11 13)")
    ap.add_argument('--repeats', type=int, default=5,
                    help="repeat the generation and report the fastest run")
    args = ap.parse_args()

    print("=" * 78)
    print("  Error syndrome pattern test set T_sig  (Algorithms 1-5)")
    print("  Separation set Delta = %s     Traversals = %s     Dedup = %s"
          % (DELTA_SET, TRAVERSALS, DEDUPLICATE))
    print("=" * 78)
    print()

    summary = []
    for d in args.distances:
        z_path = os.path.join(args.dir, "Qbit_and_Z_d%d.txt" % d)
        x_path = os.path.join(args.dir, "Qbit_and_X_d%d.txt" % d)
        for pth in (z_path, x_path):
            if not os.path.exists(pth):
                ap.error("missing input file: %s" % pth)
        summary.append(run_one_distance(d, z_path, x_path, args.out,
                                        args.repeats))

    print("=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(" %-4s %-8s %-6s %-14s %-14s %-10s %-10s %-10s"
          % ("d", "n", "p", "gen time (ms)", "|T_sig|", "memory",
             "degen z", "degen x"))
    print("-" * 78)
    for s in summary:
        print(" %-4d %-8d %-6d %-14.4f %-14d %-10s %-10d %-10d"
              % (s['d'], s['n'], s['p'], s['time_total'] * 1e3, s['size'],
                 human_bytes(s['mem']), s['degen_z']['n_classes'],
                 s['degen_x']['n_classes']))
    print("=" * 78)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as err:
        print("error: %s" % err, file=sys.stderr)
        sys.exit(1)
