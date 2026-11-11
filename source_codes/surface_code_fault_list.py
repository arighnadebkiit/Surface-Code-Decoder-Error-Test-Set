"""
=============================================================================
 Automatic generation of the significant data-qubit error (fault) list
 for a distance-d rotated surface code.
=============================================================================

 Implements

   Algorithm 2 : Significant consecutive multiple data-qubit error list
                 (all sizes 2 <= s <= m)
   Algorithm 3 : Significant NON-consecutive multiple data-qubit error list
                 (all sizes 2 <= s <= m, separation set Delta)
   Algorithm 4 : Significant consecutive (m + 1) data-qubit error list
                 (single size s = m + 1)

 Conventions
 -----------
   * Data qubits are labelled row-major:  q[r][c] = r*d + c,  0 <= r,c <= d-1
   * m = floor((d - 1) / 2)                      (correctable-weight bound)
   * Traversal modes tau in {'H', 'V', 'D'}:
        'H' -> horizontal (along a row)
        'V' -> vertical   (along a column)
        'D' -> both diagonals (principal and anti-diagonal)

 The "complete fault list" is the union of the outputs of Algorithms 2, 3
 and 4 taken over every traversal mode tau in {'H', 'V', 'D'}.

 Reported for each distance d:
   1. Time to generate the complete fault list (file writing excluded)
   2. Size (number of error patterns) of the complete fault list
   3. Number of s-data-qubit error sets for each s
   4. Memory footprint of storing the complete fault list
   5. The list written to a text file, one pattern per line, indices
      separated by a single space.
=============================================================================
"""

import sys
import time
import os
import gc

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DISTANCES = [5, 7, 9, 11, 13]   # code distances to run
TRAVERSALS = ['H', 'V', 'D']    # traversal modes
DELTA_SET = [2]                 # separation set Delta used by Algorithm 3
DEDUPLICATE = True              # drop repeated patterns from the final list
TIMING_REPEATS = 5              # report the fastest of this many runs
OUTPUT_DIR = '.'                # where the fault-list files are written


# ==========================================================================
#  ALGORITHM 2 : Significant consecutive multiple data-qubit error list
# ==========================================================================

def consecutive_error_list(d, tau):
    """Algorithm 2.

    Returns a list of lists.  Each inner list holds s consecutive erroneous
    data-qubit indices along the traversal tau, for every 2 <= s <= m.
    """
    Consecutive_Error_List = []                 # line 1
    m = (d - 1) // 2                            # line 1

    for s in range(2, m + 1):                   # line 2   (s = 2 .. m)

        # ---------------- horizontal traversal ----------------  line 3
        if tau == 'H':
            c_hat = 1                                            # line 4
            for r in range(0, d):                                # line 5
                if r == 0 or r == d - 1:                         # line 6
                    # boundary rows: sweep every window in the row
                    for c in range(0, d - s + 1):                # line 7
                        temp_list = []                           # line 8
                        for j in range(c, c + s):                # line 9
                            temp_list.append(r * d + j)
                        Consecutive_Error_List.append(temp_list)  # line 10
                else:                                            # line 11
                    # interior rows: one sliding-window sample per row
                    if c_hat > d - s:                            # line 12
                        c_hat = 1
                    temp_list = []                               # line 13
                    for j in range(c_hat, c_hat + s):            # line 14
                        temp_list.append(r * d + j)
                    Consecutive_Error_List.append(temp_list)     # line 15
                    c_hat = c_hat + 1

        # ---------------- vertical traversal ------------------  line 16
        elif tau == 'V':
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

        # ---------------- diagonal traversal ------------------  line 29
        else:
            # principal diagonal
            for r in range(0, d - s + 1):                        # line 30
                temp_list = []                                   # line 31
                for j in range(r, r + s):                        # line 32
                    temp_list.append(j * d + j)
                Consecutive_Error_List.append(temp_list)         # line 33
            # anti-diagonal
            for r in range(0, d - s + 1):                        # line 34
                c = d - 1 - r                                    # line 35
                temp_list = []
                for j in range(0, s):                            # line 36
                    temp_list.append((r + j) * d + (c - j))
                Consecutive_Error_List.append(temp_list)         # line 37

    return Consecutive_Error_List                                # line 38


# ==========================================================================
#  ALGORITHM 3 : Significant non-consecutive multiple data-qubit error list
# ==========================================================================

def nc_error_list(d, tau, Delta):
    """Algorithm 3.

    Returns a list of lists.  Each inner list holds s erroneous data-qubit
    indices spaced by delta along the traversal tau, for every 2 <= s <= m
    and every delta in the separation set Delta.
    """
    NC_Error_List = []                          # line 1
    m = (d - 1) // 2                            # line 1

    for s in range(2, m + 1):                   # line 2
        for delta in Delta:                     # line 3

            sigma = (s - 1) * delta + 1         # line 4  span of the pattern
            L = d - sigma                       # line 4  last valid start
            a_0 = min(1, L)                     # line 4  interior start point

            if L < 0:                           # line 5  pattern cannot fit
                continue

            # ------------- horizontal traversal ---------------  line 6
            if tau == 'H':
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

            # ------------- vertical traversal -----------------  line 19
            elif tau == 'V':
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

            # ------------- diagonal traversal -----------------  line 32
            else:
                # principal diagonal
                for a in range(0, L + 1):                        # line 33
                    temp_list = []                               # line 34
                    for k in range(0, s):                        # line 35
                        temp_list.append((a + k * delta) * d + (a + k * delta))
                    NC_Error_List.append(temp_list)              # line 36
                # anti-diagonal
                for a in range(0, L + 1):                        # line 37
                    temp_list = []                               # line 38
                    for k in range(0, s):                        # line 39
                        temp_list.append((a + k * delta) * d +
                                         (d - 1 - a - k * delta))
                    NC_Error_List.append(temp_list)              # line 40

    return NC_Error_List                                         # line 41


# ==========================================================================
#  ALGORITHM 4 : Significant consecutive (m + 1) data-qubit error list
# ==========================================================================

def extended_error_list(d, tau):
    """Algorithm 4.

    Same window construction as Algorithm 2, but for the single weight
    s = m + 1, i.e. one qubit beyond the guaranteed correctable bound.
    """
    Extended_Error_List = []                    # line 1
    m = (d - 1) // 2                            # line 1
    s = m + 1                                   # line 1

    # ---------------- horizontal traversal ----------------  line 2
    if tau == 'H':
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

    # ---------------- vertical traversal ------------------  line 15
    elif tau == 'V':
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

    # ---------------- diagonal traversal ------------------  line 28
    else:
        # principal diagonal
        for r in range(0, d - s + 1):                        # line 29
            temp_list = []                                   # line 30
            for j in range(r, r + s):                        # line 31
                temp_list.append(j * d + j)
            Extended_Error_List.append(temp_list)            # line 32
        # anti-diagonal
        for r in range(0, d - s + 1):                        # line 33
            c = d - 1 - r                                    # line 34
            temp_list = []
            for j in range(0, s):                            # line 35
                temp_list.append((r + j) * d + (c - j))
            Extended_Error_List.append(temp_list)            # line 36

    return Extended_Error_List                               # line 37


# ==========================================================================
#  Complete fault list = Algorithm 2 + Algorithm 3 + Algorithm 4
#                        over every traversal mode
# ==========================================================================

def build_complete_fault_list(d, Delta, deduplicate):
    """Run all three algorithms for all traversals and merge the results.

    Every generated pattern is already in ascending index order, so a plain
    tuple() is a safe canonical key for duplicate removal.
    """
    fault_list = []

    for tau in TRAVERSALS:
        fault_list.extend(consecutive_error_list(d, tau))       # Algorithm 2
        fault_list.extend(nc_error_list(d, tau, Delta))         # Algorithm 3
        fault_list.extend(extended_error_list(d, tau))          # Algorithm 4

    raw_count = len(fault_list)

    if deduplicate:
        seen = set()
        unique_list = []
        for pattern in fault_list:
            key = tuple(pattern)
            if key not in seen:
                seen.add(key)
                unique_list.append(pattern)
        fault_list = unique_list

    return fault_list, raw_count


# ==========================================================================
#  Helper routines: counting, memory measurement, file output
# ==========================================================================

def count_by_weight(fault_list):
    """Number of s-data-qubit error sets, for each s present in the list."""
    counts = {}
    for pattern in fault_list:
        s = len(pattern)
        counts[s] = counts.get(s, 0) + 1
    return counts


def deep_size(obj, seen=None):
    """Total memory footprint of a nested list structure, in bytes.

    Object identities are tracked so that shared/interned objects (small
    integers in CPython) are counted once, giving the true resident size
    rather than an inflated per-reference sum.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            size += deep_size(item, seen)
    return size


def human_bytes(n):
    """Format a byte count with a readable unit."""
    unit_list = ['B', 'KB', 'MB', 'GB']
    value = float(n)
    for unit in unit_list:
        if value < 1024.0 or unit == 'GB':
            return "%.2f %s" % (value, unit)
        value = value / 1024.0


def write_fault_list(fault_list, filename):
    """One error pattern per line, indices separated by a single space."""
    with open(filename, 'w') as f:
        for pattern in fault_list:
            f.write(' '.join(str(q) for q in pattern))
            f.write('\n')


# ==========================================================================
#  Main driver
# ==========================================================================

def main():
    summary_rows = []

    print("=" * 74)
    print(" Significant data-qubit error list generation "
          "(Algorithms 2, 3 and 4)")
    print(" Separation set Delta = %s      Traversals = %s"
          % (DELTA_SET, TRAVERSALS))
    print(" Duplicate removal = %s" % DEDUPLICATE)
    print("=" * 74)

    for d in DISTANCES:
        m = (d - 1) // 2

        # ---- 1. generation time (file writing NOT included) --------------
        #
        # The generation takes only tens of microseconds, so a single cold
        # measurement is dominated by effects that have nothing to do with d:
        # one-off interpreter warm-up, and above all a cyclic garbage-collection
        # pass, whose ~45 us cost lands inside some distances and not others and
        # can make a larger d appear FASTER than a smaller one.  So: collect the
        # garbage first, disable the collector inside the timed region, run once
        # untimed to warm up, then keep the fastest of TIMING_REPEATS runs.
        #
        build_complete_fault_list(d, DELTA_SET, DEDUPLICATE)   # warm-up

        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            gen_time = float('inf')
            for _ in range(max(1, TIMING_REPEATS)):
                t_start = time.perf_counter()
                fault_list, raw_count = build_complete_fault_list(
                    d, DELTA_SET, DEDUPLICATE)
                t_end = time.perf_counter()
                gen_time = min(gen_time, t_end - t_start)
        finally:
            if gc_was_enabled:
                gc.enable()

        # ---- 2. size of the complete fault list --------------------------
        list_size = len(fault_list)

        # ---- 3. number of s-data-qubit error sets for each s -------------
        weight_counts = count_by_weight(fault_list)

        # ---- 4. memory footprint of the stored fault list ----------------
        mem_bytes = deep_size(fault_list)
        total_indices = sum(len(p) for p in fault_list)

        # ---- report ------------------------------------------------------
        print("")
        print("-" * 74)
        print(" d = %-3d   n = %-4d data qubits   m = floor((d-1)/2) = %d"
              % (d, d * d, m))
        print("-" * 74)
        print("  1. Generation time                : %.6f s  (%.1f us)"
              % (gen_time, gen_time * 1e6))
        print("       [fastest of %d runs, after warm-up, collector paused]"
              % max(1, TIMING_REPEATS))
        print("  2. Size of complete fault list    : %d error patterns"
              % list_size)
        if DEDUPLICATE:
            print("       (patterns emitted by the algorithms: %d, "
                  "duplicates removed: %d)"
                  % (raw_count, raw_count - list_size))
        print("  3. Error sets per weight s        :")
        for s in sorted(weight_counts):
            tag = "  <- s = m+1 (Algorithm 4)" if s == m + 1 else ""
            print("        s = %-3d : %8d pattern(s)%s"
                  % (s, weight_counts[s], tag))
        print("  4. Memory of stored fault list    : %s (%d bytes)"
              % (human_bytes(mem_bytes), mem_bytes))
        print("       total qubit indices stored   : %d" % total_indices)

        # ---- 5. write the fault list to file (excluded from timing) ------
        filename = os.path.join(OUTPUT_DIR, "fault_list_d%d.txt" % d)
        write_fault_list(fault_list, filename)
        file_bytes = os.path.getsize(filename)
        print("  5. Fault list written to          : %s (%s on disk)"
              % (filename, human_bytes(file_bytes)))

        summary_rows.append((d, gen_time, list_size, mem_bytes, file_bytes))

    # ---- combined summary table ------------------------------------------
    print("")
    print("=" * 74)
    print(" SUMMARY")
    print("=" * 74)
    print(" %-5s %-16s %-14s %-14s %-14s"
          % ("d", "gen time (s)", "list size", "memory", "file size"))
    print("-" * 74)
    for d, t, n, mem, fsz in summary_rows:
        print(" %-5d %-16.6f %-14d %-14s %-14s"
              % (d, t, n, human_bytes(mem), human_bytes(fsz)))
    print("=" * 74)


if __name__ == "__main__":
    main()
