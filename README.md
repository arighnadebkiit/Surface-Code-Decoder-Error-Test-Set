# Surface Code Decoder — Error Test Set Generation

Automatic generation of data-qubit error (fault) lists and error-syndrome test patterns for testing and verifying decoders of the **distance-d rotated surface code**.

For each code distance d ∈ {5, 7, 9, 11, 13}, the scripts produce:

- **T₁**: the single data-qubit error test set (syndrome of every single-qubit error)
- **L**: the list of *significant* multi-qubit error configurations (consecutive, non-consecutive and (m+1)-weight errors along horizontal, vertical and diagonal traversals)
- **T_sig**: the error-syndrome pattern test set, i.e. the Z- and X-stabilizer syndrome of every configuration in L, together with an analysis of degenerate (syndrome-sharing) cases

## Conventions

| Symbol | Meaning |
|---|---|
| d | code distance |
| n = d² | number of data qubits, labelled row-major: q[r][c] = r·d + c |
| p = (d² − 1)/2 | number of stabilizers of each Pauli type |
| m = ⌊(d − 1)/2⌋ | maximum correctable error weight |
| τ ∈ {H, V, D} | traversal mode: horizontal, vertical, diagonal (principal and anti-diagonal) |
| Δ | separation set for non-consecutive errors (default Δ = {2}) |

Syndromes are computed over GF(2): the syndrome of an error configuration is the XOR of the single-qubit syndrome columns of the qubits it contains.

## Algorithms

| Algorithm | Purpose | Implemented in |
|---|---|---|
| 1 | Single data-qubit error test set T₁ = {T₁ᶻ, T₁ˣ} | `gen_T1.py`, `gen_Tsig.py` |
| 2 | Significant consecutive multi-qubit errors, 2 ≤ s ≤ m | `surface_code_fault_list.py`, `gen_Tsig.py` |
| 3 | Significant non-consecutive multi-qubit errors, 2 ≤ s ≤ m, separation Δ | `surface_code_fault_list.py`, `gen_Tsig.py` |
| 4 | Significant consecutive (m+1)-qubit errors | `surface_code_fault_list.py`, `gen_Tsig.py` |
| 5 | Error-syndrome pattern test set T_sig | `gen_Tsig.py` |

The complete fault list L is the union of the outputs of Algorithms 2–4 over all traversal modes, with duplicates removed.

## Repository layout

```
input_files/            Stabilizer-incidence files for each distance
  Qbit_and_Z_d<d>.txt     data qubit -> Z-stabilizers it touches
  Qbit_and_X_d<d>.txt     data qubit -> X-stabilizers it touches
source_codes/
  gen_T1.py                   Algorithm 1
  surface_code_fault_list.py  Algorithms 2-4
  gen_Tsig.py                 Algorithms 1-5 (self-contained)
output_error_lists/     fault_list_d<d>.txt   significant error list L
output_test_patterns/   T1_{X,Z}syndrome_d<d>.txt, Tsig_{X,Z}syndrome_d<d>.txt
output_stats/           full_report.txt (timing, sizes, degeneracy), T1_stats.jpg
```

## Input file format

One line per data qubit, listing the stabilizers of one type that act on it. A boundary qubit touches only one stabilizer of that type.

```
q0 z8
q1 z0 z8
```

## Requirements

Python 3.8 or later. No third-party packages are needed.

## Usage

Run from the repository root.

```bash
# Algorithm 1: single-qubit test set T1
python3 source_codes/gen_T1.py --dir input_files --out output_test_patterns

# Algorithms 1-5: syndrome test set T_sig and the full statistics report
python3 source_codes/gen_Tsig.py --dir input_files --out output_test_patterns > output_stats/full_report.txt

# Algorithms 2-4: significant error list L (writes to the current directory)
cd output_error_lists && python3 ../source_codes/surface_code_fault_list.py && cd ..
```

Options for `gen_T1.py` and `gen_Tsig.py`:

| Option | Meaning |
|---|---|
| `--dir DIR` | directory containing the `Qbit_and_*` input files |
| `--out DIR` | directory for output files |
| `-d 5 7 ...` | process only these distances |
| `--repeats N` | repeat generation N times and report the fastest run |

The distances, traversal modes, separation set Δ and deduplication flag are set in the configuration block at the top of each script.

## Output formats

- **Fault list** (`fault_list_d<d>.txt`): one error configuration per line, data-qubit indices separated by spaces.
- **T₁ files**: one row per data qubit giving the syndrome as a bit string, LSB first (leftmost character is stabilizer 0).
- **T_sig files**: syndrome patterns grouped by error weight s, each paired with the error configuration that produces it.

## Results

From `output_stats/full_report.txt` (Δ = {2}, traversals H, V, D, deduplicated):

| d | n | p | \|T_sig\| | Memory | Degenerate classes (Z) | Degenerate classes (X) |
|---|---|---|---|---|---|---|
| 5 | 25 | 12 | 156 | 28.21 KB | 10 | 11 |
| 7 | 49 | 24 | 376 | 68.21 KB | 17 | 22 |
| 9 | 81 | 40 | 688 | 128.40 KB | 31 | 37 |
| 11 | 121 | 60 | 1092 | 209.67 KB | 41 | 52 |
| 13 | 169 | 84 | 1588 | 312.83 KB | 53 | 65 |

Generation time for the full T_sig stays under 1 ms up to d = 13 (file writing excluded). See the report for per-algorithm timing and the list of degenerate cases.

## Authors

- Mr. Ho Le Minh Toan, School of Electronics Engineering, KIIT Deemed To Be University, Bhubaneswar
- Dr. Arighna Deb, School of Electronics Engineering, KIIT Deemed To Be University, Bhubaneswar

## Citation

To Be Announced Later.
