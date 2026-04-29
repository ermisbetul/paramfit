"""
io.py

Handles reading reference data and quantum chemistry output files.
"""

import glob
import re
import numpy as np


def count_ref_entries(ref_file):
    nentries = 0

    with open(ref_file, "r") as f:
        for line in f:
            sp = line.split()
            if len(sp) >= 2:
                nentries += 1

    return nentries


def count_output_files():
    return len(glob.glob("[0-9]*_*.out"))


# -------------------------------------------------------------------------
# Reads reference solvation energies from file.
# Format:
# index   value
# -------------------------------------------------------------------------
def read_ref(ref_file, ndata):
    ref = np.zeros(ndata)

    with open(ref_file, "r") as f:
        for line in f:
            sp = line.split()
            if len(sp) < 2:
                continue

            idx = int(sp[0]) - 1
            if idx < 0 or idx >= ndata:
                raise RuntimeError(
                    f"Reference index {idx + 1} is out of range."
                )

            ref[idx] = float(sp[1])

    return ref


# -------------------------------------------------------------------------
# Reads all output files in current directory.
# Expected names:
# 1_xxx.out, 2_xxx.out, ...
# -------------------------------------------------------------------------
def read_outputs(ndata, qcmethod="SCF", charge_type="IAO_MULLIKEN"):

    files = sorted(
        glob.glob("[0-9]*_*.out"),
        key=lambda x: int(x.split("_")[0])
    )

    if len(files) != ndata:
        raise RuntimeError(
            f"Number of output files ({len(files)}) does not match ndata ({ndata})."
        )

    natoms = []
    symbols = []
    coords = []
    charges = []
    cds = []

    if charge_type.upper() == "IAO_MULLIKEN":
        charge_label = f"Mulliken Population Analysis for {qcmethod}-IAO:"
    else:
        charge_label = f"Lowdin Population Analysis for {qcmethod}-IAO:"

    for fn in files:
        with open(fn, "r", errors="ignore") as f:
            lines = f.read().splitlines()

        n = _read_natoms(lines, fn)
        mol_symbols, mol_coords = _read_coordinates(lines, fn, n)
        mol_charges = _read_charges(lines, fn, n, charge_label)
        cds_val = _read_cds(lines, fn, qcmethod)

        natoms.append(n)
        symbols.append(mol_symbols)
        coords.append(mol_coords)
        charges.append(mol_charges)
        cds.append(cds_val)

    return {
        "natoms": np.array(natoms, dtype=int),
        "symbols": symbols,
        "coords": coords,
        "charges": charges,
        "cds": np.array(cds, dtype=float),
    }


# -------------------------------------------------------------------------
# Reads number of atoms
# -------------------------------------------------------------------------
def _read_natoms(lines, fn):
    for line in lines:
        m = re.match(r"\s*Number of Atoms\s*:\s*(\d+)", line, re.I)
        if m:
            return int(m.group(1))

    raise RuntimeError(f"{fn}: Number of Atoms was not found.")


# -------------------------------------------------------------------------
# Reads Cartesian coordinates
# -------------------------------------------------------------------------
def _read_coordinates(lines, fn, n):
    mol_symbols = []
    mol_coords = []

    coord_rx = re.compile(
        r"^\s*\d+\s+([A-Za-z]+)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
    )

    for line in lines:
        m = coord_rx.match(line)
        if m:
            mol_symbols.append(m.group(1).upper())
            mol_coords.append([
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
            ])

            if len(mol_symbols) == n:
                break

    if len(mol_symbols) != n:
        raise RuntimeError(f"{fn}: Cartesian coordinates were not found.")

    return mol_symbols, np.array(mol_coords, dtype=float)


# -------------------------------------------------------------------------
# Reads atomic charges
# -------------------------------------------------------------------------
def _read_charges(lines, fn, n, charge_label):
    q = []
    read_block = False
    number_rx = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?")

    for line in lines:
        if not read_block and charge_label in line:
            read_block = True
            continue

        if read_block:
            if not line.strip():
                break

            nums = number_rx.findall(line)
            for val in nums:
                q.append(float(val))

    if len(q) < n:
        raise RuntimeError(f"{fn}: Charge block was not found or is incomplete.")

    return np.array(q[:n], dtype=float)


# -------------------------------------------------------------------------
# Reads CDS free energy
# -------------------------------------------------------------------------
def _read_cds(lines, fn, qcmethod):
    cds_rx = re.compile(
        r".*" + re.escape(qcmethod) +
        r"::CDS Free Energy \(kcal/mol\):\s*(-?\d+\.\d+)",
        re.I
    )

    for line in lines:
        m = cds_rx.match(line)
        if m:
            return float(m.group(1))

    raise RuntimeError(f"{fn}: CDS free energy was not found.")
