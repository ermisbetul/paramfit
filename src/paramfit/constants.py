"""
constants.py

Physical constants and default model parameters.
"""

# Unit conversions
BOHR2ANG = 0.529177210903
HARTREE2KCAL = 627.5094740631

# Intrinsic Coulomb radii (Angstrom)
SMD_ICR_ANG = {
    "H": 1.20,
    "C": 1.85,
    "N": 1.55,
    "O": 1.52,
    "F": 1.50,
    "P": 1.85,
    "S": 1.80,
    "CL": 1.70,
    "BR": 1.85,
    "I": 1.98,
}

# Default Sij parameters
DEFAULT_SIJ = {
    # H as descreened atom
    ("H", "H"): 0.84,
    ("H", "C"): 1.00,
    ("H", "N"): 0.81,
    ("H", "O"): 0.50,
    ("H", "F"): 0.50,
    ("H", "P"): 1.00,
    ("H", "S"): 0.50,
    ("H", "CL"): 0.50,
    ("H", "BR"): 1.00,
    ("H", "I"): 1.00,

    # H as descreening atom
    ("C", "H"): 0.76,
    ("N", "H"): 0.75,
    ("O", "H"): 0.50,
    ("F", "H"): 0.65,
    ("P", "H"): 0.98,
    ("S", "H"): 0.78,
    ("CL", "H"): 0.76,
    ("BR", "H"): 0.59,
    ("I", "H"): 0.63,
}

# Fill remaining pairs
for a in ["C", "N", "O", "F", "P", "S", "CL", "BR", "I"]:
    for b in ["C", "N", "O", "F", "P", "S", "CL", "BR", "I"]:
        if a == "C":
            DEFAULT_SIJ[(a, b)] = 0.67
        elif a == "N":
            DEFAULT_SIJ[(a, b)] = 0.52
        elif a == "O":
            DEFAULT_SIJ[(a, b)] = 1.00
        elif a == "F":
            DEFAULT_SIJ[(a, b)] = 1.00
        elif a == "P":
            DEFAULT_SIJ[(a, b)] = 0.50
        elif a == "S":
            DEFAULT_SIJ[(a, b)] = 1.00
        elif a == "CL":
            DEFAULT_SIJ[(a, b)] = 1.00
        elif a == "BR":
            DEFAULT_SIJ[(a, b)] = 0.86
        elif a == "I":
            DEFAULT_SIJ[(a, b)] = 1.00
