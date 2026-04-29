"""Command-line entry point for paramfit."""

import argparse

from .io import count_output_files, count_ref_entries, read_ref, read_outputs
from .optimizer import ParamFit


def _parse_atoms(text):
    if text.lower() == "all":
        return None
    return [item.strip().upper() for item in text.split(",") if item.strip()]


def _parse_pairs(text):
    if text.lower() == "auto":
        return None

    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        if "-" in item:
            left, right = item.split("-", 1)
        elif ":" in item:
            left, right = item.split(":", 1)
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid pair '{item}'. Use H-O,C-H or 'auto'."
            )

        pairs.append((left.strip().upper(), right.strip().upper()))

    return pairs


def _auto_sij_pairs(data, fit_atoms):
    atoms = fit_atoms
    if atoms is None:
        atoms = sorted(set(a for mol in data["symbols"] for a in mol))

    return [(a, b) for a in atoms for b in atoms]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="paramfit",
        description="Fit GB solvation parameters against reference solvation energies.",
    )

    parser.add_argument("--ndata", type=int, default=None)
    parser.add_argument("--ref-file", default="ref.txt")
    parser.add_argument("--output", default="paramfit.out")
    parser.add_argument("--qcmethod", default="SCF")
    parser.add_argument("--charge-type", default="IAO_MULLIKEN")
    parser.add_argument("--solvent-eps", type=float, default=78.3553)
    parser.add_argument("--gb-sij", type=float, default=0.75)
    parser.add_argument("--gb-dij", type=float, default=4.0)
    parser.add_argument("--gb-dch", type=float, default=4.0)
    parser.add_argument("--gb-cij", type=float, default=0.0)

    parser.add_argument("--fit-atoms", type=_parse_atoms, default=_parse_atoms("H,C,O"))
    parser.add_argument(
        "--parameter-mode",
        choices=["explicit", "hco-paper"],
        default="explicit",
        help="Use explicit Sij pairs or the seven HCO scale-factor groups used in the PD parametrization paper.",
    )
    parser.add_argument("--optimize-sij", action="store_true")
    parser.add_argument("--fit-sij-pairs", type=_parse_pairs, default=None)
    parser.add_argument("--optimize-dij", action="store_true")
    parser.add_argument("--optimize-dch", action="store_true")

    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--global-maxiter", type=int, default=80)
    parser.add_argument("--popsize", type=int, default=15)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--objective", choices=["mae", "rmse"], default="mae")

    parser.add_argument("--ftol", type=float, default=1.0e-12)
    parser.add_argument("--xtol", type=float, default=1.0e-12)
    parser.add_argument("--gtol", type=float, default=1.0e-12)
    parser.add_argument("--mae-tol", type=float, default=0.50)
    parser.add_argument("--rms-tol", type=float, default=0.80)
    parser.add_argument("--max-tol", type=float, default=2.50)

    return parser


def main():
    args = build_parser().parse_args()

    print("PARAMFIT: reading reference data and calculation outputs...")

    ndata = args.ndata
    if ndata is None:
        print(f"  Reading reference file: {args.ref_file}")
        nref = count_ref_entries(args.ref_file)
        print("  Scanning output files: [index]_[name].out")
        nout = count_output_files()
        if nref != nout:
            raise RuntimeError(
                f"Reference entries ({nref}) do not match output files ({nout}). "
                "Pass --ndata explicitly if this is intentional."
            )
        ndata = nref

    print(f"  Number of data points: {ndata}")
    refdata = read_ref(args.ref_file, ndata)
    print(f"  Reading {ndata} output files with {args.charge_type} charges...")
    data = read_outputs(
        ndata=ndata,
        qcmethod=args.qcmethod,
        charge_type=args.charge_type,
    )
    print("  Input data loaded successfully.")

    fit_sij_pairs = args.fit_sij_pairs
    if args.optimize_sij and fit_sij_pairs is None:
        fit_sij_pairs = _auto_sij_pairs(data, args.fit_atoms)

    fit = ParamFit(
        ndata=ndata,
        refdata=refdata,
        data=data,
        ref_file=args.ref_file,
        output_file=args.output,

        qcmethod=args.qcmethod,
        solvent_eps=args.solvent_eps,
        charge_type=args.charge_type,

        gb_sij=args.gb_sij,
        gb_dij=args.gb_dij,
        gb_dch=args.gb_dch,
        gb_cij=args.gb_cij,

        fit_atoms=args.fit_atoms,

        optimize_sij=args.optimize_sij,
        optimize_dij=args.optimize_dij,
        optimize_dch=args.optimize_dch,
        parameter_mode=args.parameter_mode,
        objective=args.objective,
        mae_tol=args.mae_tol,
        rms_tol=args.rms_tol,
        max_tol=args.max_tol,

        fit_sij_pairs=fit_sij_pairs,
    )

    if args.global_search:
        print(f"PARAMFIT: starting global {args.objective.upper()} optimization...")
        fit.fit_global(
            maxiter=args.global_maxiter,
            popsize=args.popsize,
            seed=args.seed,
            polish=not args.no_polish,
        )
    else:
        print("PARAMFIT: starting local least-squares optimization...")
        fit.fit_local(
            max_nfev=args.max_nfev,
            ftol=args.ftol,
            xtol=args.xtol,
            gtol=args.gtol,
        )

    print(f"PARAMFIT: finished. Output written to {args.output}")
