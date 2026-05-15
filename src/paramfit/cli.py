"""Command-line entry point for paramfit."""

import argparse
import dataclasses
from pathlib import Path
import tomllib

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


def _sij_pairs_with_base(fit_atoms, base_atoms):
    if not fit_atoms:
        raise RuntimeError("sij_pair_selection='fit-with-base' requires fit_atoms.")

    fit = [atom.upper() for atom in fit_atoms]
    base = [atom.upper() for atom in (base_atoms or [])]
    pairs = []

    for atom in fit:
        pairs.append((atom, atom))
        for other in base:
            if other == atom:
                continue
            pairs.append((atom, other))
            pairs.append((other, atom))

    return sorted(set(pairs))


@dataclasses.dataclass
class InputConfig:
    ndata: int | None = None
    ref_file: str = "ref.txt"
    output: str = "paramfit.out"
    parameter_output_file: str | None = None
    qcmethod: str = "SCF"
    charge_type: str = "IAO_MULLIKEN"
    solvent_eps: float = 78.3553
    gb_sij: float = 0.75
    gb_dij: float = 3.7
    dch: float = 3.7
    doh: float = 3.7
    dnh: float = 3.7
    gb_cij: float = 0.0
    fit_atoms: list[str] | None = dataclasses.field(default_factory=lambda: ["H", "C", "O"])
    base_atoms: list[str] | None = dataclasses.field(default_factory=list)
    initial_parameter_file: str | None = None
    sij_mode: str | None = None
    sij_group_scheme: str | None = None
    parameter_mode: str = "explicit"
    optimize_sij: bool = False
    sij_pair_selection: str | None = None
    sij_pair_mode: str = "auto"
    fit_sij_pairs: list[tuple[str, str]] | None = None
    optimize_dij: bool = False
    optimize_dch: bool = False
    optimize_doh: bool = False
    optimize_dnh: bool = False
    global_search: bool = False
    global_maxiter: int = 80
    popsize: int = 15
    polish: bool = True
    max_nfev: int = 200
    seed: int = 1
    objective: str = "mae"
    max_penalty_weight: float = 1.0
    ftol: float = 1.0e-12
    xtol: float = 1.0e-12
    gtol: float = 1.0e-12
    mae_tol: float = 0.50
    rms_tol: float = 0.80
    max_tol: float = 2.50


def _require_table(config, name):
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise RuntimeError(f"Input section [{name}] must be a TOML table.")
    return value


def _optional_atoms(value):
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_atoms(value)
    return [str(item).upper() for item in value]


def _optional_pairs(value):
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_pairs(value)

    pairs = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RuntimeError("fit_sij_pairs entries must be two-item arrays, e.g. ['H', 'O'].")
        pairs.append((str(item[0]).upper(), str(item[1]).upper()))
    return pairs


def _output_from_input(input_file):
    return str(Path(input_file).with_suffix(".out"))


def _normalize_sij_options(cfg):
    if cfg.sij_pair_selection is not None:
        selection = cfg.sij_pair_selection.lower()
        if selection == "manual":
            cfg.sij_pair_mode = "explicit"
        elif selection in ("auto", "fit-with-base"):
            cfg.sij_pair_mode = selection
        else:
            raise RuntimeError(
                "sij_pair_selection must be 'auto', 'fit-with-base', or 'manual'."
            )

    if cfg.sij_mode is not None:
        sij_mode = cfg.sij_mode.lower()
        if sij_mode == "explicit":
            cfg.parameter_mode = "explicit"
        elif sij_mode == "grouped":
            scheme = (cfg.sij_group_scheme or "hco").lower()
            if scheme in ("hco", "cho"):
                cfg.parameter_mode = "hco-grouped"
            elif scheme in ("halogen", "hal", "halo"):
                cfg.parameter_mode = "hal-grouped"
            else:
                raise RuntimeError(
                    "sij_group_scheme must be 'hco' or 'halogen' when sij_mode='grouped'."
                )
        else:
            raise RuntimeError("sij_mode must be 'explicit' or 'grouped'.")


def read_input_file(input_file):
    with open(input_file, "rb") as handle:
        raw = tomllib.load(handle)

    cfg = InputConfig(output=_output_from_input(input_file))

    files = _require_table(raw, "files")
    data = _require_table(raw, "data")
    parameters = _require_table(raw, "parameters")
    optimization = _require_table(raw, "optimization")
    tolerances = _require_table(raw, "tolerances")

    cfg.ref_file = files.get("ref_file", cfg.ref_file)
    cfg.output = files.get("output_file", cfg.output)
    cfg.parameter_output_file = files.get("parameter_output_file", cfg.parameter_output_file)

    cfg.ndata = data.get("ndata", cfg.ndata)
    cfg.qcmethod = data.get("qcmethod", cfg.qcmethod)
    cfg.charge_type = data.get("charge_type", cfg.charge_type)

    cfg.fit_atoms = _optional_atoms(parameters.get("fit_atoms", cfg.fit_atoms))
    cfg.base_atoms = _optional_atoms(parameters.get("base_atoms", cfg.base_atoms))
    cfg.initial_parameter_file = parameters.get("initial_parameter_file", cfg.initial_parameter_file)
    cfg.solvent_eps = parameters.get("solvent_eps", cfg.solvent_eps)
    cfg.gb_sij = parameters.get("gb_sij", cfg.gb_sij)
    cfg.gb_dij = parameters.get("gb_dij", cfg.gb_dij)
    cfg.dch = parameters.get("dch", cfg.dch)
    cfg.doh = parameters.get("doh", cfg.doh)
    cfg.dnh = parameters.get("dnh", cfg.dnh)
    cfg.gb_cij = parameters.get("gb_cij", cfg.gb_cij)
    cfg.sij_mode = parameters.get("sij_mode", cfg.sij_mode)
    cfg.sij_group_scheme = parameters.get("sij_group_scheme", cfg.sij_group_scheme)
    cfg.optimize_sij = parameters.get("optimize_sij", cfg.optimize_sij)
    cfg.sij_pair_selection = parameters.get("sij_pair_selection", cfg.sij_pair_selection)
    cfg.fit_sij_pairs = _optional_pairs(parameters.get("fit_sij_pairs", cfg.fit_sij_pairs))
    cfg.optimize_dij = parameters.get("optimize_dij", cfg.optimize_dij)
    cfg.optimize_dch = parameters.get("optimize_dch", cfg.optimize_dch)
    cfg.optimize_doh = parameters.get("optimize_doh", cfg.optimize_doh)
    cfg.optimize_dnh = parameters.get("optimize_dnh", cfg.optimize_dnh)

    algorithm = optimization.get("algorithm")
    if algorithm is not None:
        cfg.global_search = str(algorithm).lower() == "global"
    cfg.global_search = optimization.get("global_search", cfg.global_search)
    cfg.global_maxiter = optimization.get("global_maxiter", cfg.global_maxiter)
    cfg.popsize = optimization.get("popsize", cfg.popsize)
    cfg.polish = optimization.get("polish", cfg.polish)
    cfg.max_nfev = optimization.get("max_nfev", cfg.max_nfev)
    cfg.seed = optimization.get("seed", cfg.seed)
    cfg.objective = optimization.get("objective", cfg.objective)
    cfg.max_penalty_weight = optimization.get("max_penalty_weight", cfg.max_penalty_weight)
    cfg.ftol = optimization.get("ftol", cfg.ftol)
    cfg.xtol = optimization.get("xtol", cfg.xtol)
    cfg.gtol = optimization.get("gtol", cfg.gtol)

    cfg.mae_tol = tolerances.get("mae_tol", cfg.mae_tol)
    cfg.rms_tol = tolerances.get("rms_tol", cfg.rms_tol)
    cfg.max_tol = tolerances.get("max_tol", cfg.max_tol)

    return cfg


def build_parser():
    parser = argparse.ArgumentParser(
        prog="paramfit",
        description="Fit GB solvation parameters against reference solvation energies.",
    )

    parser.add_argument("--input", "-i", default=None, help="Read settings from a TOML-style .inp file.")
    parser.add_argument("--ndata", type=int, default=None)
    parser.add_argument("--ref-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--parameter-output-file", default=None)
    parser.add_argument("--qcmethod", default=None)
    parser.add_argument("--charge-type", default=None)
    parser.add_argument("--solvent-eps", type=float, default=None)
    parser.add_argument("--gb-sij", type=float, default=None)
    parser.add_argument("--gb-dij", type=float, default=None)
    parser.add_argument("--dch", type=float, default=None)
    parser.add_argument("--doh", type=float, default=None)
    parser.add_argument("--dnh", type=float, default=None)
    parser.add_argument("--gb-cij", type=float, default=None)

    parser.add_argument("--fit-atoms", type=_parse_atoms, default=None)
    parser.add_argument("--base-atoms", type=_parse_atoms, default=None)
    parser.add_argument("--initial-parameter-file", default=None)
    parser.add_argument(
        "--sij-mode",
        choices=["explicit", "grouped"],
        default=None,
        help="Use explicit pairwise Sij values or grouped Sij parameters.",
    )
    parser.add_argument(
        "--sij-group-scheme",
        choices=["hco", "cho", "halogen", "hal", "halo"],
        default=None,
        help="Grouped Sij scheme. Used only with --sij-mode grouped.",
    )
    parser.add_argument("--optimize-sij", action="store_true")
    parser.add_argument(
        "--sij-pair-selection",
        choices=["auto", "fit-with-base", "manual"],
        default=None,
        help=(
            "How explicit Sij pairs are selected. 'manual' requires "
            "--fit-sij-pairs."
        ),
    )
    parser.add_argument("--fit-sij-pairs", type=_parse_pairs, default=None)
    parser.add_argument("--optimize-dij", action="store_true")
    parser.add_argument("--optimize-dch", action="store_true")
    parser.add_argument("--optimize-doh", action="store_true")
    parser.add_argument("--optimize-dnh", action="store_true")

    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--global-maxiter", type=int, default=None)
    parser.add_argument("--popsize", type=int, default=None)
    parser.add_argument("--no-polish", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--objective", choices=["mae", "rmse", "mae-max"], default=None)
    parser.add_argument(
        "--max-penalty-weight",
        type=float,
        default=None,
        help="Penalty weight for objective=mae-max: MAE + weight * max(0, MAX - max_tol)^2.",
    )

    parser.add_argument("--ftol", type=float, default=None)
    parser.add_argument("--xtol", type=float, default=None)
    parser.add_argument("--gtol", type=float, default=None)
    parser.add_argument("--mae-tol", type=float, default=None)
    parser.add_argument("--rms-tol", type=float, default=None)
    parser.add_argument("--max-tol", type=float, default=None)

    return parser


def build_config(args):
    cfg = read_input_file(args.input) if args.input else InputConfig()

    for attr in [
        "ndata",
        "ref_file",
        "output",
        "parameter_output_file",
        "qcmethod",
        "charge_type",
        "solvent_eps",
        "gb_sij",
        "gb_dij",
        "dch",
        "doh",
        "dnh",
        "gb_cij",
        "fit_atoms",
        "base_atoms",
        "initial_parameter_file",
        "sij_mode",
        "sij_group_scheme",
        "sij_pair_selection",
        "fit_sij_pairs",
        "global_maxiter",
        "popsize",
        "max_nfev",
        "seed",
        "objective",
        "max_penalty_weight",
        "ftol",
        "xtol",
        "gtol",
        "mae_tol",
        "rms_tol",
        "max_tol",
    ]:
        value = getattr(args, attr)
        if value is not None:
            setattr(cfg, attr, value)

    for attr in [
        "optimize_sij",
        "optimize_dij",
        "optimize_dch",
        "optimize_doh",
        "optimize_dnh",
        "global_search",
    ]:
        if getattr(args, attr):
            setattr(cfg, attr, True)

    if args.no_polish:
        cfg.polish = False

    _normalize_sij_options(cfg)

    return cfg


def main():
    args = build_parser().parse_args()
    cfg = build_config(args)

    print("PARAMFIT: reading reference data and calculation outputs...")

    ndata = cfg.ndata
    if ndata is None:
        print(f"  Reading reference file: {cfg.ref_file}")
        nref = count_ref_entries(cfg.ref_file)
        print("  Scanning output files: [index]_[name].out")
        nout = count_output_files()
        if nref != nout:
            raise RuntimeError(
                f"Reference entries ({nref}) do not match output files ({nout}). "
                "Pass --ndata explicitly if this is intentional."
            )
        ndata = nref

    print(f"  Number of data points: {ndata}")
    refdata = read_ref(cfg.ref_file, ndata)
    print(f"  Reading {ndata} output files with {cfg.charge_type} charges...")
    data = read_outputs(
        ndata=ndata,
        qcmethod=cfg.qcmethod,
        charge_type=cfg.charge_type,
    )
    print("  Input data loaded successfully.")

    fit_sij_pairs = cfg.fit_sij_pairs
    if (
        cfg.optimize_sij
        and fit_sij_pairs is None
        and cfg.parameter_mode not in ("hco-grouped", "hal-grouped")
    ):
        if cfg.sij_pair_mode == "fit-with-base":
            fit_sij_pairs = _sij_pairs_with_base(cfg.fit_atoms, cfg.base_atoms)
        elif cfg.sij_pair_mode == "explicit":
            raise RuntimeError("sij_pair_selection='manual' requires fit_sij_pairs.")
        else:
            fit_sij_pairs = _auto_sij_pairs(data, cfg.fit_atoms)

    if fit_sij_pairs:
        print("  Sij pairs selected for optimization:")
        for left, right in fit_sij_pairs:
            print(f"    S_{left}_{right}")

    fit = ParamFit(
        ndata=ndata,
        refdata=refdata,
        data=data,
        ref_file=cfg.ref_file,
        output_file=cfg.output,
        parameter_output_file=cfg.parameter_output_file,
        initial_parameter_file=cfg.initial_parameter_file,

        qcmethod=cfg.qcmethod,
        solvent_eps=cfg.solvent_eps,
        charge_type=cfg.charge_type,

        gb_sij=cfg.gb_sij,
        gb_dij=cfg.gb_dij,
        dch=cfg.dch,
        doh=cfg.doh,
        dnh=cfg.dnh,
        gb_cij=cfg.gb_cij,

        fit_atoms=cfg.fit_atoms,

        optimize_sij=cfg.optimize_sij,
        optimize_dij=cfg.optimize_dij,
        optimize_dch=cfg.optimize_dch,
        optimize_doh=cfg.optimize_doh,
        optimize_dnh=cfg.optimize_dnh,
        parameter_mode=cfg.parameter_mode,
        objective=cfg.objective,
        max_penalty_weight=cfg.max_penalty_weight,
        mae_tol=cfg.mae_tol,
        rms_tol=cfg.rms_tol,
        max_tol=cfg.max_tol,

        fit_sij_pairs=fit_sij_pairs,
    )

    if cfg.initial_parameter_file:
        print(f"  Loading initial parameters from: {cfg.initial_parameter_file}")
        fit.load_parameter_file(cfg.initial_parameter_file)

    if cfg.global_search:
        print(f"PARAMFIT: starting global {cfg.objective.upper()} optimization...")
        fit.fit_global(
            maxiter=cfg.global_maxiter,
            popsize=cfg.popsize,
            seed=cfg.seed,
            polish=cfg.polish,
        )
    else:
        print("PARAMFIT: starting local least-squares optimization...")
        fit.fit_local(
            max_nfev=cfg.max_nfev,
            ftol=cfg.ftol,
            xtol=cfg.xtol,
            gtol=cfg.gtol,
        )

    print(f"PARAMFIT: finished. Output written to {cfg.output}")
