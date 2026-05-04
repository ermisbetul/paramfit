"""
optimizer.py

Parameter optimization driver.

This module contains:
- parameter vector packing/unpacking
- residual calculation
- local nonlinear least-squares fitting
- optional differential evolution + local refinement
- output file printing
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from scipy.optimize import least_squares, differential_evolution

from .constants import BOHR2ANG, SMD_ICR_ANG, DEFAULT_SIJ
from .solvation import GBSolvationModel

HALOGENS = ("F", "CL", "BR")
HALOGEN_GROUPS = ("XX", "XH", "HX", "XC", "CX", "XO", "OX", "XN", "NX", "XS", "SX", "XP", "PX")


@dataclass
class ParamFit:
    ndata: int
    refdata: np.ndarray
    data: dict
    ref_file: str = "ref.txt"
    output_file: str = "paramfit.out"
    parameter_output_file: str | None = None

    qcmethod: str = "SCF"
    solvent_eps: float = 78.3553
    charge_type: str = "IAO_MULLIKEN"

    gb_sij: float = 0.75
    gb_dij: float = 4.0
    gb_dch: float = 4.0
    gb_dnh: float = 4.0
    gb_cij: float = 0.0

    optimize_sij: bool = False
    optimize_dij: bool = False
    optimize_dch: bool = False
    optimize_dnh: bool = False
    use_dnh: bool = False
    parameter_mode: str = "explicit"
    objective: str = "mae"
    max_penalty_weight: float = 1.0

    mae_tol: float = 0.50
    rms_tol: float = 0.80
    max_tol: float = 2.50

    fit_atoms: list[str] | None = None
    fit_sij_pairs: list[tuple[str, str]] | None = None

    rho_bohr: dict[str, float] = field(default_factory=dict, init=False)
    sij: dict[tuple[str, str], float] = field(default_factory=lambda: dict(DEFAULT_SIJ), init=False)

    out: object = field(default=None, init=False)
    iteration: int = field(default=0, init=False)

    model: GBSolvationModel = field(default=None, init=False)

    # -------------------------------------------------------------------------
    # Initializes the model and intrinsic Coulomb radii.
    # -------------------------------------------------------------------------
    def __post_init__(self):
        self.model = GBSolvationModel(
            solvent_eps=self.solvent_eps,
            gb_sij=self.gb_sij,
            gb_dij=self.gb_dij,
            gb_cij=self.gb_cij,
        )

        self._build_distance_matrices()
        self._initialize_radii()

    def load_parameter_file(self, parameter_file):
        with open(parameter_file, "rb") as handle:
            params = tomllib.load(handle)

        for atom, value in params.get("rho_ang", {}).items():
            self.rho_bohr[atom.upper()] = float(value) / BOHR2ANG

        for key, value in params.get("sij", {}).items():
            left, right = key.upper().split("_", 1)
            self.sij[(left, right)] = float(value)

        dij = params.get("dij", {})
        if "d0" in dij:
            self.gb_dij = float(dij["d0"])
        if "dCH" in dij:
            self.gb_dch = float(dij["dCH"])
        if "dNH" in dij:
            self.gb_dnh = float(dij["dNH"])
            self.use_dnh = True

    def parameter_file_name(self):
        if self.parameter_output_file:
            return self.parameter_output_file
        return str(Path(self.output_file).with_suffix(".params.toml"))

    def write_parameter_file(self, x, names):
        rho, sij, gb_dij = self.unpack(x, names)
        parameter_file = self.parameter_file_name()

        with open(parameter_file, "w") as handle:
            handle.write("[metadata]\n")
            handle.write(f'source_output = "{self.output_file}"\n')
            handle.write(f'parameter_mode = "{self.parameter_mode}"\n')
            handle.write(f'objective = "{self.objective}"\n')
            handle.write("cds_treatment = \"fixed\"\n")
            handle.write("\n")

            handle.write("[rho_ang]\n")
            for atom in sorted(rho):
                handle.write(f"{atom} = {rho[atom] * BOHR2ANG:.10f}\n")
            handle.write("\n")

            handle.write("[sij]\n")
            for pair in sorted(sij):
                handle.write(f"{pair[0]}_{pair[1]} = {sij[pair]:.10f}\n")
            handle.write("\n")

            handle.write("[dij]\n")
            if isinstance(gb_dij, dict):
                handle.write(f"d0 = {gb_dij['d0']:.10f}\n")
                handle.write(f"dCH = {gb_dij['dCH']:.10f}\n")
                if "dNH" in gb_dij:
                    handle.write(f"dNH = {gb_dij['dNH']:.10f}\n")
            else:
                handle.write(f"d0 = {gb_dij:.10f}\n")
                handle.write(f"dCH = {self.gb_dch:.10f}\n")
                if self.use_dnh or self.optimize_dnh:
                    handle.write(f"dNH = {self.gb_dnh:.10f}\n")

        return parameter_file

    # -------------------------------------------------------------------------
    # Opens the output file. This replaces outfile->Printf behavior in C++.
    # -------------------------------------------------------------------------
    def open_output(self):
        self.out = open(self.output_file, "w")

    # -------------------------------------------------------------------------
    # Closes the output file.
    # -------------------------------------------------------------------------
    def close_output(self):
        if self.out is not None:
            self.out.close()
            self.out = None

    # -------------------------------------------------------------------------
    # Writes formatted text to the output file.
    # -------------------------------------------------------------------------
    def printf(self, msg="", *args):
        if args:
            msg = msg % args

        if self.out is not None:
            self.out.write(msg)
            self.out.flush()
        else:
            print(msg, end="")

    # -------------------------------------------------------------------------
    # Builds distance matrices Rij for all molecules.
    # Distances are stored in Angstrom.
    # -------------------------------------------------------------------------
    def _build_distance_matrices(self):
        rij = []

        for xyz in self.data["coords"]:
            diff = xyz[:, None, :] - xyz[None, :, :]
            rij.append(np.linalg.norm(diff, axis=2))

        self.data["rij"] = rij

    # -------------------------------------------------------------------------
    # Initializes intrinsic Coulomb radii.
    # Internal unit is bohr.
    # Printed unit is Angstrom.
    # -------------------------------------------------------------------------
    def _initialize_radii(self):
        atoms = sorted(set(a for mol in self.data["symbols"] for a in mol))

        for atom in atoms:
            if atom not in SMD_ICR_ANG:
                raise RuntimeError(f"No intrinsic Coulomb radius is available for atom type {atom}.")

            self.rho_bohr[atom] = SMD_ICR_ANG[atom] / BOHR2ANG

        if self.fit_atoms is None:
            self.fit_atoms = atoms

        if self.fit_sij_pairs is None:
            self.fit_sij_pairs = []

    # -------------------------------------------------------------------------
    # Packs fitting parameters into optimizer vector x.
    # -------------------------------------------------------------------------
    def pack_x0_and_bounds(self):
        x0 = []
        lower = []
        upper = []
        names = []
        parameter_mode = self.parameter_mode.lower()

        for atom in self.fit_atoms:
            if atom not in self.rho_bohr:
                raise RuntimeError(f"Atom type {atom} was requested for fitting but is not present.")

            x0.append(self.rho_bohr[atom])
            lower.append(0.5 / BOHR2ANG)
            upper.append(3.0 / BOHR2ANG)
            names.append(("rho", atom))

        if self.optimize_sij and parameter_mode in ("hco-grouped", "hco-paper"):
            for group in ["HH", "HC", "HO", "CH", "OH", "CX", "OX"]:
                x0.append(self._initial_sij_group_value(group))
                lower.append(0.5)
                upper.append(1.0)
                names.append(("sij_group", group))
        elif self.optimize_sij and parameter_mode == "hal-grouped":
            for group in HALOGEN_GROUPS:
                x0.append(self._initial_hal_sij_group_value(group))
                lower.append(0.5)
                upper.append(1.0)
                names.append(("hal_sij_group", group))
        elif self.optimize_sij:
            for pair in self.fit_sij_pairs:
                if pair not in self.sij:
                    raise RuntimeError(f"No initial Sij value is available for pair {pair}.")

                x0.append(self.sij[pair])
                lower.append(0.5)
                upper.append(1.0)
                names.append(("sij", pair))

        if self.optimize_dij:
            x0.append(self.gb_dij)
            if parameter_mode in ("hco-grouped", "hco-paper") or self.optimize_dch:
                lower.append(3.0)
                upper.append(5.0)
            else:
                lower.append(2.0)
                upper.append(8.0)
            names.append(("dij", "d0"))

        if self.optimize_dch:
            x0.append(self.gb_dch)
            lower.append(3.0)
            upper.append(5.0)
            names.append(("dij", "dCH"))

        if self.optimize_dnh:
            x0.append(self.gb_dnh)
            lower.append(3.0)
            upper.append(5.0)
            names.append(("dij", "dNH"))

        return np.array(x0), (np.array(lower), np.array(upper)), names

    def _initial_sij_group_value(self, group):
        values = {
            "HH": self.sij[("H", "H")],
            "HC": self.sij[("H", "C")],
            "HO": self.sij[("H", "O")],
            "CH": self.sij[("C", "H")],
            "OH": self.sij[("O", "H")],
            "CX": self.sij[("C", "C")],
            "OX": self.sij[("O", "C")],
        }

        if group not in values:
            raise RuntimeError(f"Unknown HCO Sij group {group}.")

        return values[group]

    def _halogen_group_pairs(self, group):
        if group == "XX":
            return [(left, right) for left in HALOGENS for right in HALOGENS]

        if len(group) != 2:
            raise RuntimeError(f"Unknown halogen Sij group {group}.")

        left, right = group
        left_atoms = HALOGENS if left == "X" else (left,)
        right_atoms = HALOGENS if right == "X" else (right,)
        return [(left_atom, right_atom) for left_atom in left_atoms for right_atom in right_atoms]

    def _initial_hal_sij_group_value(self, group):
        pairs = self._halogen_group_pairs(group)
        values = []

        for pair in pairs:
            if pair not in self.sij:
                raise RuntimeError(f"No initial Sij value is available for pair {pair}.")
            values.append(self.sij[pair])

        return float(np.mean(values))

    def _apply_sij_group(self, sij, group, value):
        if group == "HH":
            sij[("H", "H")] = value
        elif group == "HC":
            sij[("H", "C")] = value
        elif group == "HO":
            sij[("H", "O")] = value
        elif group == "CH":
            sij[("C", "H")] = value
        elif group == "OH":
            sij[("O", "H")] = value
        elif group == "CX":
            sij[("C", "C")] = value
            sij[("C", "O")] = value
        elif group == "OX":
            sij[("O", "C")] = value
            sij[("O", "O")] = value
        else:
            raise RuntimeError(f"Unknown HCO Sij group {group}.")

    def _apply_hal_sij_group(self, sij, group, value):
        for pair in self._halogen_group_pairs(group):
            sij[pair] = value

    # -------------------------------------------------------------------------
    # Converts optimizer vector x back to physical parameters.
    # -------------------------------------------------------------------------
    def unpack(self, x, names):
        rho = dict(self.rho_bohr)
        sij = dict(self.sij)
        gb_dij = {"d0": self.gb_dij, "dCH": self.gb_dch}
        if self.use_dnh or self.optimize_dnh:
            gb_dij["dNH"] = self.gb_dnh

        for value, name in zip(x, names):
            if name[0] == "rho":
                rho[name[1]] = value
            elif name[0] == "sij":
                sij[name[1]] = value
            elif name[0] == "sij_group":
                self._apply_sij_group(sij, name[1], value)
            elif name[0] == "hal_sij_group":
                self._apply_hal_sij_group(sij, name[1], value)
            elif name[0] == "dij":
                gb_dij[name[1]] = value
            else:
                raise RuntimeError(f"Unknown parameter type {name[0]}.")

        return rho, sij, gb_dij

    # -------------------------------------------------------------------------
    # Computes full energy/error table for all data points.
    # Rows are sorted by absolute error from largest to smallest.
    # -------------------------------------------------------------------------
    def compute_energy_table(self, x, names):
        rho, sij, gb_dij = self.unpack(x, names)

        rows = []

        for i in range(self.ndata):
            symbols = self.data["symbols"][i]
            charges = self.data["charges"][i]
            rij_ang = self.data["rij"][i]
            cds_e = self.data["cds"][i]
            ref = self.refdata[i]

            pol_e = self.model.compute_polarization_energy(
                symbols=symbols,
                charges=charges,
                rij_ang=rij_ang,
                rho=rho,
                sij=sij,
                gb_dij=gb_dij,
                optimize_sij=self.optimize_sij,
            )

            calc = pol_e + cds_e
            err = calc - ref

            rows.append((abs(err), i + 1, ref, pol_e, cds_e, calc, err))

        rows.sort(reverse=True)
        return rows

    # -------------------------------------------------------------------------
    # Computes residuals without printing.
    # residual_i = calculated_i - reference_i
    # -------------------------------------------------------------------------
    def residuals_no_print(self, x, names):
        try:
            rows = self.compute_energy_table(x, names)
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError):
            return np.full(self.ndata, 1.0e6)

        residual = np.zeros(self.ndata)

        for row in rows:
            _, idx, ref, pol_e, cds_e, calc, err = row
            residual[idx - 1] = err

        if not np.all(np.isfinite(residual)):
            return np.full(self.ndata, 1.0e6)

        return residual

    # -------------------------------------------------------------------------
    # Computes residuals and prints iteration information.
    # scipy may call this more often than formal optimizer iterations.
    # -------------------------------------------------------------------------
    def residuals(self, x, names):
        res = self.residuals_no_print(x, names)

        self.iteration += 1
        self.print_iteration(x, names, res)

        return res

    # -------------------------------------------------------------------------
    # Objective function for global optimization.
    # Uses MAE.
    # -------------------------------------------------------------------------
    def objective_mae(self, x, names):
        res = self.residuals_no_print(x, names)
        return np.mean(np.abs(res))

    # -------------------------------------------------------------------------
    # Objective function for global optimization.
    # Uses RMSE.
    # -------------------------------------------------------------------------
    def objective_rmse(self, x, names):
        res = self.residuals_no_print(x, names)
        return np.sqrt(np.mean(res**2))

    def objective_value(self, x, names):
        objective = self.objective.lower()
        if objective == "mae":
            return self.objective_mae(x, names)
        if objective == "rmse":
            return self.objective_rmse(x, names)
        if objective == "mae-max":
            res = self.residuals_no_print(x, names)
            mae = np.mean(np.abs(res))
            maxerr = np.max(np.abs(res))
            excess = max(0.0, maxerr - self.max_tol)
            return mae + self.max_penalty_weight * excess**2

        raise RuntimeError(f"Unknown objective function {self.objective}.")

    # -------------------------------------------------------------------------
    # Prints initial job information and initial parameters.
    # -------------------------------------------------------------------------
    def print_header(self, x0, names):
        self.printf("\n")
        self.printf("  ============================================================\n")
        self.printf("                    PARAMETER OPTIMIZATION\n")
        self.printf("  ============================================================\n\n")

        self.printf("  Number of data points        : %8d\n", self.ndata)
        self.printf("  Solvent dielectric constant  : %12.6f\n", self.solvent_eps)
        self.printf("  QC method                    : %s\n", self.qcmethod)
        self.printf("  Charge type                  : %s\n", self.charge_type)
        self.printf("  CDS treatment                : fixed from output files\n")
        self.printf("  Parameter mode               : %s\n", self.parameter_mode)
        self.printf("  Objective                    : %s\n", self.objective.upper())
        self.printf("  Optimize Sij                 : %s\n", str(self.optimize_sij))
        self.printf("  Optimize d0                  : %s\n", str(self.optimize_dij))
        self.printf("  Optimize dCH                 : %s\n", str(self.optimize_dch))
        self.printf("  Use dNH                     : %s\n", str(self.use_dnh or self.optimize_dnh))
        self.printf("  Optimize dNH                : %s\n", str(self.optimize_dnh))
        self.printf("  Fit atoms                    : %s\n", ", ".join(self.fit_atoms))
        if self.optimize_sij and self.fit_sij_pairs:
            self.printf("  Optimized Sij pairs          :\n")
            for pair in self.fit_sij_pairs:
                self.printf("    S_%s_%s\n", pair[0], pair[1])
        self.printf("\n")

        self.printf("  Initial Parameters\n")
        self.printf("  ------------------\n")

        for name, value in zip(names, x0):
            if name[0] == "rho":
                self.printf("    rho_%-2s = %14.8f ang\n", name[1], value * BOHR2ANG)
            elif name[0] == "sij":
                self.printf("    S_%-2s_%-2s = %14.8f\n", name[1][0], name[1][1], value)
            elif name[0] == "sij_group":
                self.printf("    S_%-3s = %14.8f\n", name[1], value)
            elif name[0] == "hal_sij_group":
                self.printf("    S_%-3s = %14.8f\n", name[1], value)
            elif name[0] == "dij":
                self.printf("    %-3s = %14.8f\n", name[1], value)

        self.printf("\n")

    # -------------------------------------------------------------------------
    # Prints initial energies before optimization.
    # -------------------------------------------------------------------------
    def print_initial_energies(self, x0, names):
        res = self.residuals_no_print(x0, names)
        rms = np.sqrt(np.mean(res**2))
        mae = np.mean(np.abs(res))
        maxerr = np.max(np.abs(res))
    
        self.printf("\n")
        self.printf("    =========================================================================================\n")
        self.printf("                           INITIAL SOLVATION FREE ENERGY RESULTS\n")
        self.printf("    =========================================================================================\n\n")
    
        rows = self.compute_energy_table(x0, names)
    
        self.printf("    ----------------------------------------------------------------------------------------------------\n")
        self.printf("    Index     Data         Ref. dGsolv         Pol. Energy       CDS        Calc. dGsolv       Error\n")
        self.printf("               No          (kcal/mol)          (kcal/mol)    (kcal/mol)      (kcal/mol)      (kcal/mol)\n")
        self.printf("    ----------------------------------------------------------------------------------------------------\n")
    
        for rank, row in enumerate(rows, start=1):
            _, idx, ref, pol_e, cds_e, calc, err = row
            self.printf(
                "    [%3d]  %6d        %10.2f          %10.2f    %10.2f      %10.2f      %10.2f\n",
                rank, idx, ref, pol_e, cds_e, calc, err
            )
    
        self.printf("    -----------------------------------------------------------------------------------------\n")
        self.printf("    RMS Error = %16.3f kcal/mol\n", rms)
        self.printf("    MAE Error = %16.3f kcal/mol\n", mae)
        self.printf("    MAX Error = %16.3f kcal/mol\n", maxerr)
        self.printf("    =========================================================================================\n\n")
    
        self.printf("\n")
        self.printf("    Initial        Tolerance  Value (kcal/mol)  \n")
        self.printf("    ----------------------------------------------------------------------------------\n")
        self.printf("    MAE Residual:     %9.3f        %9.3f    \n", self.mae_tol, mae)
        self.printf("    RMS Residual:     %9.3f        %9.3f    \n", self.rms_tol, rms)
        self.printf("    MAX Residual:     %9.3f        %9.3f    \n", self.max_tol, maxerr)
        self.printf("\n")
    # -------------------------------------------------------------------------
    # Prints one full optimization iteration.
    # Includes parameters, statistics, and the full sorted energy table.
    # -------------------------------------------------------------------------
    def print_iteration(self, x, names, res):
        rms = np.sqrt(np.mean(res**2))
        mae = np.mean(np.abs(res))
        maxerr = np.max(np.abs(res))
    
        self.printf("    =========================================================================================\n")
        self.printf("                                      ITER: %d \n", self.iteration)
        self.printf("    =========================================================================================\n\n")
    
        self.printf("    Iter: %-6d      Tolerance          Value (kcal/mol)  Converged? \n", self.iteration)
        self.printf("    ----------------------------------------------------------------------------------\n")
        self.printf(
            "    MAE Residual:     %9s        %9.3f    %s\n",
            f"{self.mae_tol:.3f}", mae, "YES" if mae < self.mae_tol else "NO"
        )
        self.printf(
            "    RMS Residual:     %9.3f        %9.3f    %s\n",
            self.rms_tol, rms, "YES" if rms < self.rms_tol else "NO"
        )
        self.printf(
            "    MAX Residual:     %9.3f        %9.3f    %s\n",
            self.max_tol, maxerr, "YES" if maxerr < self.max_tol else "NO"
        )
        self.printf("\n")
    
        self.printf("                     Updated parameters (in Ang):\n")
        self.printf("                    -----------------------------\n")
    
        for name, value in zip(names, x):
            if name[0] == "rho":
                self.printf("                          %-2s        %8.2f\n", name[1], value * BOHR2ANG)
            elif name[0] == "sij":
                self.printf("                          S_%-2s_%-2s   %8.4f\n", name[1][0], name[1][1], value)
            elif name[0] == "sij_group":
                self.printf("                          S_%-3s     %8.4f\n", name[1], value)
            elif name[0] == "dij":
                self.printf("                          %-3s       %8.4f\n", name[1], value)
    
        self.printf("\n")
    # -------------------------------------------------------------------------
    # Performs local nonlinear least-squares optimization.
    # -------------------------------------------------------------------------
    def fit_local(self, max_nfev=200, ftol=1.0e-12, xtol=1.0e-12, gtol=1.0e-12):
        x0, bounds, names = self.pack_x0_and_bounds()

        self.open_output()
        self.print_program_banner()
        self.print_input_summary()
        self.print_header(x0, names)
        self.print_initial_radii(x0, names)
        self.print_initial_energies(x0, names)

        result = least_squares(
            lambda x: self.residuals(x, names),
            x0,
            bounds=bounds,
            max_nfev=max_nfev,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            x_scale="jac",
            verbose=0,
        )

        self.print_result(result.x, names)
        parameter_file = self.write_parameter_file(result.x, names)

        self.printf("\n")
        self.printf("  Optimization Finished\n")
        self.printf("  ---------------------\n")
        self.printf("  Success : %s\n", str(result.success))
        self.printf("  Message : %s\n", result.message)
        self.printf("  Nfev    : %d\n", result.nfev)
        self.printf("  Parameter file : %s\n", parameter_file)

        self.close_output()

        return result, names

    # -------------------------------------------------------------------------
    # Performs global population-based optimization against the selected scalar
    # objective. For CHO parametrization this is normally MAE.
    # -------------------------------------------------------------------------
    def fit_global(
        self,
        maxiter=80,
        popsize=15,
        seed=1,
        polish=True,
    ):
        x0, bounds, names = self.pack_x0_and_bounds()
        scipy_bounds = list(zip(bounds[0], bounds[1]))

        self.open_output()
        self.print_program_banner()
        self.print_input_summary()
        self.print_header(x0, names)
        self.print_initial_energies(x0, names)

        self.printf("  Starting Global Optimization\n")
        self.printf("  ----------------------------\n")
        self.printf("  Algorithm       : scipy.optimize.differential_evolution\n")
        self.printf("  Scalar objective: %s\n", self.objective.upper())
        self.printf("  Max generations : %d\n", maxiter)
        self.printf("  Population size : %d x n_parameters\n", popsize)
        self.printf("  Random seed     : %d\n", seed)
        self.printf("  Final polish    : %s\n\n", str(polish))

        def callback(xk, convergence):
            value = self.objective_value(xk, names)
            self.iteration += 1
            self.printf(
                "    DE generation %-6d best %s = %14.8f kcal/mol   convergence = %.6e\n",
                self.iteration,
                self.objective.upper(),
                value,
                convergence,
            )

        de_result = differential_evolution(
            lambda x: self.objective_value(x, names),
            scipy_bounds,
            maxiter=maxiter,
            popsize=popsize,
            polish=polish,
            updating="immediate",
            workers=1,
            seed=seed,
            callback=callback,
            disp=False,
        )

        self.printf("\n")
        self.printf("  Differential Evolution Finished\n")
        self.printf("  Best %s = %14.8f kcal/mol\n\n", self.objective.upper(), de_result.fun)

        self.print_result(de_result.x, names)
        parameter_file = self.write_parameter_file(de_result.x, names)

        self.printf("\n")
        self.printf("  Optimization Finished\n")
        self.printf("  ---------------------\n")
        self.printf("  Success : %s\n", str(de_result.success))
        self.printf("  Message : %s\n", de_result.message)
        self.printf("  Nfev    : %d\n", de_result.nfev)
        self.printf("  Parameter file : %s\n", parameter_file)

        self.close_output()

        return de_result, names

    # -------------------------------------------------------------------------
    # Prints final parameters, statistics, and final full sorted error table.
    # -------------------------------------------------------------------------
    def print_result(self, x, names):
        res = self.residuals_no_print(x, names)

        rmse = np.sqrt(np.mean(res**2))
        mae = np.mean(np.abs(res))
        maxerr = np.max(np.abs(res))
        rows = self.compute_energy_table(x, names)
        positive_pol = sum(1 for row in rows if row[3] > 0.0)

        self.printf("\n")
        self.printf("  Final Parameters\n")
        self.printf("  ----------------\n")

        for name, value in zip(names, x):
            if name[0] == "rho":
                self.printf("    rho_%-2s = %14.8f ang\n", name[1], value * BOHR2ANG)
            elif name[0] == "sij":
                self.printf("    S_%-2s_%-2s = %14.8f\n", name[1][0], name[1][1], value)
            elif name[0] == "sij_group":
                self.printf("    S_%-3s = %14.8f\n", name[1], value)
            elif name[0] == "dij":
                self.printf("    %-3s = %14.8f\n", name[1], value)

        self.printf("\n")
        self.printf("  Final Statistics\n")
        self.printf("  ----------------\n")
        self.printf("    RMSE = %14.8f kcal/mol\n", rmse)
        self.printf("    MAE  = %14.8f kcal/mol\n", mae)
        self.printf("    MAX  = %14.8f kcal/mol\n", maxerr)
        self.printf("    Target MAE = %10.4f kcal/mol : %s\n", self.mae_tol, "PASS" if mae < self.mae_tol else "FAIL")
        self.printf("    Target RMS = %10.4f kcal/mol : %s\n", self.rms_tol, "PASS" if rmse < self.rms_tol else "FAIL")
        self.printf("    Target MAX = %10.4f kcal/mol : %s\n", self.max_tol, "PASS" if maxerr < self.max_tol else "FAIL")
        self.printf("    Positive polarization energies = %d\n", positive_pol)

        self.print_bound_report(x, names)

        self.printf("\n")
        self.printf("  Final Full Data Set Sorted by Absolute Error\n")
        self.printf("  --------------------------------------------\n")
        self.printf(
            "  %6s%8s%16s%16s%16s%16s%16s\n",
            "Rank", "Index", "Ref", "PolE", "CDS", "Calc", "Err"
        )

        for rank, row in enumerate(rows, start=1):
            _, idx, ref, pol_e, cds_e, calc, err = row
            self.printf(
                "  %6d%8d%16.6f%16.6f%16.6f%16.6f%16.6f\n",
                rank, idx, ref, pol_e, cds_e, calc, err
            )

    def print_bound_report(self, x, names):
        _, bounds, _ = self.pack_x0_and_bounds()
        lower, upper = bounds
        hits = []

        for value, lo, hi, name in zip(x, lower, upper, names):
            if np.isclose(value, lo, rtol=0.0, atol=1.0e-7):
                hits.append((name, "lower", lo))
            elif np.isclose(value, hi, rtol=0.0, atol=1.0e-7):
                hits.append((name, "upper", hi))

        self.printf("\n")
        self.printf("  Bound Check\n")
        self.printf("  -----------\n")

        if not hits:
            self.printf("    No optimized parameter is pinned at a bound.\n")
            return

        for name, side, value in hits:
            if name[0] == "rho":
                self.printf("    rho_%-2s is at %s bound: %.8f ang\n", name[1], side, value * BOHR2ANG)
            elif name[0] == "sij":
                self.printf("    S_%-2s_%-2s is at %s bound: %.8f\n", name[1][0], name[1][1], side, value)
            elif name[0] == "sij_group":
                self.printf("    S_%-3s is at %s bound: %.8f\n", name[1], side, value)
            elif name[0] == "dij":
                self.printf("    %-3s is at %s bound: %.8f\n", name[1], side, value)
    # -------------------------------------------------------------------------
    # Prints the program banner.
    # -------------------------------------------------------------------------
    def print_program_banner(self):
        self.printf("\n")
        self.printf("    ================================================================================\n")
        self.printf("                         PARAMFIT: GB SOLVATION PARAMETER FITTING\n")
        self.printf("    ================================================================================\n")
        self.printf("      Authors          : Ugur Bozkaya and Betul Ermis\n")
        self.printf("      Model            : Generalized Born / pairwise descreening\n")
        self.printf("      CDS contribution : fixed from previously computed output data\n")
        self.printf("      Revision         : April 2026\n")
        self.printf("    ================================================================================\n\n")

    def print_input_summary(self):
        self.printf("    =========================================================================================\n")
        self.printf("                                      INPUT DATA SUMMARY\n")
        self.printf("    =========================================================================================\n")
        self.printf("    Reference file              : %s\n", self.ref_file)
        self.printf("    Calculation output files    : [index]_[name].out\n")
        self.printf("    Number of data points       : %d\n", self.ndata)
        self.printf("    Quantum chemistry method    : %s\n", self.qcmethod)
        self.printf("    Atomic charge model         : %s\n", self.charge_type)
        self.printf("    Reference energy unit       : kcal/mol\n")
        self.printf("    CDS free energy treatment   : read from output files and kept fixed\n")
        self.printf("\n")
        self.printf("    Reference file format:\n")
        self.printf("       [data index] [solvation free energy in kcal/mol]\n")
        self.printf("\n")
        self.printf("    Output file requirements:\n")
        self.printf("       - files must follow [index]_[name].out\n")
        self.printf("       - indices must match reference-data ordering\n")
        self.printf("       - coordinates, charges, and CDS energies must be present\n")
        self.printf("    =========================================================================================\n\n")
    
    
    # -------------------------------------------------------------------------
    # Prints reference-data reading information.
    # -------------------------------------------------------------------------
    def print_reference_reading_info(self):
        self.printf("    =========================================================================================\n")
        self.printf("                READING REFERENCE SOLVATION FREE ENERGY DATA FROM FILE: %s ...\n", self.ref_file.upper())
        self.printf("\n")
        self.printf("    Each line must contain:\n")
        self.printf("       [data index] [solvation free energy in kcal/mol]\n")
        self.printf("\n")
        self.printf("    Example:\n")
        self.printf("\n")
        self.printf("    1  -6.31\n")
        self.printf("    2  -4.29\n")
        self.printf("    3  -0.87\n")
        self.printf("    4   1.83\n")
        self.printf("\n")
        self.printf("    Checking that the number of reference entries matches ndata.\n")
        self.printf("    Reference solvation free energies will be used as target values in the fitting procedure.\n")
        self.printf("    =========================================================================================\n\n")
    
    
    # -------------------------------------------------------------------------
    # Prints calculation-output reading information.
    # -------------------------------------------------------------------------
    def print_output_reading_info(self):
        self.printf("    =========================================================================================\n")
        self.printf("                        READING CALCULATION OUTPUTS FOR PARAMETER FITTING ...\n")
        self.printf("\n")
        self.printf("    Output files must be provided in the working directory and follow the naming convention:\n")
        self.printf("                          [index]_[name].out \n")
        self.printf("\n")
        self.printf("    Example:\n")
        self.printf("\n")
        self.printf("    1_water_mp2_qz.out  \n")
        self.printf("    2_ammonia_mp2_qz.out\n")
        self.printf("    3_benzene_mp2_qz.out\n")
        self.printf("    4_ethane_mp2_qz.out \n")
        self.printf("\n")
        self.printf("    The index must start from 1 and match the reference data ordering.\n")
        self.printf("    Each index corresponds to single data point (molecule).\n")
        self.printf("    Only MacroQC output files (.out) are supported.\n")
        self.printf("    Note: Missing or incorrectly indexed files will lead to inconsistencies in the fitting procedure.\n")
        self.printf("\n")
        self.printf("    Scanning directory and matching output files with data indices ...\n")
        self.printf("\n")
        self.printf("    Found %d matching output files. Number of output files matches ndata.\n", self.ndata)
        self.printf("    =========================================================================================\n\n")
    
    
    # -------------------------------------------------------------------------
    # Prints initial intrinsic Coulomb radii.
    # -------------------------------------------------------------------------
    def print_initial_radii(self, x0, names):
        self.printf("    =========================================================================================\n")
        self.printf("                     INITIAL INTRINSIC COULOMB RADII USED IN THE FITTING PROCEDURE           \n")
        self.printf("\n")
        self.printf("    The following intrinsic Coulomb radii (in angstrom) are used as the initial parameter \n")
        self.printf("    values for the fitting procedure:\n")
        self.printf("\n")
        self.printf("                     Element        Initial Radius (in ang)  \n")
        self.printf("                    ---------      ------------------------- \n")
    
        for name, value in zip(names, x0):
            if name[0] == "rho":
                self.printf("                        %-2s                  %6.2f \n", name[1], value * BOHR2ANG)
    
        self.printf("\n")
        self.printf("    =========================================================================================\n\n")
