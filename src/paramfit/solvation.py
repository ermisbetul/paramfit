"""
solvation.py

Generalized Born / pairwise descreening solvation model.

This module contains the physics part of the calculation:
- effective Born radii alpha_i
- GB gamma matrix
- polarization free energy
"""

import math
import numpy as np

from .constants import BOHR2ANG, HARTREE2KCAL


class GBSolvationModel:
    """
    Generalized Born solvation model.

    This class computes the polarization free energy for one molecule
    using intrinsic Coulomb radii, pairwise descreening scale factors,
    and the GB distance parameter d0.
    """

    def __init__(self, solvent_eps=78.3553, gb_sij=0.75, gb_dij=3.7, gb_cij=0.0):
        self.solvent_eps = solvent_eps
        self.gb_sij = gb_sij
        self.gb_dij = gb_dij
        self.gb_cij = gb_cij

    # -------------------------------------------------------------------------
    # Lower integration limit for pairwise descreening.
    # -------------------------------------------------------------------------
    @staticmethod
    def get_lij(ai, aj, rij, sij):
        if rij + sij * aj <= ai:
            return 1.0
        elif rij - sij * aj <= ai and rij + sij * aj > ai:
            return ai
        elif rij - sij * aj >= ai:
            return rij - sij * aj

        raise RuntimeError("Unexpected case in get_lij.")

    # -------------------------------------------------------------------------
    # Upper integration limit for pairwise descreening.
    # -------------------------------------------------------------------------
    @staticmethod
    def get_uij(ai, aj, rij, sij):
        if rij + sij * aj <= ai:
            return 1.0
        elif rij + sij * aj > ai:
            return rij + sij * aj

        raise RuntimeError("Unexpected case in get_uij.")

    # -------------------------------------------------------------------------
    # Computes effective Born radii alpha_i for one molecule.
    # -------------------------------------------------------------------------
    def compute_alpha(self, symbols, rij_ang, rho, sij, optimize_sij=False):
        n = len(symbols)
        alpha = np.zeros(n)

        for i in range(n):
            rho_i = rho[symbols[i]]
            value = 1.0 / rho_i
            total = 0.0

            for j in range(n):
                if i == j:
                    continue

                rij = rij_ang[i, j] / BOHR2ANG
                rho_j = rho[symbols[j]]

                if optimize_sij:
                    sji = sij.get((symbols[j], symbols[i]), self.gb_sij)
                else:
                    sji = self.gb_sij

                lij = self.get_lij(rho_i, rho_j, rij, sji)
                uij = self.get_uij(rho_i, rho_j, rij, sji)

                total += (
                    1.0 / lij
                    - 1.0 / uij
                    + 0.25 * rij * (1.0 / uij**2 - 1.0 / lij**2)
                    + 1.0 / (2.0 * rij) * math.log(lij / uij)
                    + sji**2 * rho_j**2 / (4.0 * rij)
                    * (1.0 / lij**2 - 1.0 / uij**2)
                )

            value -= 0.5 * total
            alpha[i] = 1.0 / value

        return alpha

    @staticmethod
    def get_dij(symbol_i, symbol_j, gb_dij):
        if isinstance(gb_dij, dict):
            pair = set((symbol_i, symbol_j))
            if pair == {"C", "H"}:
                return gb_dij.get("dch", gb_dij.get("d0"))
            if pair == {"O", "H"}:
                return gb_dij.get("doh", gb_dij.get("d0"))
            if pair == {"N", "H"}:
                return gb_dij.get("dnh", gb_dij.get("d0"))
            return gb_dij.get("d0")

        return gb_dij

    # -------------------------------------------------------------------------
    # Computes GB gamma matrix for one molecule.
    # -------------------------------------------------------------------------
    def compute_gamma(self, alpha, rij_ang, symbols=None, gb_dij=None):
        n = len(alpha)
        gamma = np.zeros((n, n))

        if gb_dij is None:
            gb_dij = self.gb_dij

        for i in range(n):
            for j in range(n):
                if i == j:
                    if alpha[i] <= 0.0 or not math.isfinite(alpha[i]):
                        raise ValueError("Non-positive or non-finite effective Born radius.")
                    gamma[i, j] = 1.0 / alpha[i]
                else:
                    rij = rij_ang[i, j] / BOHR2ANG
                    dij = gb_dij
                    if symbols is not None:
                        dij = self.get_dij(symbols[i], symbols[j], gb_dij)

                    if dij <= 0.0 or alpha[i] <= 0.0 or alpha[j] <= 0.0:
                        raise ValueError("Invalid GB gamma parameters.")

                    exponent = -rij**2 / (dij * alpha[i] * alpha[j])
                    denom = rij**2 + alpha[i] * alpha[j] * (
                        math.exp(exponent) + self.gb_cij
                    )

                    if denom <= 0.0 or not math.isfinite(denom):
                        raise ValueError("Invalid GB gamma denominator.")

                    gamma[i, j] = denom ** -0.5

        return gamma

    # -------------------------------------------------------------------------
    # Computes polarization free energy for one molecule.
    # Returned energy is in kcal/mol.
    # -------------------------------------------------------------------------
    def compute_polarization_energy(
        self,
        symbols,
        charges,
        rij_ang,
        rho,
        sij,
        gb_dij=None,
        optimize_sij=False,
    ):
        if gb_dij is None:
            gb_dij = self.gb_dij

        alpha = self.compute_alpha(
            symbols=symbols,
            rij_ang=rij_ang,
            rho=rho,
            sij=sij,
            optimize_sij=optimize_sij,
        )

        gamma = self.compute_gamma(
            alpha=alpha,
            rij_ang=rij_ang,
            symbols=symbols,
            gb_dij=gb_dij,
        )

        q = charges
        gb_pol_e = np.sum(q[:, None] * q[None, :] * gamma)

        gb_pol_e *= -0.5 * (1.0 - 1.0 / self.solvent_eps)
        gb_pol_e *= HARTREE2KCAL

        return gb_pol_e
