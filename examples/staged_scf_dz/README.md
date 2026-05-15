# Staged SCF/DZ PARAMFIT Example

This directory contains a full runnable staged PARAMFIT example, including the
MacroQC output files required by the fitting code.

The stages are organized as:

```text
1_CHO/
2_CHON/
3_CHONS/
4_CHONSP/
5_CHONSP_hal/
```

Run each stage from inside its own directory. For example:

```bash
cd examples/staged_scf_dz/1_CHO
paramfit -i cho_explicit_maemax.inp
```

For later stages, the input files may load a parameter file produced by an
earlier stage through `initial_parameter_file`. If you want to reproduce the
staged workflow from scratch, run the stages in numerical order.

This example is intentionally complete rather than minimal. It includes the
reference files, input files, MacroQC output files, PARAMFIT output files, and
generated parameter files used during development.
