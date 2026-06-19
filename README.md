<div align="center">

# Quantum Information Dynamics in de Sitter Space

### Figure data and reproducibility code for QED<sub>2</sub> in an expanding universe

[![arXiv](https://img.shields.io/badge/arXiv-2604.02777-b31b1b.svg)](https://arxiv.org/abs/2604.02777)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2604.02777-blue.svg)](https://doi.org/10.48550/arXiv.2604.02777)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![Reproducibility](https://img.shields.io/badge/Figures-1%E2%80%933%20reproducible-2E8B57.svg)](#reproducing-the-figures)

<br>

<img src="QED%20in%20dS/reference_figures/figure3.png" width="82%" alt="Irreversibility front and operational diagnostics in de Sitter QED2">

</div>

---

## Overview

This repository accompanies

> K. Ikeda and Y. Oz,  
> **“Quantum Information Dynamics of QED<sub>2</sub>  in Expanding de Sitter Universe,”**  
> arXiv: [2604.02777](https://arxiv.org/abs/2604.02777) [hep-th].

The project studies QED<sub>2</sub> in an expanding de Sitter background as a controlled setting where gauge dynamics, spectral flow, real-time evolution, and quantum-information diagnostics can be analyzed together. In cosmic time, the hopping term redshifts while the electric term grows with the scale factor, driving the system through a moving narrow-gap region in the $(\tau,m)$ plane. The resulting dynamics are probed through exact diagonalization, matrix-product-state calculations, and finite-temperature irreversibility diagnostics.

This repository provides the figure-level reproducibility package for the manuscript: numerical data products, plotting routines, reference figures, and validation scripts for Figures 1–3.

---

## What is included

- **Figure 1:** dynamical-transition diagnostics from exact diagonalization, including the gap landscape, loss of adiabaticity, excitation-energy density, and structure-factor response.
- **Figure 2:** fixed-mass thermodynamic and continuum extrapolation data from matrix-product-state calculations.
- **Figure 3:** finite-temperature irreversibility-front diagnostics, including relative entropy, temperature dependence, finite-size checks, and LOCC-accessible observables.
- **Validation utilities:** scripts for regenerating the figures and recording numerical checks in `generated/reproduction_checks.json`.

---

## Repository structure

```text
.
├── README.md
└── QED in dS/
    ├── data/
    │   ├── figure_1/
    │   ├── figure_2/
    │   └── figure_3/
    ├── src/
    │   ├── plot_figure1.py
    │   ├── plot_figure2.py
    │   ├── plot_figure3.py
    │   └── file_utils.py
    ├── scripts/
    │   ├── reproduce_figure1.py
    │   ├── reproduce_figure2.py
    │   ├── reproduce_figure3.py
    │   ├── check_reproduction.py
    │   ├── clean_generated.py
    │   └── update_manifest.py
    ├── reference_figures/
    ├── generated/
    ├── requirements.txt
    ├── environment.yml
    ├── MANIFEST.csv
    └── checksums.sha256
```

The `data/` directory contains the processed numerical arrays used by the plotting scripts. The `reference_figures/` directory contains the reference versions of the manuscript figures. Regenerated figures are written to `generated/`.

---

## Quick start

Clone the repository:

```bash
git clone https://github.com/IKEDAKAZUKI/Quantum-Information-Dynamics-in-de-Sitter-Space.git
cd Quantum-Information-Dynamics-in-de-Sitter-Space
cd "QED in dS"
```

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Windows:

```bash
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, with conda:

```bash
conda env create -f environment.yml
conda activate qed2-figures
```

---

## Reproducing the figures

From the `QED in dS/` directory, run:

```bash
python scripts/reproduce_figure1.py
python scripts/reproduce_figure2.py
python scripts/reproduce_figure3.py
python scripts/check_reproduction.py
```

Expected generated files:

```text
generated/
├── figure1.pdf
├── figure2.pdf
├── figure2.png
├── figure3.pdf
├── figure3.png
└── reproduction_checks.json
```

The check script verifies that the regenerated outputs exist and records the numerical diagnostics used to confirm the figure data.

---

## Figure map

| Manuscript figure | Data | Plotting script | Output |
| --- | --- | --- | --- |
| Figure 1 | `data/figure_1/transition_data.zip` | `scripts/reproduce_figure1.py` | `generated/figure1.pdf` |
| Figure 2 | `data/figure_2/fixed_mass_gap_summary.npz` | `scripts/reproduce_figure2.py` | `generated/figure2.pdf`, `generated/figure2.png` |
| Figure 3 | `data/figure_3/` | `scripts/reproduce_figure3.py` | `generated/figure3.pdf`, `generated/figure3.png` |

---

## Checksums and manifest

The package includes

```text
MANIFEST.csv
checksums.sha256
```

for tracking the distributed files. After modifying the package contents, the manifest can be refreshed with

```bash
python scripts/update_manifest.py
```

Generated outputs can be removed with

```bash
python scripts/clean_generated.py
```

---

## Citation

If you use this repository in your research, please cite the associated manuscript:

```bibtex
@article{Ikeda:2026nvc,
  title         = {Quantum Information Dynamics of QED$_2$ in Expanding de Sitter Universe},
  author        = {Ikeda, Kazuki and Oz, Yaron},
  year          = {2026},
  eprint        = {2604.02777},
  archivePrefix = {arXiv},
  primaryClass  = {hep-th},
  doi           = {10.48550/arXiv.2604.02777},
  url           = {https://arxiv.org/abs/2604.02777}
}
```

---

## Funding

The work of KI was supported by the U.S. Department of Energy, Office of Science, under Contract No. DE-SC0026415 and in part by the NSF under Grant No. OSI-2328774, in particular, on the connection between quantum fundamentals and quantum systems. The work of YO was supported by the Israeli Science Foundation Excellence Center, the US-Israel Binational Science Foundation, and the Israel Ministry of Science.

---

<div align="center">

**QED<sub>2</sub> · de Sitter dynamics · spectral flow · matrix product states · quantum irreversibility**

</div>
