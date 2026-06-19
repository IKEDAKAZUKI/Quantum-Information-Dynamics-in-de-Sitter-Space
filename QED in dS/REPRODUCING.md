# Reproducing the figures

## Python environment

```bash
python -m pip install -r requirements.txt
```

A Conda environment can also be created with:

```bash
conda env create -f environment.yml
conda activate qed2-figures
```

## Regenerate the figures

```bash
python scripts/reproduce_figure1.py
python scripts/reproduce_figure2.py
python scripts/reproduce_figure3.py
python scripts/check_reproduction.py
```

## Regenerate one figure

```bash
python scripts/reproduce_figure1.py
python scripts/reproduce_figure2.py
python scripts/reproduce_figure3.py
```

## Thermodynamic-limit Figure 2 workflow

A source archive, extracted run directory, or summary `.npz` file can be passed directly to the thermodynamic-limit plotting script:

```bash
python scripts/reproduce_figure2_thermodynamic_limit.py /path/to/source_archive.zip
```

The default outputs are:

```text
generated/figure2_thermodynamic_limit.pdf
generated/figure2_thermodynamic_limit.png
```

## Check generated files

```bash
python scripts/check_reproduction.py
```

The check script writes `generated/reproduction_checks.json`.
