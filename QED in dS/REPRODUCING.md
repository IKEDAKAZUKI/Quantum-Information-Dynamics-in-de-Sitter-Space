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

## Check generated files

```bash
python scripts/check_reproduction.py
```

The check script writes `generated/reproduction_checks.json`.
