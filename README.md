# Spiked SVM under HDLSS — arXiv manuscript

## Build the PDF

```bash
cd arxiv
make          # runs pdflatex twice → main.pdf
```

Requirements: TeX Live with `amsart`, `amsmath`, `amssymb`, `amsthm`,
`mathtools`, `booktabs`, `graphicx`, `hyperref`, `natbib`, `enumitem`, `geometry`.

## Reproduce the numerical experiments

Requires Python 3 with NumPy and SciPy. Each experiment in Section 11 names its
script in the text.

| Section | Command | Exhibit |
|---------|---------|---------|
| 11.1 | `python spiked_svm_sim.py` | Table 2 |
| 11.2 | `python spiked_svm_gram.py` | Table 3 (Gram) |
| 11.3 | `python spiked_svm_inconsistency.py` | Figure 1, Table 4 |
| 11.4 | `python spiked_svm_bias.py` | Table 5 (bias) |
| 11.5 | `python spiked_svm_classifiers.py` | Table 6, Figure 2 |

Shared code: `spiked_svm_common.py`.

Saved stdout from a full run on this machine, plus a comparison to the
manuscript tables and host specs:

- `results/` — raw outputs (`table2_thm4.txt`, …)
- `results/COMPARISON.md` — diffs vs `main.pdf`
- `results/ENVIRONMENT.md` — CPU / RAM / OS / Python versions

```bash
make sim        # Table 2 only
make sim-all    # all five experiments (can take a long time)
```

## Files

| Path | Role |
|------|------|
| `main.tex` | Master file |
| `sections/*.tex` | §§1–12 and Appendix A |
| `figures/` | Figures for §§11.3 and 11.5 |
| `spiked_svm_*.py` | Simulation scripts |
| `Makefile` | `make` / `make sim` / `make sim-all` |

## arXiv upload

Upload `main.tex`, `sections/`, `figures/`, and all `spiked_svm_*.py` files.
Do **not** upload `../not_for_submission/`.
