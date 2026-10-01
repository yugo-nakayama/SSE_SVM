# Experiment outputs vs manuscript (`main.pdf`)

Generated from runs on this host (see `ENVIRONMENT.md`).
Raw stdout files are in this directory.

## Files

| Output file | Script | Manuscript exhibit |
|-------------|--------|--------------------|
| `table2_thm4.txt` | `spiked_svm_sim.py` | Table 2 |
| `table_gram.txt` | `spiked_svm_gram.py` | Table 3 (Gram) |
| `fig1_inconsistency.txt` | `spiked_svm_inconsistency.py` | Figure 1 + Table 4 (n-vary) |
| `table_bias.txt` | `spiked_svm_bias.py` | Table 5 (bias) |
| `table_classifiers.txt` | `spiked_svm_classifiers.py` | Table 6 + Figure 2 |

## Comparison summary

### Exact match (shared seeds with manuscript)

- **Table 2**: max \|diff\| = 0
- **Table Gram**: max \|diff\| = 0 (including ratios 3.292, 2.900, 3.021, 3.471)

### Bias (Table 5) — measured intercept \(b\)

| \((n_1,n_2,c_1,c_2)\) | paper \(b\) | run \(b\) | \|diff\| |
|----------------------|-------------|-----------|----------|
| (5,5,0.2,0.8) | −0.0899 | −0.0899 | 0.0000 |
| (4,4,0.1,0.7) | −0.0921 | −0.0894 | 0.0027 |
| (10,5,0.4,0.7) | 0.0071 | 0.0068 | 0.0003 |
| (8,4,0.3,0.65) | 0.0103 | 0.0120 | 0.0017 |
| (8,4,0.5,0.5) | 0.0557 | 0.0586 | 0.0029 |

Max \|diff\| ≈ 0.003 (order of the reported s.e.). Qualitative conclusions match.

### Inconsistency vs \(d\) (Figure 1)

Seeds differ from the Copilot draft that produced the manuscript numbers.
Max \|diff\| ≈ 0.077 (median ≈ 0.025; typical s.e. ≈ 0.02–0.025).
Qualitative pattern unchanged: spiked rates stay \(O(0.3)\), non-spiked → 0.

### Classifiers

- Table compare: max \|diff\| up to ≈ 0.09 on BC-SVM(\(\kappa\)) (high variance; s.e. ≈ 0.04).
- n-scale: BC-SVM(\(\kappa\)) remains high (≈ 0.76–0.92); BC-SVM(\(\kappa_*\))/SC drop with \(n\).

## Reproduce

```bash
cd arxiv
python3 spiked_svm_sim.py            | tee results/table2_thm4.txt
python3 spiked_svm_gram.py           | tee results/table_gram.txt
python3 spiked_svm_inconsistency.py  | tee results/fig1_inconsistency.txt
python3 spiked_svm_bias.py           | tee results/table_bias.txt
python3 spiked_svm_classifiers.py    | tee results/table_classifiers.txt
```
