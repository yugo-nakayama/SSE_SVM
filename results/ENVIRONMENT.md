# Computational environment

## Hardware

| Item | Value |
|------|-------|
| CPU | Intel Core i7-14700 |
| Architecture | x86_64 |
| Logical CPUs | 28 (`nproc`) |
| CPU frequency | 800 MHz – 5400 MHz |
| Memory | 125 GiB RAM (≈126 GiB total), 2.0 GiB swap |
| GPU | NVIDIA GeForce RTX 4090 (not used by the CPU-only NumPy/SciPy scripts) |
| Disk | NVMe (`/dev/nvme0n1p2`), 3.6 T total, ≈2.5 T free on `/` |

## Software

| Item | Value |
|------|-------|
| OS | Ubuntu 22.04.5 LTS (Jammy) |
| Kernel | Linux 6.8.0-138-generic |
| Python | 3.13.12 |
| NumPy | 2.2.6 |
| SciPy | 1.17.1 |
| Platform tag | `Linux-6.8.0-138-generic-x86_64-with-glibc2.35` |

## Notes

- Simulations are CPU-bound (dual SLSQP + dense linear algebra). The RTX 4090 is present but unused.
- Approximate wall times on this machine (full paper settings):
  - `spiked_svm_sim.py` ≈ 14 s
  - `spiked_svm_gram.py` ≈ 10 s
  - `spiked_svm_bias.py` ≈ 3 s
  - `spiked_svm_inconsistency.py` ≈ 36 s
  - `spiked_svm_classifiers.py` ≈ 193 s
