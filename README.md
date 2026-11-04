# Three-layer fairness assessment: code

Code for a paper under anonymous review. It compares inequality verdicts on pairs of policies at three points of an evaluation pipeline:

| Layer | Pipeline | Sample |
|---|---|---|
| 1 | burdens b -> metric -> verdict | synthetic burden vectors |
| 2 | outcomes y -> impact function -> b -> metric | synthetic outcome shares |
| 3 | policy x -> digital twin -> y -> impact function -> b -> metric | policies evaluated by a city-scale digital twin |

## Files

| File | Role |
|---|---|
| `sampling.py` | Samples of burdens (Layer 1), outcomes (Layer 2), policies and the epistemic parameter c50 (Layer 3) |
| `impact_function.py` | `b_k = c_k + w_mode m_k + w_time (a_k + p_k)` |
| `metrics.py` | Gini, Theil T and the Atkinson index for burdens (`bad_atkinson`, with beta = -epsilon), raw and normalized |
| `lorenz.py` | Lorenz descriptors Z (`zheng`) and Gamma (`LDI`) |
| `visualize.py` | `pairwise_disagreement` and exploratory figures |
| `utils.py` | Shared helpers |
| `results/m1_reference/m1_reference.py` | M1 reference table and figures, from the samples |
| `results/m1_reference/check_layer3_denominator.py` | Numeric check of the Layer 3 denominator; needs the digital twin |
| `generate_m*.py`, `plot_m*.py` | Tables and figures of the other measures |
| `results/seed42_K52_groups.csv` | The 52 groups and their population weights p |

In the code, `w_mode` and `w_time` are the paper's phi_mode and phi_time, and `p` is the paper's population weights w.

## Setup

Run with Python 3.10.11, numpy 1.26.4, pandas 2.3.0, scipy 1.15.3, pyarrow 24.0.0 and matplotlib 3.10.0.

```bash
pip install numpy scipy pandas pyarrow psutil tqdm matplotlib
```

## Regenerating the samples of Layers 1 and 2

```bash
P=$(python -c "import csv; print(' '.join(r['p'] for r in csv.DictReader(open('results/seed42_K52_groups.csv'))))")
python sampling.py b --K 52 --n-samples 250000 --seed 42 --p $P
python sampling.py y --K 52 --n-samples 250000 --seed 42 --p $P
python lorenz.py classification results/seed42_K52_n250000_b.parquet
python metrics.py compute results/seed42_K52_n250000_b.parquet
```

Each script explains its options with `--help`.

## What is not included

- **The digital twin and its outputs.** The twin's code is private, and its Layer 3 output files take about 8 GB. Layer 3 results cannot be regenerated from this repository alone. The scripts that read Layer 3 files expect them in `results/layer3/`.
- **Sample and result files.** They are too large or can be regenerated. Scripts that read `results/` need those files first. `results/m1/*.csv`, which some `generate_*` and `plot_*` scripts read, are not produced by a script in this repository.

## Layer 3 denominator

In Layer 3, outcome shares are relative to each group's whole-day baseline trips. The docstring of `impact_function.py` mentions baseline trips in the time window, which does not apply to Layer 3.
