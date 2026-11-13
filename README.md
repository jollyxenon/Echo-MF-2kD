# Echo-MF-2kD: Training-Free Mean-Field Adjoint Manifold Initialization for Constrained Variational Quantum Optimization

Official open-source implementation of **`Echo-MF-2kD`** and **`Echo-Flat-2kD`** (*Training-free mean-field adjoint manifold initialization for constrained variational quantum optimization*).

This repository provides the complete codebase for:
1. **Exact $2k$-Dimensional Spin-Echo Parity Decoupling & Quantum–Classical Product-Energy Identity**:
   - Exact cancellation of quadratic spin-spin couplings $H_J$ across the entire $2k$-dimensional complete spin-echo pair submanifold $(\boldsymbol{g},\boldsymbol{\theta})\in\mathbb{R}^{2k}$ within the unmodified standard $p$-layer global QAOA circuit ansatz at positive even depth $p=2k$.
   - Exact algebraic equivalence between the multi-body quantum expected energy and the classical product-state energy on the $2k$-dimensional spin-echo submanifold:
     $$\langle\psi_p(\boldsymbol{g},\boldsymbol{\theta})|H_C|\psi_p(\boldsymbol{g},\boldsymbol{\theta})\rangle \equiv E_{\mathrm{prod}}(\boldsymbol{g},\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{m}^\top J\boldsymbol{m} + \boldsymbol{h}^\top\boldsymbol{m} + C$$
     with $O(kn)$ reverse-mode $\mathrm{SU}(2)$-adjoint gradient evaluation (`src/echo/multiscalar.py`).
2. **Training-Free Mean-Field Adjoint Manifold Initialization (`Echo-MF-2kD`)**:
   - Requires **zero training data, zero data leakage, and zero online quantum selection measurements**, accepting only problem coefficients $(J,h)$.
   - Combines 80-step damped mean-field inference ($\beta_{\mathrm{mf}}^\star=2.0$), deterministic temperature-continuation cluster orientation ($\beta\in\{2.5,6.0,16.0\}$), 1D Daubechies flat-top convex-interval root anchoring (Yao, 2026), and staged $O(kn)$ $\mathrm{SU}(2)$-adjoint L-BFGS-B refinement within a non-degenerate mixer bound $|\theta_j|\le 1.40$ (jointly optimizing flip-sensitivity-weighted log-likelihood, local effective-field inversion guards, and classical product energy $E_{\mathrm{prod}}$), followed by Stage-2 full-angle L-BFGS-B energy recoupling.
   - Across 1,024 held-out MWIS center test instances ($n=10, p\in\{8,10,12\}$) and 5,120 cross-scale test instances ($n\in\{6,8,10,12,14\}$), `Echo-MF-2kD` doubles final ground-state probability ($46.16\%$ vs. $22.42\%$ for 32-evaluation online TQA) while cutting Stage-2 energy–gradient evaluations by $45.2\%$ and achieving $100\%$ Stage-1 shot-noise immunity.

---

## Installation

Dependencies are managed via [`pixi`](https://pixi.sh/):

```bash
pixi install
pixi install -e learning
pixi install -e figures
```

---

## Quick Start

### 1. Training-Free Online Single-Instance Solving (Input $(J,h)$ Only)

```bash
# Run training-free 2kD adjoint manifold initialization (Echo-MF-2kD) + Stage-2 full-angle L-BFGS-B energy optimization (0 Stage-1 quantum calls)
pixi run -e learning predict-learning --instance data/benchmark/mwis/mwis_n08_00.npz --mf-method Echo-MF-2kD --depth 8

# Run training-free 1D mean-field overlap initialization (Echo-MF-Flat / Echo-MF-End / Echo-MF-Alt)
pixi run -e learning predict-learning --instance data/benchmark/mwis/mwis_n08_00.npz --mf-method Echo-MF-Flat --depth 8
```

### 2. Batch Evaluation of Training-Free & External Baselines

Evaluate `Echo-MF-2kD`, `Echo-MF-Flat`, `Echo-MF-End`, `Echo-MF-Alt`, `Echo-Energy`, `TQA`, `RI (1)`, `RI (32)`, `Classical-MF`, `Classical-Cluster`, `Warm-Start QAOA`, and `ma-QAOA`:

```bash
pixi run -e learning evaluate-baselines --dataset data/mwis/n10/dataset.npz --depth 10 --method Echo-MF-2kD --split test --output runs/baselines/mf_2kd_p10
pixi run -e learning evaluate-baselines --dataset data/mwis/n10/dataset.npz --depth 10 --method Echo-MF-Flat --split test --output runs/baselines/mf_flat_p10
pixi run -e learning evaluate-baselines --dataset data/mwis/n10/dataset.npz --depth 10 --method TQA --split test --output runs/baselines/tqa_p10
pixi run -e learning evaluate-baselines --dataset data/mwis/n10/dataset.npz --depth 10 --method "Warm-Start QAOA" --split test --output runs/baselines/ws_p10
```

### 3. Canonical Experiments & Manuscript Figures

```bash
# Run canonical multi-scalar scaling, ablation, multi-family, and finite-shot noise benchmarks
pixi run -e learning canonical-experiments --task all

# Generate all Paper 2 publication figures and Source Data CSVs
pixi run -e figures paper2-figures
```

---

## Manuscript & Repository Structure

- [`paper/paper2_preprint.pdf`](paper/paper2_preprint.pdf) / [`paper/paper2_preprint.tex`](paper/paper2_preprint.tex) / [`paper/paper2-ncs-en.md`](paper/paper2-ncs-en.md) / [`paper/paper2-ncs-zh.md`](paper/paper2-ncs-zh.md): Preprint LaTeX/PDF manuscript (`sn-ncs.sty`, `convert_md_to_ncs_tex.py`), full bilingual Markdown manuscripts, vector figures and source data CSVs (`paper/figures/`, `plot_fig1_pipeline.py`, `generate_paper2_figures.py`), and verified BibTeX (`paper/references.bib`).
- `src/echo/multiscalar.py`: Core $2k$D $\mathrm{SU}(2)$-adjoint engine, temperature-continuation cluster orientation (`resolve_prior_direction`), and staged L-BFGS-B manifold refinement (`Echo-Flat-2kD` / `Echo-MF-2kD`).
- `src/echo/`: 1D Daubechies flat-top synthesis and root anchoring (`core.py`, `flat_synthesis.py`, `flat_scan.py`, `variants.py`).
- `src/learning/`: Mean-field solvers, decoders, online baseline suite, and Stage-2 energy optimization (`features.py`, `echo_decoder.py`, `baselines.py`, `optimization.py`).
- `src/instances/`: Five-family Ising and cross-scale MWIS benchmark generators (`benchmark.py`, `mwis.py`).

---

## Citation

```bibtex
@article{yao2026echomf2kd,
  title   = {Training-free mean-field adjoint manifold initialization for constrained variational quantum optimization},
  author  = {Yao, Jianxuan},
  journal = {Research Square preprint},
  year    = {2026}
}

@article{yao2026echo,
  title   = {{Echo-QAOA}: Constructive Spin-Echo Response Design and Per-Qubit Marginal Distillation for Variational Quantum Optimization},
  author  = {Yao, Jianxuan},
  journal = {Research Square preprint rs-11707296/v1},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-11707296/v1},
  url     = {https://www.researchsquare.com/article/rs-11707296/v1}
}
```
