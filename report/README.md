# Theory report

Complete lay-friendly theory guide for `order-reduction-sim`.

## Build

```bash
cd report
pdflatex theory.tex
pdflatex theory.tex   # second pass for TOC and cross-references
```

Output: `theory.pdf` (14 pages, hyperlinked).

Figures are pulled from `../outputs/learner2/` and `../outputs/curriculum/`.
Re-run `./scripts/run_learner2.sh` and `./scripts/run_curriculum.sh` first if
those folders are empty.

## Contents

- Mass--spring--damper derivation and $\rho(\xi)$
- Third-order plant and co-contraction coupling
- Bias law $(\tau_1+\tau_2)/\rho(\xi)$
- Learner 1 (gradient descent, S1--S5) and why it refuted the speed claim
- Learner 2 (EKF + ARD, K1--K5) with plain-language guides
- Conclusions and what not to claim in the proposal
