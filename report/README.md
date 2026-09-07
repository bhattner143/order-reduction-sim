# Theory report

Complete lay-friendly theory guide for `order-reduction-sim`.

## Build

```bash
cd report
pdflatex theory.tex
pdflatex theory.tex   # second pass for TOC and cross-references
```

Output: `theory.pdf` (hyperlinked).

Figures are pulled from `../outputs/learner2/`, `../outputs/curriculum/`
and `../outputs/learner4/`. Re-run the matching `./scripts/run_*.sh` first
if those folders are empty.

Learner~4 one-trial block diagram (standalone landscape PDF):

```bash
./scripts/build_learner4_block_diagram.sh
# -> outputs/learner4/learner4_block_diagram.pdf
```

## Contents

- Mass--spring--damper derivation and $\rho(\xi)$
- Third-order plant and co-contraction coupling
- Bias law $(\tau_1+\tau_2)/\rho(\xi)$
- Learner 1 (gradient descent, S1--S5) and why it refuted the speed claim
- Learner 2 (EKF + ARD, K1--K5) with plain-language guides
- Learner 4 (slow-only identification; the first speed result)
- Conclusions and what not to claim in the proposal
