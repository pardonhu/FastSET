# Paper information

**FastSET: Fast Sequential Retraining for Online Adaptation of Neural Receivers
via Physics-Guided Decomposition**

Shuo Wang, Facheng Hu, Hongzi Zhu, Yibo Pi, and Xudong Wang.
Shuo Wang and Facheng Hu contributed equally. Xudong Wang is the corresponding author.

The repository accompanies the journal manuscript. A final journal volume, issue,
and DOI will be added with the publication record. The manuscript extends the
FastSET conference study with estimator-stage stopping, a 40-split evaluation,
training-strategy analysis, additional backbones, and controlled TDL/urban RT
experiments. The implementation and experimental artifacts are indexed in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Main method

The estimator maps the received grid and pilots to a channel estimate, using a
learned residual over LS. The demapper combines the estimated channel and received
grid to produce coded-bit likelihoods. Online adaptation updates the estimator
first, freezes its weights, and updates the demapper when the BER target requires
further training. Offline PEBS selection measures performance and adaptation
computation time under the E2E model's parameter budget.

## Software citation

Use GitHub's “Cite this repository” action or the versioned `CITATION.cff`.
The release tag identifies the code and artifacts used in an experiment.
