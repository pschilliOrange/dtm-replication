# dtm-replication

**dtm-replication** reproduces the research code for **Denoising Thermodynamic Models (DTMs)** used in the paper An Efficient Probabilistic Hardware Architecture for Diffusion-like Models. It’s built on **thrml**, Extropic’s sampling library, and is designed to easily train a DTM while keeping hyperparameters and graph topology fully configurable. 
We believe the DTM design space—and even generally the architectures for connecting Boltzmann machines—remains largely unexplored and are excited to see how the community builds on these ideas and models.

---

## Installation

thrmlDenoising is built on top of JAX, thrml, and other standard python libraries.

After installing JAX, running

```bash
pip install -e .
```

from the root of the repository will install thrml and all other python packages needed for standard runtime, as well as thrmlDenoising itself in editable mode.

---

## Quick API example

Create a DTM with defaults and train:

```python
from thrmlDenoising.DTM import DTM

dtm = DTM()
dtm.train(n_epochs=50, evaluate_every=5)
```

Optionally override config values:

```python
from thrmlDenoising.utils import make_cfg
from thrmlDenoising.DTM import DTM

cfg = make_cfg(
    exp=dict(seed=256, descriptor="mnist_1step"),
    data=dict(dataset_name="mnist", target_classes=tuple(range(10))),
    graph=dict(graph_preset_architecture=60_12, grayscale_levels=1, base_graph_manager=1),
)

dtm = DTM(cfg)
dtm.train(n_epochs=50, evaluate_every=5)
```
See training_script.py for an example where all parameters can easily be overridden.

> Training and FID evaluation are currently setup for **MNIST** and **Fashion-MNIST**. There also exists a toy `smoke_testing` dataset in smoke_testing.py along with some very simple accuracy helpers to evaluate models trained on this dataset. 

---

## The `DTM` class

`DTM` exposes a small set of methods, such as the training method. Some are called automatically during training for the data put in the model saving and logging directory when enabled, but most can also usefully be called directly by the user after training or loading. See docstrings for details.

## Saving & logging

The DTM saves after evaluate_every epochs to the model's saving directory, which can later be loaded from to reproduce that state of the model. Because each step's training is completely independent, loading separate steps from separately trained models is also supported when configs are compatible. See exploratory_examples/01_frankenstein.ipynb for a discussion of loading diffusion steps from different training stages.

Training with `evaluate_every` creates a run folder:

model_logging_and_saving/{descriptor}_{time stamp}/

which contains, if enabled,

- a `training_log_{descriptor}.txt` file with FIDs and autocorrelations printed,
- an `images/` folder with generated images from the model after each epoch,
- a `gifs/` folder with the images animated through the gibbs samling process,
- a `fid_vs_epoch.png` and `autocorr_vs_epoch.png` plot,
- an `autocorrelation_vs_lags/` folder containing each step's autocorrelation plotted across lag time,
- and a `model_saving/` directory with a `config.yaml` and per-epoch checkpoints: `epoch_XXX/step_YY.eqx`

---

## Figures from the paper

All relevant figures are included in the `figures/` folder, where there are .py scripts that can easily be called to replicate these figures as well as csv data files and corresponding plots created from previously trained models. The four folders included contain:

- grayscale/
    - images generated with binomial nodes, similar to those shown in Figure 5a of the paper
- MEBM_DTM_ACP_ablation/
    - performance and autocorrelation plots comparing an MEBM, DTM, and DTM with ACP, as shown in Figure 5b of the paper
- degree_ablation/
    - plots for the degree and visible-node fraction ablation, corresponding to the upper half of Figure 5c in the paper
- chain_depth_ablation/
    - plots for the chain depth and visible-node fraction ablation, corresponding to the lower half of Figure 5c in the paper

---

## How this codebase differs from the original research code

The biggest algorithmic change in this codebase is the **flexible graph manager**, which is relevant when we are generating images with multiple grayscale levels. In the paper’s code, image pixels were binomial nodes whose *n* sub‑trials' parameters were tied: they had the same neighbors, the same edge weights, and the same biases. In our **Poisson‑binomial graph manager**, sub‑trials are **not tied**—they are separate Ising nodes that we treat as composing a pixel. This allows different connectivity and weights per sub‑trial, and we think it performs slightly better because the model is able to produce arbitrarily sharp peaks at any grayscale level when not forced to a binomial distribution. We call this separated ising node representation poisson binomial nodes, despite the trials not being independent. 

But the best grayscale ising encoding is still an open research question, as is the best graph topology of the base graph. The Poisson-binomial encoding may work well at low grayscale levels, but the ising nodes required to encode a pixel grows linearly with n_grayscale_levels which means the state space of pixel nodes (proporitional to 2^n_grayscale_levels) dwarfs the state space of valid pixel encodings (directly proportional to n_grayscale_levels) for large n. This is not desirable both because the model is forced to 'waste' parameters on remembering which states are valid, and because it just forces there to be many image nodes in the graph. The abstract base graph manager makes it easy to experiment with different encodings and graph topologies. See exploratory_examples/02_convolved_graphs.ipynb for an example construction of a base graph manager which tries to hardwire the local correlation image pixels have with each other.

---

## Acknowledgements

I would like to thank Andraz Jelincic for meeting my many questions with nothing but patience, Julian Zucker for the consistent check ins, and the team who wrote thrml for making the API very easy to work with.
