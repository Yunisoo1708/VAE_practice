# Mini La-Proteina: a latent-representation coding exercise

A small, CPU-only exercise based on **La-Proteina: Atomistic Protein Generation via
Partially Latent Flow Matching** (Geffner, Didi et al., ICLR 2026, arXiv:2507.09466).

You build a toy version of the paper's pipeline: a VAE that compresses residue type and
variable-size side chains into a fixed per-residue latent, an analysis of that latent
space, and a flow-matching model with separate times for CA coordinates and latents.

| File | Purpose |
|------|---------|
| `exercise.py` | Fill in the 8 `TODO` blocks (each has a hint) |
| `solution.py` | Complete reference solution (~4 min on CPU) |

## Run

```bash
pip install torch numpy matplotlib
python3 exercise.py   # raises NotImplementedError until you fill in the TODOs
python3 solution.py   # writes latent_pca.png and latent_locality.png
```

## Parts
1. Toy protein data with variable-size side chains
2. VAE: reparameterization, KL, masked reconstruction loss
3. Latent analysis: type clustering and per-residue locality (paper Fig. 8)
4. Partially latent flow matching with two interpolation times (Eq. 5-6)
5. Sampling with independent CA / latent schedules

Reflection questions are at the end of `solution.py`.
