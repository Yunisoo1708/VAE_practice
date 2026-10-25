# %% [markdown]
# # Mini La-Proteina: per-residue latent representations (toy version)
#
# Paper: "La-Proteina: Atomistic Protein Generation via Partially Latent Flow Matching"
# (Geffner, Didi et al., ICLR 2026, arXiv:2507.09466).
#
# THE BIG IDEA OF THE PAPER
# -------------------------
# A real protein residue has a *variable* number of side-chain atoms and a *categorical*
# residue type. That mixed discrete/continuous, variable-size object is awkward to model
# with flow matching. La-Proteina's trick:
#
#   * keep the alpha-carbon (CA) backbone explicit            -> x_CA  in R^{L x 3}
#   * squeeze residue type + ALL other atoms into a
#     fixed-size continuous vector per residue (the latent)   -> z     in R^{L x 8}
#   * a VAE learns the mapping  protein <-> (x_CA, z)
#   * a flow-matching model then generates (x_CA, z) jointly, with TWO separate times
#     (t_x for CA, t_z for latents) so each can use its own integration schedule.
#
# WHAT YOU WILL BUILD (all on CPU, ~2-4 minutes total)
# ----------------------------------------------------
#   Part 1  Toy "proteins" with variable-size side chains         (data)
#   Part 2  VAE: encoder q(z|x), decoder p(s, x_sidechain | x_CA, z)  (Eq. 2-3)
#   Part 3  Latent-space analysis: clustering + locality          (paper Fig. 8)
#   Part 4  Partially latent flow matching with two times         (Eq. 5-6)
#   Part 5  Sampling and the "faster CA schedule" ablation        (Sec. 3.4)
#
# The toy world is a drastic simplification (5 residue types, <=4 side-chain atoms,
# no rotations/frames, no pair-biased attention) but keeps the *structure* of the method.

# %%
# ---------------------------------------------------------------------------
# Imports and global constants
# ---------------------------------------------------------------------------
# torch      : tensors + autograd + neural-network layers
# math/numpy : small numeric helpers
# matplotlib : saves figures next to this file (no GUI needed; we use the Agg backend)
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# K      : number of residue types (real proteins: 20)
# A      : max side-chain atoms per residue. Plays the role of "Atom37 minus the CA":
#          a FIXED-size padded slot array; a mask says which slots are real.
# L      : protein length (real: 100-800; we use 16)
# D_Z    : latent size per residue. The paper uses 8, and so do we.
K, A, L, D_Z = 5, 4, 16, 8
CA_BOND = 3.8   # consecutive CA atoms in real proteins are ~3.8 Angstrom apart
DEVICE = "cpu"


# %% [markdown]
# ## Part 1 - Toy protein data
#
# Each residue type k owns a fixed side-chain "template": n_atoms[k] points around the CA.
# Type 0 has no side chain (like Glycine), type 4 has four atoms. On top of that, every
# residue draws a discrete "rotamer" angle chi and rotates its side chain about the z-axis.
#
# So a residue is described by TWO hidden factors:  (type, chi).
# A good latent z must carry both, and the decoder must turn them back into atoms.
# This mirrors the paper: z has to encode amino-acid identity AND side-chain conformation.

# %%
# Number of real side-chain atoms for each residue type (varying dimensionality!).
N_ATOMS = torch.tensor([0, 1, 2, 3, 4])

# Fixed template offsets from the CA, shape [K, A, 3]. Generated once with a private
# generator so every run (and every student) sees the same toy "chemistry".
_g = torch.Generator().manual_seed(123)
TEMPLATES = torch.randn(K, A, 3, generator=_g) * 1.2

# Discrete rotamer angles (3 allowed conformations).
CHI_CHOICES = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3])

# Slot mask per type: SLOT_MASK[k, a] = 1 if type k really has atom slot a.
SLOT_MASK = (torch.arange(A)[None, :] < N_ATOMS[:, None]).float()   # [K, A]


def rot_z(angle):
    """Rotation matrices about the z-axis. angle: [...] -> [..., 3, 3]."""
    c, s = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    row0 = torch.stack([c, -s, zero], -1)
    row1 = torch.stack([s, c, zero], -1)
    row2 = torch.stack([zero, zero, one], -1)
    return torch.stack([row0, row1, row2], -2)


def make_batch(B):
    """Sample B toy proteins.

    Returns a dict with
      ca   [B, L, 3]    CA coordinates (random-walk chain, centred)
      s    [B, L]       residue type (long)
      sc   [B, L, A, 3] side-chain atom offsets relative to the CA, zero-padded
      mask [B, L, A]    1 where the slot holds a real atom
      chi  [B, L]       index of the rotamer (kept only for analysis)

    Job of this function: be the "PDB". Everything downstream sees only these tensors.
    """
    # Backbone: random unit directions scaled to 3.8 A, cumulative-summed into a chain.
    steps = F.normalize(torch.randn(B, L, 3), dim=-1) * CA_BOND
    ca = torch.cumsum(steps, dim=1)
    ca = ca - ca.mean(dim=1, keepdim=True)          # remove global translation

    s = torch.randint(0, K, (B, L))                 # residue types
    chi_idx = torch.randint(0, len(CHI_CHOICES), (B, L))
    chi = CHI_CHOICES[chi_idx] + 0.05 * torch.randn(B, L)   # small conformer jitter

    tmpl = TEMPLATES[s]                             # [B, L, A, 3]  per-residue template
    R = rot_z(chi)                                  # [B, L, 3, 3]
    sc = torch.einsum("blij,blaj->blai", R, tmpl)   # rotate every atom of every residue
    sc = sc + 0.02 * torch.randn_like(sc)           # coordinate noise
    mask = SLOT_MASK[s]                             # [B, L, A]
    sc = sc * mask[..., None]                       # zero out padding slots
    return dict(ca=ca, s=s, sc=sc, mask=mask, chi=chi_idx)


b = make_batch(2)
print({k: tuple(v.shape) for k, v in b.items()})


# %% [markdown]
# ## Part 2 - The VAE (paper Sec. 3.2.1)
#
# Encoder  q_psi(z | x_CA, x_notCA, s):  full protein -> mean & log-scale of a Gaussian per residue.
# Decoder  p_phi(x_notCA, s | x_CA, z):  (CA, z)      -> sequence logits + side-chain coordinates.
#
# Loss (Eq. 3, the beta-weighted ELBO):
#     CE(sequence)  +  masked squared-L2(side chain)  +  beta * KL(q || N(0, I))
# The paper uses beta = 1e-4: tiny, so latents carry lots of information (near-lossless
# reconstruction) while still being roughly standard-normal, which is what makes them
# easy for a flow model to learn afterwards.

# %%
class ResidueTransformer(nn.Module):
    """Shared backbone for encoder / decoder / denoiser.

    The paper uses one transformer family (pair-biased attention) for all three networks.
    We use a plain nn.TransformerEncoder: per-residue input features -> embeddings,
    attention across residues lets the network see context, per-residue output head.

    Job: map [B, L, d_in] -> [B, L, d_out], letting residues exchange information.
    """

    def __init__(self, d_in, d_out, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)
        # Learned position embedding: attention alone is permutation-equivariant, so without
        # this the network cannot know residue i is bonded to i+1 (essential for the chain).
        self.pos = nn.Parameter(0.02 * torch.randn(L, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.body = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, x):
        return self.out(self.body(self.inp(x) + self.pos))


class Encoder(nn.Module):
    """q_psi(z | protein): returns mean and log-variance of a factorised Gaussian.

    Input features per residue = [one-hot type (K), side-chain coords (A*3), CA (3)].
    We hand it the CA too, because the paper's encoder sees the whole protein.
    """

    def __init__(self):
        super().__init__()
        self.net = ResidueTransformer(K + A * 3 + 3, 2 * D_Z)

    def forward(self, ca, s, sc):
        B = ca.shape[0]
        feats = torch.cat([F.one_hot(s, K).float(), sc.reshape(B, L, A * 3), ca], dim=-1)
        mu, logvar = self.net(feats).chunk(2, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    """p_phi(s, x_sidechain | x_CA, z).

    Output per residue = K sequence logits + A*3 side-chain offsets.
    Always emits A slots (like Atom37) -- which slots are *real* is decided by the residue
    type: the ground-truth type during training, the decoded argmax type at inference.
    """

    def __init__(self):
        super().__init__()
        self.net = ResidueTransformer(D_Z + 3, K + A * 3)

    def forward(self, ca, z):
        out = self.net(torch.cat([z, ca], dim=-1))
        logits = out[..., :K]
        sc = out[..., K:].reshape(ca.shape[0], L, A, 3)
        return logits, sc


# %%
def reparameterize(mu, logvar):
    """Sample z ~ N(mu, exp(logvar)) in a differentiable way.

    z = mu + sigma * eps with eps ~ N(0, I). Gradients flow into mu and logvar while the
    randomness lives only in eps.  (The "reparameterisation trick".)
    """
    # TODO: std = exp(0.5*logvar); eps = randn_like(std); return mu + std*eps
    raise NotImplementedError("fill in this block")


def kl_to_standard_normal(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, I) ), averaged over batch & residues, summed over latent dims.

    Closed form per dimension:  0.5 * (mu^2 + sigma^2 - 1 - log sigma^2).
    """
    # TODO: 0.5*(mu^2 + exp(logvar) - 1 - logvar), sum over last dim, mean over rest
    raise NotImplementedError("fill in this block")


def vae_loss(enc, dec, batch, beta):
    """Beta-ELBO (Eq. 3) as a loss to MINIMISE.  Returns total and the three parts.

    Key detail from the paper: the side-chain squared error is only supervised on atoms
    that exist for the GROUND-TRUTH residue type (the mask), because padding slots have
    no defined position.
    """
    ca, s, sc, mask = batch["ca"], batch["s"], batch["sc"], batch["mask"]
    mu, logvar = enc(ca, s, sc)
    z = reparameterize(mu, logvar)
    logits, sc_hat = dec(ca, z)

    # TODO: ce = cross_entropy over residues; mse = masked mean sq error over real atoms
    raise NotImplementedError("fill in this block")
    kl = kl_to_standard_normal(mu, logvar)
    return ce + mse + beta * kl, ce, mse, kl


# %%
# Train the VAE. Fresh random proteins every step = effectively an infinite dataset.
enc, dec = Encoder().to(DEVICE), Decoder().to(DEVICE)
opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=2e-3)
BETA = 1e-3   # paper: 1e-4; slightly larger here because our model/data are tiny

for step in range(1, 2501):
    loss, ce, mse, kl = vae_loss(enc, dec, make_batch(64), BETA)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 500 == 0:
        print(f"[VAE] step {step:4d} loss {loss.item():.3f} CE {ce.item():.4f} "
              f"MSE {mse.item():.4f} KL {kl.item():.2f}")


# %%
@torch.no_grad()
def reconstruct(batch):
    """Encode with the posterior MEAN (no sampling noise), decode, and score.

    Mirrors the paper's evaluation: decoded sequence = argmax of logits, decoded atoms =
    the mean of the Gaussian decoder. Returns (sequence recovery, side-chain RMSD).
    """
    mu, _ = enc(batch["ca"], batch["s"], batch["sc"])
    logits, sc_hat = dec(batch["ca"], mu)
    seq_acc = (logits.argmax(-1) == batch["s"]).float().mean().item()
    sq = ((sc_hat - batch["sc"]) ** 2).sum(-1) * batch["mask"]
    rmsd = (sq.sum() / batch["mask"].sum()).sqrt().item()
    return seq_acc, rmsd


enc.eval(); dec.eval()
test = make_batch(256)
acc, rmsd = reconstruct(test)
print(f"Reconstruction: sequence recovery {acc:.3f}, side-chain RMSD {rmsd:.3f} A")
# Paper (real proteins): recovery 1.0, all-atom RMSD ~0.12 A.


# %% [markdown]
# ## Part 3 - Analysing the latent space (paper Fig. 8)
#
# The paper makes two claims about z:
#   (a) it is *organised*: residues of the same / similar type cluster together;
#   (b) it is *local*: z_i mostly describes residue i, even though the transformers could
#       spread information across all residues.
# We test both.

# %%
@torch.no_grad()
def encode_mean(batch):
    mu, _ = enc(batch["ca"], batch["s"], batch["sc"])
    return mu                                                   # [B, L, D_Z]


# (a) Clustering.  Project latents to 2-D with PCA (the paper uses t-SNE; PCA is
# deterministic and needs no extra library) and colour by residue type.
mu_all = encode_mean(test).reshape(-1, D_Z)
types_all = test["s"].reshape(-1)

centered = mu_all - mu_all.mean(0, keepdim=True)
_, _, Vh = torch.linalg.svd(centered, full_matrices=False)     # rows of Vh = principal axes
proj = (centered @ Vh[:2].T).numpy()

plt.figure(figsize=(5, 4))
for k in range(K):
    sel = (types_all == k).numpy()
    plt.scatter(proj[sel, 0], proj[sel, 1], s=4, label=f"type {k} ({int(N_ATOMS[k])} atoms)")
plt.legend(markerscale=3, fontsize=7)
plt.title("Latent means, PCA (colour = residue type)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_pca.png"), dpi=120)
plt.close()

# A numeric version of "do types cluster?": nearest-centroid classification in latent space.
# If z is organised by type, assigning each latent to the closest per-type mean should work.
half = len(types_all) // 2
cent = torch.stack([mu_all[:half][types_all[:half] == k].mean(0) for k in range(K)])  # [K, D_Z]
pred = torch.cdist(mu_all[half:], cent).argmin(-1)
print(f"Nearest-centroid residue-type accuracy from z alone: "
      f"{(pred == types_all[half:]).float().mean().item():.3f}  (chance = {1 / K:.2f})")


# %%
# (b) Locality via perturbation.  Encode a protein, add noise of size sigma to the latent
# of ONE residue j, decode, and measure the reconstruction error at residue j versus the
# average over all other residues. Locality => error at j grows, the others stay ~flat.
@torch.no_grad()
def perturbation_test(sigmas, n_batches=8, j=L // 2):
    rows = []
    for sigma in sigmas:
        err_j, err_other, seq_j, seq_other = [], [], [], []
        for _ in range(n_batches):
            bt = make_batch(64)
            mu = encode_mean(bt)
            mu_p = mu.clone()
            # TODO: add sigma * randn noise to mu_p[:, j, :] only (one residue)
            raise NotImplementedError("fill in this block")
            logits, sc_hat = dec(bt["ca"], mu_p)
            # per-residue side-chain squared error (masked) and per-residue sequence error
            sq = (((sc_hat - bt["sc"]) ** 2).sum(-1) * bt["mask"]).sum(-1)   # [B, L]
            wrong = (logits.argmax(-1) != bt["s"]).float()                    # [B, L]
            others = torch.arange(L) != j
            err_j.append(sq[:, j].mean().item());     err_other.append(sq[:, others].mean().item())
            seq_j.append(wrong[:, j].mean().item());  seq_other.append(wrong[:, others].mean().item())
        rows.append((sigma, np.mean(err_j), np.mean(err_other), np.mean(seq_j), np.mean(seq_other)))
    return rows


sigmas = [0.0, 0.5, 1.0, 2.0, 4.0]
rows = perturbation_test(sigmas)
print("sigma | struct err @j | struct err others | seq err @j | seq err others")
for r in rows:
    print(f"{r[0]:5.1f} | {r[1]:13.4f} | {r[2]:17.4f} | {r[3]:10.3f} | {r[4]:14.3f}")

plt.figure(figsize=(5, 3.5))
plt.plot(sigmas, [r[1] for r in rows], "b-o", label="structure err, perturbed residue")
plt.plot(sigmas, [r[2] for r in rows], "b--s", label="structure err, other residues")
plt.plot(sigmas, [r[3] for r in rows], "r-o", label="sequence err, perturbed residue")
plt.plot(sigmas, [r[4] for r in rows], "r--s", label="sequence err, other residues")
plt.xlabel("perturbation size sigma"); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "latent_locality.png"), dpi=120)
plt.close()


# %% [markdown]
# ## Part 4 - Partially latent flow matching (paper Sec. 3.2.2)
#
# Freeze the VAE. Each training protein now becomes a per-residue vector
#     x_1 = [ CA (3) , z (8) ]       (z sampled from the frozen encoder)
# We learn a velocity field v_theta that transports Gaussian noise x_0 to x_1.
#
# Linear interpolation:   x_t = (1 - t) * x_0 + t * x_1,  target velocity = x_1 - x_0.
#
# The paper's crucial twist: CA and z get SEPARATE times (t_x, t_z), sampled independently
# from different Beta mixtures (Eq. 6). At sampling time this lets us integrate CA faster
# than z. The denoiser is told both times.

# %%
# Latents are not exactly unit-variance, so we standardise each of the D_Z channels for the
# flow model (the CA chain scale ~ 10 A, so we also divide CA by a constant).
with torch.no_grad():
    # Use SAMPLED latents z ~ q(z|x) (what the flow will be trained on), not posterior means:
    # unused ("collapsed") latent dims have mean ~0 but sample std ~1, and dividing by a
    # near-zero mean-std would blow those channels up by orders of magnitude.
    _b = make_batch(512)
    _mu, _lv = enc(_b["ca"], _b["s"], _b["sc"])
    stat = reparameterize(_mu, _lv).reshape(-1, D_Z)
    Z_MEAN, Z_STD = stat.mean(0), stat.std(0).clamp(min=1e-3)
    print("latent std per dim (sampled):", np.round(Z_STD.numpy(), 2))
CA_SCALE = 8.0


def to_flow_space(ca, z):
    """(CA, z) -> normalised [B, L, 3 + D_Z] tensor the flow model works on."""
    return torch.cat([ca / CA_SCALE, (z - Z_MEAN) / Z_STD], dim=-1)


def from_flow_space(x):
    """Inverse of to_flow_space."""
    return x[..., :3] * CA_SCALE, x[..., 3:] * Z_STD + Z_MEAN


def sample_times(B):
    """Independent t_x, t_z from Eq. (6):
         p(t_x) = 0.02 Unif(0,1) + 0.98 Beta(1.9, 1)     (mass near 1: CA is 'ahead')
         p(t_z) = 0.02 Unif(0,1) + 0.98 Beta(1,  1.5)    (more spread out)
    A mixture is implemented by flipping a coin per sample.
    """
    def mix(a, b):
        use_uniform = torch.rand(B) < 0.02
        beta = torch.distributions.Beta(torch.tensor(a), torch.tensor(b)).sample((B,))
        return torch.where(use_uniform, torch.rand(B), beta)
    # TODO: tx = mix(1.9, 1.0); tz = mix(1.0, 1.5)
    raise NotImplementedError("fill in this block")
    return tx, tz


class Denoiser(nn.Module):
    """v_theta(x_t, t_x, t_z): predicts a velocity for every channel of every residue.

    The two times are broadcast to every residue and appended as input features. (The
    paper injects them with adaptive layer-norm; concatenation is the simplest stand-in.)
    """

    def __init__(self):
        super().__init__()
        self.net = ResidueTransformer(3 + D_Z + 2, 3 + D_Z, d_model=128, n_layers=4)

    def forward(self, x_t, tx, tz):
        B = x_t.shape[0]
        t_feat = torch.stack([tx, tz], dim=-1)[:, None, :].expand(B, L, 2)
        return self.net(torch.cat([x_t, t_feat], dim=-1))


def flow_matching_loss(model, x1):
    """Conditional flow matching loss, Eq. (5), with two interpolation times.

    CA channels are interpolated with t_x, latent channels with t_z.
    The network must output (x1 - x0) in every channel.
    """
    B = x1.shape[0]
    x0 = torch.randn_like(x1)                     # noise from the standard Gaussian base
    tx, tz = sample_times(B)
    # TODO: build a [B,1,3+D_Z] time tensor (tx for first 3 channels, tz for the rest);
    #       x_t = (1-t)*x0 + t*x1; target = x1 - x0; loss = mse(model(x_t,tx,tz), target)
    raise NotImplementedError("fill in this block")


# %%
# Train the flow model on (CA, z) pairs produced by the FROZEN encoder.
for p in list(enc.parameters()) + list(dec.parameters()):
    p.requires_grad_(False)

flow = Denoiser().to(DEVICE)
fopt = torch.optim.AdamW(flow.parameters(), lr=5e-4)
for step in range(1, 4001):
    bt = make_batch(128)
    with torch.no_grad():
        mu, logvar = enc(bt["ca"], bt["s"], bt["sc"])
        z = reparameterize(mu, logvar)            # sample from q(z|x), as in the paper's footnote 1
        x1 = to_flow_space(bt["ca"], z)
    loss = flow_matching_loss(flow, x1)
    fopt.zero_grad(); loss.backward(); fopt.step()
    if step % 500 == 0:
        print(f"[Flow] step {step:4d} loss {loss.item():.4f}")


# %% [markdown]
# ## Part 5 - Sampling (paper Sec. 3.4) and the two-schedule ablation
#
# Generation: start from noise at (t_x, t_z) = (0, 0), integrate to (1, 1) with Euler steps
# (we use the deterministic ODE; the paper adds a reduced-noise SDE term on top), then run
# the decoder on (CA, z).
#
# The paper's finding: moving CA *faster* than z works better than a coupled schedule.
# We compare the two schedules on a simple physical check: do generated consecutive CA
# atoms sit ~3.8 A apart?

# %%
@torch.no_grad()
def sample(n, n_steps=100, ca_schedule="fast"):
    """Euler integration of the joint flow with independent time grids.

    ca_schedule="coupled": t_x = t_z = u           (one shared clock)
    ca_schedule="fast"   : t_x = 1-(1-u)^2 >= u    (CA runs ahead of the latents)
    where u in [0, 1] is a plain uniform grid of n_steps points.
    """
    u = torch.linspace(0, 1, n_steps + 1)
    tz_grid = u
    # TODO: tx_grid = u for 'coupled'; 1-(1-u)**2 for 'fast'
    raise NotImplementedError("fill in this block")

    x = torch.randn(n, L, 3 + D_Z)
    for i in range(n_steps):
        tx0, tx1 = tx_grid[i], tx_grid[i + 1]
        tz0, tz1 = tz_grid[i], tz_grid[i + 1]
        v = flow(x, tx0.expand(n), tz0.expand(n))
        # TODO: Euler step with each channel group using ITS OWN time increment
        raise NotImplementedError("fill in this block")
    return from_flow_space(x)


def evaluate_samples(ca, z):
    """Decode (CA, z) and report simple sanity metrics for generated 'proteins'."""
    logits, sc_hat = dec(ca, z)
    bond = (ca[:, 1:] - ca[:, :-1]).norm(dim=-1)                       # consecutive CA distances
    bond_err = (bond - CA_BOND).abs().mean().item()
    types = logits.argmax(-1)
    # Template consistency: after decoding, real atoms should sit near SOME rotation of the
    # type's template, so their distance from the CA should match the template's radius.
    radius_hat = sc_hat.norm(dim=-1)                                   # [B, L, A]
    radius_ref = TEMPLATES.norm(dim=-1)[types]                         # radius of decoded type
    m = SLOT_MASK[types]
    radius_err = ((radius_hat - radius_ref).abs() * m).sum().item() / m.sum().item()
    return bond_err, radius_err, torch.bincount(types.flatten(), minlength=K).float() / types.numel()


# Reference: how good are REAL proteins under the same metrics?
rb = make_batch(256)
with torch.no_grad():
    ref = evaluate_samples(rb["ca"], encode_mean(rb))
print(f"Real data   : CA-bond error {ref[0]:.3f} A | side-chain radius error {ref[1]:.3f} A")

results = {}
for sched in ["coupled", "fast"]:
    ca_s, z_s = sample(256, ca_schedule=sched)
    with torch.no_grad():
        results[sched] = evaluate_samples(ca_s, z_s)
    print(f"Sampled ({sched:7s}): CA-bond error {results[sched][0]:.3f} A | "
          f"side-chain radius error {results[sched][1]:.3f} A | "
          f"type freqs {np.round(results[sched][2].numpy(), 2)}")
# Expect: roughly uniform type frequencies (the data is uniform) and a CA-bond error far
# below ~3.8 A (what unstructured noise would give). In this tiny toy the 'fast' and
# 'coupled' schedules come out nearly the same; the paper's gain appears at full scale.
# Try more steps / a longer-trained flow / reduced-noise SDE sampling to see if it separates.


# %% [markdown]
# ## Reflection questions
# 1. Why does the decoder receive the CA coordinates as input instead of having them inside z?
#    (Paper Sec. 3.1 (ii): modelling CA explicitly beat putting CA in the latent.)
# 2. Set BETA = 1.0 and retrain Part 2. What happens to reconstruction and to the clustering
#    accuracy? Why does the paper choose a tiny beta?
# 3. In the locality test, which does a perturbation hurt more: sequence or structure? Why?
# 4. Replace the shared clock with two independent grids in a different way
#    (e.g. CA slower than z) and measure the bond error. Does the paper's claim hold here?
