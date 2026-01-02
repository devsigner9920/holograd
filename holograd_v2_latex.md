# HoloGrad: Proof-of-Gradient-Projection (PoGP) for Verifiable,
Mining-Style Distributed LLM Pretraining
WonJune Kang
devsigner9920@gmail.com
Abstract
Large-scale language model (LLM) pretraining is bottlenecked by high-dimensional gradient/parameter
tensor synchronization and straggler effects, effectively restricting training to data-center-grade clusters.
We propose HoloGrad, a protocol and system that replaces tensor communication with verifiable scalar
proofs based on gradient projections (directional derivatives). Each worker submits only a seed and a
scalar projection value, while the coordinator reconstructs directions from seeds and aggregates updates
via linear synthesis. We introduce Proof-of-Gradient-Projection (PoGP): a sampling-based verification
mechanism enabling low-cost validation and incentive-compatible accounting in a mining-style open
network. To address variance/sample-inefficiency of random-direction estimators, we further propose an
Adaptive Direction Codebook updated by a fixed Streaming Oja-QR rule to track a low-rank subspace
of frequent update directions. We provide smoothness-based descent guarantees under an idealized
(honest, mean aggregation) setting, clarify how stochasticity and robust aggregation introduce additional
bias/variance terms, analyze verification/slashing parameters, and present a reproducible experimental
protocol with heterogeneity and adversarial participants.
## 1 Introduction
Background.
Distributed training typically requires communicating high-dimensional gradients (or equiva-
lent sufficient statistics), inducing bandwidth and latency bottlenecks. Synchronous training further suffers
from stragglers, making permissionless heterogeneous nodes impractical.
Goal.
We aim to reshape pretraining into a mining-like protocol: permissionless participation, cheap
verification, and incentive-compatible rewards—while still improving a global model.
Key idea.
Replace tensor communication with scalar proofs:
$$
a = \langle \nabla_{\theta}LB(\theta), v \rangle, a \in \mathbb{R}
$$
where v is deterministically generated from a public seed. The coordinator reconstructs v and synthesizes
updates from scalar-direction pairs. Directional-derivative-based gradient estimators are connected to forward-
mode AD and "gradients without backpropagation" directions [1, 4].
## 2 Related Work
Directional-derivative / forward-mode gradient construction has been studied as an alternative to back-
propagation [1, 4]. Communication-efficient distributed optimization includes sign-based methods such
as signSGD [2]. Locking-free training via synthetic gradients decouples modules [3]. Our contribution
differs by protocolizing directional-derivative scalars into verifiable proofs with economic enforcement and
by specifying a fixed adaptive direction codebook update rule (Streaming Oja-QR) for sample-efficiency.
## 1
## 3 Problem Formulation and Notation
Let \theta \in \mathbb{R}^D be model parameters, LB(\theta) the minibatch loss, and L(\theta) = EB[LB(\theta)] the population
$$
objective. Let g_B(\theta) = \nabla_{\theta}LB(\theta) and g(\theta) = \nabla_{\theta}L(\theta). System constraints
$$
(i) per-worker uplink must be low-dimensional, ideally O(1) scalars per task; (ii)
verification must be feasible at scale via sampling; (iii) robustness to heterogeneity and adversarial behavior.
## 4 HoloGrad and PoGP Protocol
4.1
Seeded direction generation (fixed)
A public seed s defines a direction via a deterministic generator
$$
v = Dir(s) \in \mathbb{R}^D
$$
We fix Dir to output a unit-norm isotropic direction (e.g., sample z ∼N(0, ID) from a seeded PRNG and
set v = z/ \|z\|). Then \|v\|= 1 and
$$
\mathbb{E}[vv^\top] = \frac{1}{D} I
$$
so throughout the full-space case we use the scale parameter
$$
\sigma^2 := \frac{1}{D}
$$
(Other choices such as unnormalized Rademacher directions are possible but are not used in this paper to
keep norms bounded.)
4.2
Worker proof: scalar projection
Given current checkpoint \thetat and minibatch Bt, a worker computes
$$
a_t = \langle g_{B_t}(\\theta_t), v \rangle
$$
and submits a proof (s, at) plus metadata (optional attestation, timing, signature).
4.3
Coordinator aggregation (HoloGrad synthesis)
Collect K proofs (sj, aj), reconstruct vj = Dir(sj), and synthesize the gradient estimate using the scale
correction:
$$
\hat g_t = 1 K K X j=1 1 \sigma^2 a_t,jvt,j, \\theta_t+1 = \\theta_t -\etabgt
$$
With the fixed unit-norm isotropic Dir above, \sigma2 = 1/D and thus bgt = D
K
P
j at,jvj.
4.4
PoGP verification
A verifier samples each proof with probability pverify and recomputes a*under the same (\thetat, Bt, s):
$$
|a -a*| \leϵ ⇒accept; else slash/penalize
$$
This enables low-cost statistical verification in expectation via sampling; ϵ handles numeric nondeterminism
and implementation differences.
## 2
## 5 Theory: Correctness and Descent
Assumption 1 (Isotropic directions up to scale). Directions satisfy \mathbb{E}[vv\top] = \sigma2I for some \sigma2 > 0.
Theorem 1 (Unbiased directional estimator with scale correction). Let ˆg(v) =
## 1 σ2 (g · v)v. Under Assump-
tion 1,
$$
Ev[ˆg(v)] = g
$$
Proof.
E
 1
\sigma2 (g \cdot v)v

$$
= 1 \sigma^2 \mathbb{E}[vv^\top]g = 1 \sigma^2 (\sigma^2I)g = g
$$
Variance and sample size.
With i.i.d. directions, bg = 1
K
PK
j=1 ˆg(vj) has variance scaling O(1/K); heavy
tails increase the constant factors [4].
5.1
Adaptive Direction Codebook (ADC)
Random directions can be sample-inefficient in high dimensions. We restrict directions to a learned low-rank
subspace.
Fixed z distribution (no normalization).
We fix z ∼N(0, Ir) (or Rademacher ±1) without normalization
so that
$$
\mathbb{E}[zz\top] = Ir
$$
This avoids the "1/r" scaling ambiguity.
Codebook and subspace directions.
Maintain an orthonormal codebook Ut \in \mathbb{R}^D\timesr with r ≪D. A seed
$$
generates z \in \mathbb{R}r and v = Utz
$$
Then \mathbb{E}[vv\top] = UtE[zz\top]U \top
$$
t = UtU \top
$$
t . In the subspace case we interpret Assumption 1 in the projected
form
$$
\mathbb{E}[vv^\top] = \sigma^2UU \top
$$
$$
with \sigma^2 = 1 under \mathbb{E}[zz\top] = Ir
$$
$$
Accordingly, the scale-corrected estimator ˆg(v) = 1
$$
\sigma2 (g \cdot v)v satisfies
$$
\mathbb{E}[ˆg(v)] = UU \topg
$$
i.e., it is unbiased for the projected gradient.
Assumption 2 (L-smoothness). L is L-smooth: for any \delta,
$$
L(\theta + \delta) \leL(\theta) + \langle \nablaL(\theta), \delta \rangle+ L
$$
$$
2 \|\delta\|2 
$$
Theorem 2 (Projected-gradient descent decrease (idealized)). Under Assumption 2, with deterministic update
$$
\delta = -\etaUU \topg
$$
$$
L(\theta -\etaUU \topg) \leL(\theta) -
$$

\eta -L\eta2
## 2 

$$
UU \topg
$$

## 2 .
Thus for \eta \in(0, 2/L), the loss decreases whenever UU \topg ̸= 0.
Proof. Apply L-smoothness with \delta = -\etaUU \topg. The linear term is -\eta

$$
UU \topg
$$

## 2 and the quadratic term
is L\eta2
## 2

$$
UU \topg
$$

2.
## 3 Remark (stochasticity and robust aggregation).
In the full protocol, ∆\thetat is stochastic (mini-batches,
random directions) and may be biased by robust aggregation (e.g., trimming) under adversaries. A full
convergence theorem requires bounding both the variance term and any aggregation-induced bias; we treat
this as an experimental and future-theoretical extension.
## 6 Fixed Codebook Update: Streaming Oja-QR
We now fix the codebook update rule (no ambiguity).
Observation.
The synthesized update provides a streaming signal of frequent update directions. We track a
top-r subspace via Oja's rule with periodic orthonormalization.
6.1
Streaming Oja step
Given the aggregated gradient estimate bgt (or parameter update direction proportional to it) and current Ut:
$$
eUt+1 = Ut + \alphat \hat g_t(bg\top t Ut)
$$
Periodically (every Tqr steps), perform QR:
$$
Ut+1 \leftarrowQR( eUt+1)
$$
$$
Otherwise set Ut+1 \leftarroweUt+1 with column normalization. Captured energy ratio. Define γt =
$$

$$
UtU \top t g(\\theta_t)
$$

## 2 ∥g(θt)∥2
$$
\in[0, 1]
$$
Then Theorem 2 implies idealized descent is proportional to γt \|g\|2.
## 7 Robust Aggregation of Scalar Proofs
Since proofs are scalars, we can robustly aggregate using trimmed mean or median-of-means.
Trimmed mean (fixed default).
Sort {aj} and remove top/bottom \tauK values:
$$
˜aj \leftarrowTrimMean\tau({aj}K j=1)
$$
Then synthesize using P ˜ajvj with the same scale correction as in Section 4.
## 8 Verification and Incentives (Mining-Style Economics)
8.1
Sampling verification and detection probability
Let each submitted proof be verified independently with probability p. Suppose an attacker submits m proofs
per step and a fraction q are invalid (would fail verification). The probability of catching at least one invalid
proof in a step is
$$
P(caught) = 1 -(1 -p)qm. 4
$$
Algorithm 1 HoloGrad Coordinator Step (scale-consistent)
Require: checkpoint \thetat, codebook Ut, minibatch ID Bt, number of proofs K, step size \eta, scale \sigma2
1: Publish tasks: seeds {sj}K
$$
j=1, with (H(\\theta_t), H(B_t), codebook id)
$$
2: Collect proofs {(sj, aj)}K
$$
j=1
$$
$$
3: Robust scalar aggregation {˜aj} \leftarrowTrimMean\tau({aj})
$$
$$
4: for j = 1..K do
$$
5:
$$
vj \leftarrowDir(sj)
$$
$$
▷full-space or vj \leftarrowUtz(sj)
$$
▷subspace
6: end for
$$
7: \hat g_t \leftarrow1 K PK j=1 1
$$
\sigma2 ˜ajvj
▷scale-corrected
$$
8: \\theta_t+1 \leftarrow\\theta_t -\etabgt
$$
9: Update codebook via Streaming Oja-QR using bgt (periodic QR)
Algorithm 2 Worker Proof Generation
Require: task (\thetat, Bt, Ut, s)
$$
1: v \leftarrowDir(s)
$$
$$
▷full-space or v \leftarrowUtz(s)
$$
▷subspace
$$
2: Compute a \leftarrow\langle \nabla_{\theta}LBt(\\theta_t), v \rangle
$$
▷directional derivative
3: Submit proof (s, a) with optional signature/attestation
8.2
Slashing model and incentive-compatibility
We state an explicit step-level slashing model: (i) if no invalid proof is caught in a step, the attacker receives
reward \mathbb{R} for each accepted invalid proof; (ii) if any invalid proof is caught, the step reward is forfeited and
an additional slashing penalty S is applied. Then the expected cheating utility is
$$
\mathbb{E}[Ucheat] = (1 -P(caught)) \cdot (qmR) -P(caught) \cdot S
$$
A sufficient condition for cheating to be non-profitable is \mathbb{E}[Ucheat] \le0, i.e.,
$$
S \ge1 -P(caught) P(caught) qmR with
$$
$$
P(caught) = 1 -(1 -p)qm
$$
Replication (assigning identical tasks to multiple workers) increases detection probability and further discour-
ages attacks.
## 9 Algorithms
## 10 Experimental Protocol (Overview)
We defer full reproducibility details (hashing, scripts, adversary injection) to Appendix A. Core ablations:
(i) K sweep (communication vs perplexity), (ii) codebook rank r sweep and captured energy γt, (iii)
heterogeneity/straggler scenarios, (iv) adversary fraction vs verification rate p.
## 11 Limitations
Key open issues: (i) sample-efficiency at very large D may still require large K without a strong codebook;
(ii) practical performance of forward-mode/JVP in mainstream frameworks; (iii) nondeterminism requires
careful ϵ selection; (iv) extending theory to robust aggregation and adversarial settings requires additional
assumptions on bounded projections and corruption rates.
## 5
## 12 Conclusion
HoloGrad reframes distributed pretraining as a verifiable, mining-style protocol by replacing tensor syn-
chronization with scalar PoGP proofs, enabling permissionless heterogeneous participation. By fixing the
direction generator and explicitly correcting for second-moment scaling, we align theory with implementation.
A fixed Streaming Oja-QR codebook improves sample-efficiency by tracking a low-rank subspace of frequent
updates. We provide descent guarantees for the idealized projected-gradient case, define incentive-compatible
verification/slashing under an explicit model, and present a reproducible evaluation plan.
References
[1] Atılım G¨unes¸ Baydin, Barak A. Pearlmutter, Don Syme, Frank Wood, and Philip Torr. Gradients without
backpropagation. arXiv preprint arXiv:2202.08587, 2022.
[2] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Anima Anandkumar. signsgd: Com-
pressed optimisation for non-convex problems. In Proceedings of the 35th International Conference on
Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, 2018.
[3] Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, David Silver,
and Koray Kavukcuoglu. Decoupled neural interfaces using synthetic gradients. In Proceedings of the
34th International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine
Learning Research, 2017.
[4] Khemraj Shukla and Yeonjong Shin.
Randomized forward mode of automatic differentiation for
optimization algorithms. arXiv preprint arXiv:2310.14168, 2023.
A
Full Reproducibility Protocol
This appendix provides an end-to-end, "clone-and-run" protocol for reproducing the core claims: (i) training
progress under HoloGrad vs baselines, (ii) scaling with number of proofs K, (iii) benefits of the Adaptive
Direction Codebook (ADC), and (iv) robustness under heterogeneity and adversarial submissions with PoGP
sampling verification and slashing.
A.1
Repository Layout (expected)
Create a repository with the following structure (names are suggestions; adjust as needed):
- main.tex, refs.bib
- repro/
– configs/ (YAML experiment configs)
– scripts/ (launch scripts)
– src/ (python modules)
– logs/ (stdout logs)
– runs/ (checkpoints, metrics)
## 6 A.2
Environment and Dependencies
Hardware.
All experiments can be run on a single machine with one GPU for proof-of-concept scaling. For
distributed/heterogeneous tests, multiple machines or containers are recommended. Minimum recommended
single-node baseline:
- GPU: 1\times 24GB VRAM (or larger)
- CPU: 16+ cores
- RAM: 64GB+
- Disk: 500GB+
Software.
- Ubuntu 22.04 (or similar Linux)
- Python 3.10+
- CUDA 12.x + matching PyTorch build
- Git, tmux (optional), Docker (optional)
Python packages.
Pin exact versions to reduce nondeterminism:
- torch, transformers, datasets, accelerate
- numpy, scipy
- pyyaml, tqdm
- wandb (optional)
A.3
Determinism and Hash Commitments
HoloGrad relies on determinism for (a) directional regeneration from seed and (b) PoGP verification.
Global seed.
Choose a global seed S0. For every training step t and proof index j, derive a per-proof seed
via a cryptographic hash:
$$
st,j = Hash(S0 \|H(\\theta_t) \|H(B_t) \|t \|j)
$$
Here, H(\thetat) is a commitment to the checkpoint (e.g., SHA-256 of serialized weights), and H(Bt) is a
commitment to the minibatch identity (dataset name, shard id, sample ids).
Direction generator.
We fix the full-space generator:
$$
z ∼N(0, I_D) from seeded PRNG, v = z
$$
$$
\|z\|
$$
This implies \sigma2 = 1/D and the scale-corrected synthesis uses 1/\sigma2 = D.
## 7 ADC codebook commit.
For ADC, commit to codebook id at each step (or epoch) by hashing the
matrix Ut:
$$
H(Ut) = SHA256(Serialize(Ut))
$$
Workers use the advertised codebook id to load the exact same Ut.
A.4
Core Components to Implement
A minimal implementation needs the following modules:
(1) Coordinator.
Responsibilities:
- Maintain checkpoint \thetat.
- Publish tasks {st,j}K
$$
j=1 and minibatch id B_t
$$
- Collect proofs (s, a).
- Aggregate scalars (mean or trimmed mean) into {˜aj}.
- Reconstruct directions vj = Dir(st,j).
- Synthesize bgt = 1
K
P
j
## 1 σ2 ˜ajvj.
- Update: \thetat+1 = \thetat -\etabgt.
- (ADC) Update Ut via Streaming Oja-QR.
(2) Worker.
Responsibilities:
- Receive (\thetat, Bt, s) and optional codebook id.
- Reconstruct v = Dir(s) (or subspace: v = Utz(s)).
- Compute directional derivative a = \langle \nablaLBt(\thetat), v \rangle.
- Submit proof (s, a) with optional signature/attestation.
(3) Verifier.
Responsibilities:
- Sample each proof with probability pverify.
- Recompute a*under identical (\thetat, Bt, s).
- Accept if |a -a*| \leϵ, else flag and apply slashing.
A.5
Directional Derivative Computation
Two acceptable approaches:
Option A (reference, simplest): full backprop then dot.
$$
Compute g_B(\\theta_t) = \nabla_{\theta}LB(\\theta_t) and set a =
$$
$$
\langle g_B, v \rangle. This is correct but may be expensive. 8
$$
Option B (preferred): forward-mode / JVP.
Compute a directly as a directional derivative (JVP) without
materializing full gB. The protocol correctness does not depend on which method is used, but cost and
feasibility do. Benchmark and report both where possible.
A.6
Datasets and Models
Recommended minimal setting (fast).
- Dataset: WikiText-103 or OpenWebText subset
- Model: GPT-2 small (117M) or similar
- Sequence length: 256–512
Scaling setting (optional).
- Dataset: C4 subset
- Model: 300M–1B parameter decoder-only transformer
- Sequence length: 512–1024
A.7
Experiment Matrix
All experiments log: training loss, validation loss/perplexity, throughput (tokens/s), verification overhead,
rejection/slash counts, and (ADC) captured energy ratio γt.
Baseline comparisons.
- BP-AllReduce: standard distributed training (if available).
- HoloGrad-Full: full-space seeded unit directions, scale-corrected synthesis.
- HoloGrad-ADC: subspace directions with Streaming Oja-QR codebook.
Ablation A (number of proofs).
Sweep K \in{8, 16, 32, 64, 128, 256} for fixed compute budget. Report
convergence vs wall-clock and vs total proofs processed.
Ablation B (codebook rank).
For ADC, sweep r \in{8, 16, 32, 64, 128}. Report γt and convergence speed.
Ablation C (heterogeneity/stragglers).
Simulate heterogeneous workers by adding random delays and
compute caps. Compare synchronous collection vs time-windowed collection (collect-first-K).
Ablation D (adversaries + robust aggregation).
$$
Inject adversarial fraction \alpha \in{0, 0.1, 0.2, 0.3}:
$$
- Random: submit random a values.
- Sign-flip: submit a = -ahonest.
- Extreme: submit a scaled by large factor.
For each, sweep trimming \tau and report bias/variance tradeoffs.
## 9 Ablation E (verification rate).
Sweep pverify \in{0.0, 0.01, 0.05, 0.1, 0.2} and tolerance ϵ. Report: detec-
tion probability, false positive rate (honest flagged), total verification cost.
A.8
Metrics and Reporting
Training quality.
Validation perplexity (or loss) vs steps and vs wall-clock.
Communication.
Bytes uploaded per worker per step. HoloGrad should be O(1) scalars plus metadata.
Verification overhead.
$$
Verifier-cost fraction = time spent verifying total step time . Security/economics
$$
Report empirical P(caught) vs theoretical 1 -(1 -p)qm under controlled injections.
A.9
Recommended Default Hyperparameters
- Learning rate \eta: match baseline optimizer scale; start with 1e-4 to 3e-4 for GPT-2-small.
- Proof count K: 64 for single-GPU PoC.
- Verification rate pverify: 0.05.
- Verification tolerance ϵ: 1e-4 (adjust for numeric nondeterminism).
- Trim rate \tau: 0.1 for moderate adversary rates.
- ADC rank r: 32.
- Oja step size \alphat: constant 1e-3 or decay \alphat = \alpha0/
√
t.
- QR period Tqr: 100 steps.
A.10
Step-by-Step Execution (example)
1) Prepare environment.
- Install pinned dependencies.
- Download dataset shards and write sample-id indices for minibatch commitments.
2) Run baseline.
Train for T steps (e.g., T = 20,000) with standard optimizer and log metrics.
3) Run HoloGrad-Full.
- Fix S0.
- For each step, generate K seeds st,j.
- Collect K proofs, synthesize bgt = D
K
P
j ˜ajvj.
## 10 4) Run HoloGrad-ADC.
- Initialize U0 with random orthonormal columns.
- Use v = Utz with fixed z ∼N(0, Ir).
- Update Ut using Streaming Oja-QR.
5) Robustness tests.
Repeat (3)–(4) with adversary injections, trimming, and PoGP verification.
A.11
Sanity Checks
- Unbiasedness check (full-space): for fixed \theta, estimate 1
K
P
j
## 1 σ2 (g · vj)vj and compare to reference g.
- Scale check: verify that omitting 1/\sigma2 yields a factor mismatch of approximately 1/D under unit-norm
directions.
- Verifier consistency: for honest proofs, measure |a -a*| distribution to set ϵ.
A.12
Artifact Checklist
To claim reproducibility, include:
- Exact commit hash of code and dependency lockfile.
- Global seed S0.
- Dataset name + exact subset/shard ids + sample-id index files.
- Training config YAMLs (all ablations).
- Logs and summary tables (CSV) for metrics.
- Plots: loss/perplexity curves, verifier overhead, detection rates, γt curves.
A.13
Notes on Numerical Nondeterminism
Even with fixed seeds, GPU kernels may be nondeterministic. We recommend:
- Using deterministic flags where available.
- Logging floating-point settings and hardware.
- Setting ϵ using empirical quantiles of |a -a*| for honest runs.
## 11
