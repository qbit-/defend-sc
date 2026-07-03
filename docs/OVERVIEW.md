# Defending from SIPIT attack with low-rank activation perturbation

## Summary

We consider a scenario in which an attacker can intercept data in the transmission channel or on the server before it enters the protected part of the model. We assume that once the data has entered this part of the model, it is no longer available to the attacker. The defense is for the client to transmit noised activations, while the server applies private noise suppression inside the network. This setting is described in the [preliminaries](#1-preliminaries) and in the [experiment idea](#2-idea-and-experiment). We build the experiment idea on our previous work on LLM robustness [ICML2026](https://icml.cc/virtual/2026/poster/66057).
The code to reproduce experiments can be found at [Github](https://github.com/qbit-/defend-sc.git).


The main result is shown in the [Pareto plot](artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_exact_privacy_utility_frontier.png): the horizontal axis shows the success rate of [SIPIT](https://github.com/giorgosnikolaou/SIPIT), meaning the frequency of successful token reconstruction, while the vertical axis shows downstream task accuracy. Noise reduces attack success, but it also degrades model quality. Without adaptation, the server side quickly loses accuracy; after applying the private suppressor, part of that quality can be recovered. We did not run a broad optimization sweep or try to maximize performance: in this proof-of-principle experiment, only the server-side input was adapted.

The [second plot](artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_utility_vs_scale.png) shows how accuracy falls as noise intensity increases, and the [third plot](artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_eve_vs_scale.png) compares several attack variants: one where the attacker knows the noise model exactly, one using a sequential prior, and one using a simple Euclidean score. For sufficiently large noise levels, SIPIT effectively fails in this experiment. This example shows that a sharp loss in accuracy can be partially compensated, but fully recovering quality and studying the limits of the approach require separate work and are outside the scope of this proof of principle.

### 1. Preliminaries
[SIPIT](https://github.com/giorgosnikolaou/SIPIT) is an algorithm to recover the original input sequence from a sequence of hidden states of the transformer model. SIPIT is based on the result of the work ["Language models are injective and hence invertible"](https://arxiv.org/pdf/2510.15511). For the inversion of the model, the attacker needs access to the full sequence of activations at some split level in the transformer model and the ability to run inference with this model up to the split level.

The setting is as follows:
1. The client encodes an input sequence with a part of the language model up to split level $k$ and transmits activations to the server.
2. The attacker **Eve** has access to the full sequence of transmitted activations and the client part of the model. The attacker tries to invert the client's sequence.

### 2. Idea and experiment
The idea of this work is to add structured noise to the activations before transmission. The transmission sequence is then:
1. Run the client-side of a language model up to split layer $k$.
2. Add structured noise to the cut activation before sending it to the server.
3. The server receives the noised activations, which are also the activations seen by the attacker.
4. The server optionally applies a private noise suppressor to recover a task-useful activation before decoding.

If $a_t \in \mathbb{R}^H$ is the activation row at token position $t$, the transmitted activation is

$$
\tilde{a}_t = a_t + \eta_t.
$$

The server either decodes from $\tilde{a}_t$ directly, or first applies a private suppressor $D_{\mathrm{priv}}$ and decodes from

$$
\hat{a}_t = D_{\mathrm{priv}}(\tilde{a}_t).
$$

The concrete settings used in this experiment:
* Model: Qwen2.5-0.5B
* Task: SST-2 (sequences of up to 64 tokens long, binary classification)
* Cut level $k$: 8.
* Number of random noise realisations $K$: 2
* Number of attack prompts for Eve $B$: 30.

### 3. Results
We measure whether the downstream task still works and the success rate of the
attacker. The main result is a Pareto graph. Each point corresponds to one
noise scale `sf`, defined in
[Low-Rank Structured Covariance](#low-rank-structured-covariance).

![Accuracy vs success rate](artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_exact_privacy_utility_frontier.png)

The plot shows downstream task accuracy (SST-2 accuracy) with respect to the attacker's success rate (exact Eve token top-1):

Blue: without server-side noise suppression. Orange: with server-side noise suppression. The dotted line is the accuracy level of the original model without activation noise.

The x-axis is measured on the raw noised channel. The y-axis is measured either from the raw noised activation $\tilde{a}$ or from the suppressed activation $D_{\mathrm{priv}}(\tilde{a})$. The graph shows that at the same raw-channel attack success rate, the server-private suppressor can recover more task accuracy.

#### Supporting plots

*Server downstream task accuracy*

![Server downstream task accuracy](artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_utility_vs_scale.png)

The plot shows the sensitivity of the downstream task to different noise scales. Blue: without server-side noise suppression. Orange: with noise suppression. The dotted line is the clean no-noise baseline. The plotted accuracy is computed from the averaged logits over $K$ the sampled noisy realizations:

$$
\bar{\ell}(x)
=
\frac{1}{K}
\sum_{i=1}^K
\ell(\tilde{a}^{(i)}),
\qquad
\mathrm{Accuracy}
=
\Pr[\arg\max_c \bar{\ell}_c(x) = y].
$$

*Suppressor gain and clean distortion*

![Suppressor gain and clean distortion](artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_gain_and_clean_distortion.png)

The plot shows two quantities as functions of noise scale. The green curve is the task-accuracy gain from the private suppressor:

$$
\Delta_{\mathrm{acc}}
=
\mathrm{Accuracy}(D_{\mathrm{priv}}(\tilde{a}))
-
\mathrm{Accuracy}(\tilde{a}).
$$

The red curve is the clean activation distortion introduced by the suppressor:

$$
\mathrm{Distortion}_{\mathrm{clean}}
=
\frac{\|D_{\mathrm{priv}}(a)-a\|_2}{\|a\|_2}.
$$

The desired regime is positive $\Delta_{\mathrm{acc}}$ with small clean distortion.

*Attacker success for different scoring rules and noise scales*

![Attacker success for different scoring rules](artifacts/plots/setting_g_qwen_sst2_lowrank_private_suppressor_eve_vs_scale.png)

The plot shows the success rate of the attacker at different noise scales. For each candidate token $v$, the attacker compares the candidate activation $c_t(v)$ to the observed noised activation $\tilde{a}_t$. Different scores were tested to choose among candidate tokens $x$

Blue (exact/Mahalanobis): exact Gaussian likelihood under the known noise covariance $\Sigma$,

$$
s_{\mathrm{exact}}(v)
=
-\frac{1}{2}
(c_t(v)-\tilde{a}_t)^\top
\Sigma^{-1}
(c_t(v)-\tilde{a}_t).
$$

Orange (seq-MAP): Mahalanobis activation score plus the model log-prior for the candidate token,

$$
s_{\mathrm{seq}}(v)
=
-(c_t(v)-\tilde{a}_t)^\top
\Sigma^{-1}
(c_t(v)-\tilde{a}_t)
+
\log p_\theta(v \mid x_{<t}).
$$

Green (vanilla): Euclidean nearest-candidate scoring,

$$
s_{\mathrm{vanilla}}(v)
=
-\|c_t(v)-\tilde{a}_t\|_2^2.
$$

In all cases Eve predicts:

$$
\hat{x}_t
=
\arg\max_{v \in \mathcal{V}_t}
s(v).
$$

It turns out that "vanilla" is the strongest attacker over much of the plotted range, but not at high noise scales; the best attacker overall depends on `sf`. We explain it by the fact that in this implementation, the low-rank covariance is singular, and Mahalanobis and seq-MAP scorings drop components outside the noised low-rank subspace. That makes `exact/Mahalanobis` and `seq-MAP` blind to some activation directions. Vanilla in contrast uses the full Euclidean distance.

### Inference-time impact measurement

The script `scripts/18_measure_inference_slowdown.py` measures the runtime cost
of adding low-rank activation noise and applying the private suppressor. This is
a timing-only benchmark: it does not estimate noise subspaces from the data. Instead,
for each model it generates a random orthonormal placeholder $U_\eta \in \mathbb{R}^{H \times r}$ with the correct hidden size $H$ and rank $r$.

For each Qwen2.5 model, the benchmark first measures ordinary full-model
inference $t_{\mathrm{full}}$.

It then measures the split protocol as separate timed components:

$$
t_{\mathrm{split}}
=
t_{\mathrm{head}}
+
t_{\mathrm{noise}}
+
t_{\mathrm{sup}}
+
t_{\mathrm{tail}}.
$$

Here $t_{\mathrm{head}}$ is the time for embeddings and transformer layers before split layer $k$, $t_{\mathrm{noise}}$ is the time to generate and add low-rank noise, $t_{\mathrm{sup}}$ is the time to apply the suppressor, and $t_{\mathrm{tail}}$ is the time for layers after the split plus final norm and
LM head. The reported slowdown is:

$$
\mathrm{slowdown}
=
\frac{t_{\mathrm{split}}}{t_{\mathrm{full}}}.
$$

For a split activation $a \in \mathbb{R}^{B \times T \times H}$, placeholder
noise is generated as:

$$
\tilde{a}
=
a
+
z U_\eta^\top,
\qquad
z \in \mathbb{R}^{B \times T \times r}.
$$

The placeholder suppressor uses the same low-rank algebraic form as the private
low-rank suppressor:

$$
D_{\mathrm{priv}}(y)
=
y
-
\left(
  (y-\mu)U_\eta
  \odot
  \gamma
\right)
U_\eta^\top.
$$

The benchmark parameters are:

- Models: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, and Qwen2.5-7B.
- Dataset: GLUE SST-2 validation prompts.
- Number of prompts: 256.
- Batch size: 32.
- Maximum sequence length: 64.
- Split layer: $k=8$.
- Noise rank: $r=8$.

We do 2 warmup iterations and 5 measured iterations by default.

The plots below show a slowdown ratio and an absolute time for noise addition and suppression.

![Slowdown plot](artifacts/plots/qwen_inference_slowdown_bar.png)

![Absolute time plot](artifacts/plots/qwen_inference_slowdown_noise_suppressor_times.png)

### 4. Implementation details

The attacker implemented here is a candidate-cloud version of SIPIT. The code
does not run the full gradient-based SIPIT attack from the paper. Instead, it
constructs a finite set of candidate tokens and asks which candidate's hidden
activation best matches the observed noisy activation.

Let a prompt be tokenized as:

$$
x = (x_1, x_2, \ldots, x_T).
$$

Let $h_k(x) \in \mathbb{R}^{T \times H}$ be the activation at split layer $k$,
where $H$ is the hidden dimension. For a token position $t$, the clean cut
activation row is:

$$
a_t = h_k(x)_t \in \mathbb{R}^H.
$$

The channel sends a perturbed version:

$$
\tilde{a}_t = a_t + \eta_t.
$$

Eve is teacher-forced on the true prefix $x_{<t}$. For each position, the code
builds a candidate set:

$$
\mathcal{V}_t
=
\{\text{top-}80\text{ model candidates}\}
\cup
\{\text{20 random vocabulary tokens}\}
\cup
\{x_t\}.
$$

For each candidate $v \in \mathcal{V}_t$, the model is run on the hypothetical
sequence $x_{<t} \Vert v$, and the candidate's cut activation is recorded:

$$
c_t(v) = h_k(x_{<t} \Vert v)_t.
$$

The set

$$
\mathcal{C}_t = \{c_t(v) : v \in \mathcal{V}_t\}
$$

is the candidate cloud. The attack predicts the token whose candidate
activation is closest to the observed noisy activation under some score.

The vanilla score is Euclidean:

$$
s_{\mathrm{vanilla}}(v)
=
-\|c_t(v) - \tilde{a}_t\|_2^2.
$$

The Mahalanobis score uses the known covariance $\Sigma$:

$$
s_{\mathrm{maha}}(v)
=
-(c_t(v)-\tilde{a}_t)^\top
\Sigma^{-1}
(c_t(v)-\tilde{a}_t).
$$

The exact Gaussian Eve score drops constants in the Gaussian log density:

$$
s_{\mathrm{exact}}(v)
=
-\frac{1}{2}
(c_t(v)-\tilde{a}_t)^\top
\Sigma^{-1}
(c_t(v)-\tilde{a}_t).
$$

Then:

$$
\hat{x}_t = \arg\max_{v \in \mathcal{V}_t} s(v).
$$

The reported Eve metric is:

$$
\mathrm{EveTop1}
=
\frac{
  \#\{(i,t): \hat{x}_{i,t} = x_{i,t}\}
}{
  \#\{(i,t): \text{position }t\text{ evaluated for prompt }i\}
}.
$$

The important detail is that this is an oracle-candidate evaluation. The
ground-truth token is always in the candidate set, and Eve receives the true
prefix. EveTop1 is an upper bound on the expected SIPIT performance.

#### Activation Perturbation Idea

The activation perturbation is the privacy mechanism. The client does not send
$a_t$ directly. It sends:

$$
\tilde{a}_t = a_t + \eta_t.
$$

If two different candidate tokens produce nearby cut activations, then adding
noise can make the observed point ambiguous. Eve sees $\tilde{a}_t$ and tries
to infer which candidate cloud point generated it. Noise is useful when it
moves $\tilde{a}_t$ along directions that separate token candidates, because
that directly damages the nearest-candidate inference problem.

However, the same activation is also used by the server to complete the task.
Noise is harmful when it moves activations along directions that the downstream
task uses. The code therefore tries to estimate two kinds of directions:

- prompt-recovery directions, where candidate tokens differ;
- task directions, where perturbations change SST-2 label logits.

The covariance construction tries to put noise in directions that are strong
for prompt recovery and comparatively weak for task accuracy.

#### Noise Construction and Prompt / Target Spaces

Noise construction is split across `02_subspace_alignment.py`,
`04_build_covariance.py`, and `12b_scale_lowrank_noise.py`.

### Cut Activation Space

All noise lives in the split-layer hidden space:

$$
\mathcal{H}_k \cong \mathbb{R}^H.
$$

For each valid token row, the clean activation is $a_t \in \mathbb{R}^H$, and
the noisy activation is $\tilde{a}_t \in \mathbb{R}^H$.

### Prompt-Recovery Space

`scripts/02_subspace_alignment.py` estimates a SIPIT-style prompt-recovery
subspace $U_S$.

For a prompt and position $t$, it creates a candidate cloud:

$$
C_t =
\begin{bmatrix}
c_t(v_1)^\top \\
c_t(v_2)^\top \\
\vdots \\
c_t(v_m)^\top
\end{bmatrix}
\in \mathbb{R}^{m \times H}.
$$

It centers the cloud:

$$
X_t = C_t - \mathbf{1}\bar{c}_t^\top,
$$

then computes an SVD:

$$
X_t = L_t S_t R_t^\top.
$$

The leading right singular vectors in $R_t$ are directions in activation space
along which candidate tokens differ. The script retains enough directions to
explain 90% of the local cloud variance, up to a rank cap. Across prompts and
positions, the local bases are concatenated and compressed by another SVD to
form the aggregate prompt-recovery basis:

$$
U_S \in \mathbb{R}^{H \times r_S}.
$$

### Target / Task Space

The target task is SST-2 sentiment classification. The task space $U_T$ is
estimated by `src/sst2_geometry.py`.

The script first constructs probe directions in activation space: some PCA
directions from clean activations and some random directions. For each probe
direction $u$, it perturbs the answer-position activation:

$$
a_{\mathrm{ans}}' = a_{\mathrm{ans}} + \tau u,
$$

runs the server tail, and records the change in the two SST-2 label logits:

$$
\Delta \ell(u)
=
\ell(a_{\mathrm{ans}} + \tau u) - \ell(a_{\mathrm{ans}})
\in \mathbb{R}^2.
$$

This is done for every prompt in the batch using the same probe direction. If
there are $B$ prompts, the code obtains a response tensor of shape
$B \times 2$: one row per prompt and one column per SST-2 label logit. This
tensor is flattened over prompts and labels into one vector:

$$
r(u)
=
\mathrm{vec}
\left(
\begin{bmatrix}
\Delta \ell_1(u)_1 & \Delta \ell_1(u)_2 \\
\vdots & \vdots \\
\Delta \ell_B(u)_1 & \Delta \ell_B(u)_2
\end{bmatrix}
\right)
\in \mathbb{R}^{2B}.
$$

The response matrix stacks one such vector for each of the $q$ probe
directions:

$$
R \in \mathbb{R}^{q \times 2B}.
$$

Equivalently, row $j$ of $R$ is the flattened label-logit response produced by
perturbing the answer-position activation with probe direction $u_j$:

$$
R_{j,:}
=
r(u_j)^\top.
$$

An SVD of $R$ identifies probe combinations that produce large changes in label
logits. Mapping those leading combinations back through the probe directions
gives:

$$
U_T \in \mathbb{R}^{H \times r_T}.
$$

This is the estimated task-sensitive subspace.

### Subspace Alignment Diagnostic

The script compares $U_S$ and $U_T$ using principal angles and the fraction of
$S$-mass outside $T$:

$$
\mathrm{mass}_{T^\perp}(S)
=
1 -
\frac{\|U_T^\top U_S\|_F^2}{r_S}.
$$

If this value is large, many prompt-recovery directions lie outside the
estimated task space, suggesting that a privacy/utility tradeoff may exist.

### Covariance Orientation

`scripts/04_build_covariance.py` constructs the actual low-rank noise basis.
It estimates a privacy geometry matrix $G_{\mathrm{priv}}$ from candidate
clouds. For each centered candidate cloud row $x$, it normalizes:

$$
\hat{x} = \frac{x}{\|x\|_2},
$$

and accumulates:

$$
G_{\mathrm{priv}}
\approx
\frac{1}{N}
\sum_i
\hat{x}_i \hat{x}_i^\top.
$$

The task geometry matrix is:

$$
G_{\mathrm{util}} = U_T U_T^\top.
$$

The code then solves a regularized generalized-eigenvalue problem. With
$\rho = 10^{-4}$:

$$
G_p = G_{\mathrm{priv}} + \rho I,
\qquad
G_u = G_{\mathrm{util}} + \rho I.
$$

It computes a Cholesky factor:

$$
G_u = L L^\top,
$$

then diagonalizes:

$$
A = L^{-1} G_p L^{-\top}.
$$

Here $G_p, G_u, L,$ and $A$ are all $H \times H$ matrices, where $H$ is the
hidden size at the cut layer. The eigendecomposition of $A$ produces
eigenvectors

$$
z_i \in \mathbb{R}^H.
$$

Directions with large eigenvalues have high prompt-recovery mass relative to
task mass. The code sorts eigenvalues from largest to smallest, maps the
corresponding eigenvectors back to activation space,

$$
u_i \propto L^{-\top} z_i,
$$

and normalizes each resulting $u_i \in \mathbb{R}^H$.

The number of retained directions is the experiment hyperparameter `rank`,
passed to `scripts/04_build_covariance.py` with `--ranks`. In the experiments we set $r = 8$. After taking the top $r$ directions,
the code QR-orthonormalizes them into:

$$
U_\eta \in \mathbb{R}^{H \times r}.
$$

These columns are the noise directions. Equivalently, $U_\eta$ defines the
$r$-dimensional subspace in which `lowrank_struct` noise is injected.

### Low-Rank Structured Covariance

The general covariance representation in `src/noise.py` is:

$$
\Sigma = \sigma_0^2 I + U_\eta \Lambda U_\eta^\top.
$$

The current `lowrank_struct` family intentionally sets:

$$
\sigma_0 = 0.
$$

So the covariance is singular low-rank:

$$
\Sigma = U_\eta \Lambda U_\eta^\top,
\qquad
\mathrm{rank}(\Sigma) \le r \ll H.
$$

Noise is sampled as:

$$
\eta = U_\eta \Lambda^{1/2} z,
\qquad
z \sim \mathcal{N}(0, I_r)
$$

for the Gaussian experiments used in the exact Eve evaluation. The code also
supports other unit-variance factor distributions, but the main frontier uses
Gaussian scoring.

The noise scale `sf` is a dimensionless noise-strength measure used when
building the covariance matrix of the noise.

For a scale `sf`, the script first computes a reference per-coordinate standard
deviation from the median clean activation norm:

$$
\sigma_{\mathrm{budget}}
=
\mathrm{sf}
\frac{\mathrm{median}\ \|a\|_2}{\sqrt{H}},
$$

and sets the total low-rank variance budget to:

$$
\mathrm{tr}(\Sigma)
=
2H\sigma_{\mathrm{budget}}^2.
$$

The top generalized eigenvalues are clipped nonnegative and normalized so their
sum equals this budget:

$$
\lambda_i
=
\frac{\max(\alpha_i,0)}
       {\sum_{j=1}^r \max(\alpha_j,0)}
\cdot
2H\sigma_{\mathrm{budget}}^2.
$$

### Noise Scaling

`scripts/12b_scale_lowrank_noise.py` creates additional covariance files
without recomputing $U_\eta$. Given a base scale `base_sf` and target scale
`target_sf`, it sets:

$$
\gamma_{\mathrm{scale}}
=
\frac{\mathrm{target\_sf}}{\mathrm{base\_sf}}.
$$

Then:

$$
U_\eta' = U_\eta,
\qquad
\Lambda' = \gamma_{\mathrm{scale}}^2 \Lambda,
\qquad
\sigma_0' = \gamma_{\mathrm{scale}}\sigma_0.
$$

For `lowrank_struct`, $\sigma_0 = 0$, so:

$$
\Sigma' = \gamma_{\mathrm{scale}}^2 \Sigma.
$$

Equivalently, sampled noise amplitudes scale linearly:

$$
\eta' = \gamma_{\mathrm{scale}}\eta.
$$

## 4. Suppressing Noise Before Decoding

The private suppressor is implemented in `src/private_denoise.py` as
`PrivateLowrankStructSuppressor`. It is a closed-form affine linear estimator fit from train-split
activation statistics in 
`scripts/14b_train_private_denoiser.py`.

The low-rank channel is:

$$
y = a + \eta,
\qquad
\eta = U_\eta \Lambda_\eta^{1/2} z.
$$

Only coordinates in $\mathrm{span}(U_\eta)$ are noised. Coordinates orthogonal
to $U_\eta$ are unchanged by the channel.

The suppressor estimates the calibration mean:

$$
\mu = \mathbb{E}_{\mathrm{train}}[a],
$$

and projects centered training activations into the noise subspace:

$$
c = U_\eta^\top(a-\mu).
$$

For each low-rank coordinate $j$, it estimates a scalar clean prior variance:

$$
\nu_j = \mathrm{Var}_{\mathrm{train}}(c_j).
$$

The known noise variance in that coordinate is:

$$
\lambda_j.
$$

Assuming the scalar Gaussian model:

$$
c_j \sim \mathcal{N}(0,\nu_j),
\qquad
\epsilon_j \sim \mathcal{N}(0,\lambda_j),
\qquad
\tilde{c}_j = c_j + \epsilon_j,
$$

the posterior mean is:

$$
\mathbb{E}[c_j \mid \tilde{c}_j]
=
\frac{\nu_j}{\nu_j+\lambda_j}\tilde{c}_j.
$$

The code stores:

$$
\gamma_j =
\frac{\lambda_j}{\nu_j+\lambda_j}.
$$

Applying the suppressor to an activation row $y$:

$$
\tilde{c} = U_\eta^\top(y-\mu),
$$

then subtracts the estimated noise component:

$$
D_{\mathrm{priv}}(y)
=
y
-
U_\eta
\operatorname{diag}(\gamma)
U_\eta^\top
(y-\mu).
$$

Equivalently, within the noised subspace:

$$
U_\eta^\top(D_{\mathrm{priv}}(y)-\mu)
=
\operatorname{diag}
\left(
  \frac{\nu}{\nu+\lambda}
\right)
U_\eta^\top(y-\mu).
$$

Outside the noised subspace, the suppressor is identity:

$$
P_{U_\eta^\perp}D_{\mathrm{priv}}(y)
=
P_{U_\eta^\perp}y.
$$

This is why it is called a server-private suppressor: the server can apply this
linear shrinkage before running the remaining model layers. It does not modify
the raw noisy channel that Eve is evaluated against in the main frontier.

## 5. Experiment and Metric Details

### Pipeline

The standard end-to-end pipeline is `run_frontier.sh`:

1. `01_collect_calibration.py` caches train/test cut activations and SST-2 task
   artifacts.
2. `02_subspace_alignment.py` estimates prompt-recovery and task subspaces.
3. `04_build_covariance.py` builds the base low-rank structured covariance.
4. `12b_scale_lowrank_noise.py` rescales that covariance over many `sf` values.
5. `14b_train_private_denoiser.py` fits a suppressor for each noise scale.
6. `15_eval_tnsc.py` evaluates utility and Eve recovery.
7. `17_plot_lowrank_private_suppressor.py` renders the plots.

Artifacts are written under `artifacts/`.

### Utility Evaluation

For each test example, the evaluator starts with cached clipped activations:

$$
a = \mathrm{clip}(h_k(x)).
$$

If no covariance is used, clean logits define the clean baseline. If noise is
used, it samples:

$$
\tilde{a}^{(i)} = a + \eta^{(i)}
$$

for `K` stochastic realizations. The raw server logits are computed by running
the model tail from $\tilde{a}^{(i)}$. The suppressed logits are computed by
running the model tail from:

$$
D_{\mathrm{priv}}(\tilde{a}^{(i)}).
$$

The evaluator averages logits across the `K` realizations:

$$
\bar{\ell}
=
\frac{1}{K}
\sum_{i=1}^K \ell^{(i)}.
$$

SST-2 accuracy is:

$$
\mathrm{Accuracy}
=
\frac{1}{B}
\sum_{b=1}^B
\mathbf{1}
\left[
  \arg\max_c \bar{\ell}_{b,c}
  =
  y_b
\right].
$$

The CSV also includes:

- `kl_no_corr` and `kl_with_corr`: average
  $\mathrm{KL}(\mathrm{softmax}(\ell_{\mathrm{clean}})
  \Vert
  \mathrm{softmax}(\ell_{\mathrm{noisy}}))$.
- `top1_no_corr` and `top1_with_corr`: agreement with clean top-1 labels.
- `clean_distortion`: relative change introduced by the suppressor on clean
  activations.
- `tame_C_sum` and `tame_V_sum`: TAME-style summaries of how raw noisy logits
  differ from clean logits across the $K$ noise realizations. If
  $\ell_{b,k,c}^{\mathrm{noisy}}$ is the noisy logit for example $b$, noise
  sample $k$, and class $c$, the code forms the shift

  $$
  \delta_{b,k,c}
  =
  \ell_{b,k,c}^{\mathrm{noisy}}
  -
  \ell_{b,c}^{\mathrm{clean}}.
  $$

  `tame_C_sum` is the sum over classes of the largest absolute mean shift across
  examples:

  $$
  \sum_c \max_b
  \left|
  \frac{1}{K}
  \sum_{k=1}^K
  \delta_{b,k,c}
  \right|.
  $$

  `tame_V_sum` is the sum over classes of the largest sample variance of that
  shift across noise realizations:

  $$
  \sum_c
  \max_b
  \widehat{\mathrm{Var}}_k
  \left[
  \delta_{b,k,c}
  \right].
  $$

  Larger values mean the activation noise causes larger systematic logit bias
  or larger logit variability.
- `margin_cert_noisy`: a one-sided margin certificate rate for the raw noisy
  logits. For each example, the clean true-class margin is compared with the
  mean and variance of the noisy margin shift. The example is counted as
  certified if:

  $$
  m_b^{\mathrm{clean}}
  +
  \mu_b
  >
  \sqrt{
    \widehat{\mathrm{Var}}_b
    \frac{1-\phi}{\phi}
  },
  \qquad
  \phi = 0.05.
  $$

  Here $m_b^{\mathrm{clean}}$ is the clean true-class logit margin and
  $\mu_b$ is the mean noisy-margin shift. The reported value is the fraction of
  examples satisfying this inequality.
- `ood_maha_mean` and `ood_maha_p95`: diagonal Mahalanobis
  out-of-distribution scores for raw noisy activations relative to clean
  activation statistics. The code estimates a coordinate-wise clean mean
  $\mu_h$ and variance $\sigma_h^2$ over clean cut activations, then scores each
  noisy activation row $x$ by:

  $$
  \sum_h
  \frac{(x_h-\mu_h)^2}
       {\max(\sigma_h^2, 10^{-8})}.
  $$

  `ood_maha_mean` is the mean of these scores over evaluated rows and noise
  samples; `ood_maha_p95` is their 95th percentile.

### Eve Evaluation

Eve is evaluated on a configurable attack split, usually SST-2 validation.
`run_frontier.sh` uses:

```bash
--repeats 1
--n-attack-prompts 30
--positions 2 5 8 10 12 15 18 20
```

For each attack prompt and position, the evaluator builds or loads candidate
clouds with:

$$
80\text{ top candidates} + 20\text{ random candidates} + 1\text{ truth token}.
$$

The exact Eve score uses the Gaussian log-likelihood under the effective
covariance. For repeat count $m$, different repeat policies define different
effective covariance scales:

$$
\Sigma_{\mathrm{eff}}
=
\frac{1}{m_{\mathrm{eff}}}\Sigma.
$$

The implemented policies are:

- `independent_average`: $m_{\mathrm{eff}} = m$.
- `sticky_same_noise`: $m_{\mathrm{eff}} = 1$.
- `mixed_retry`: $m_{\mathrm{eff}} = \max(1,\lfloor(m+1)/2\rfloor)$.

The main plotted frontier uses one-shot rows:

$$
m = 1,
$$

so all policies reduce to the same covariance strength for the plotted
one-shot attack.

### Singular Covariance Handling

Because `lowrank_struct` has:

$$
\Sigma = U_\eta \Lambda U_\eta^\top,
\qquad
\sigma_0=0,
$$

the covariance is singular in the full hidden space. The code handles this in
`GaussianCov.whiten()` by whitening only inside $\mathrm{span}(U_\eta)$:

$$
\Sigma^{-1/2}x
\equiv
U_\eta
\Lambda^{-1/2}
U_\eta^\top x.
$$

Perpendicular components are dropped for Mahalanobis scoring. Therefore exact
Eve compares candidates only through the noised low-rank support. This matches
the modeled support of the noise distribution used by the code, but it is also
an important modeling choice when interpreting `eve_raw_exact`.

### Interpretation

The experiment is best read as a controlled candidate-set reconstruction test,
not a complete privacy proof. The frontier shows how task accuracy and
candidate-token recovery move as the low-rank noise scale changes. The
interesting regime is where EveTop1 is low, while SST2Accuracyremains high after suppression.

That regime indicates that the estimated prompt-recovery directions and
task-utility directions are sufficiently separated for this split, model, task,
and candidate attack setup.
