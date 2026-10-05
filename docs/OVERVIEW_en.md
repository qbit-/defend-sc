# Defending from SIPIT attack with low-rank activation perturbation

## Summary

We consider a scenario in which an attacker can intercept data in the transmission channel or on the server before it enters the protected part of the model. We assume that once the data has entered this part of the model, it is no longer available to the attacker. The defense is for the client to transmit noised activations, while the server applies private noise suppression inside the network. This setting is described in the [preliminaries](#1-preliminaries) and in the [experiment idea](#2-idea-and-experiment). We build the experiment idea on our previous work on LLM robustness [ICML2026](https://icml.cc/virtual/2026/poster/66057).
The experimental code can be found at [Github](https://github.com/qbit-/defend-sc.git)

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

For each model, the benchmark first measures ordinary full-model
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

- Models: Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, and Qwen3.5-4B.
- Dataset: GLUE SST-2 validation prompts.
- Number of prompts: 256.
- Batch size: 4.
- Maximum sequence length: 64.
- Split layer: $k=8$.
- Noise rank: $r=8$.

We do 2 warmup iterations and 5 measured iterations by default.

The plots below show a slowdown ratio and an absolute time for noise addition and suppression.

![Slowdown plot](artifacts/plots/qwen_inference_slowdown_bar.png)

![Absolute time plot](artifacts/plots/qwen_inference_slowdown_noise_suppressor_times.png)