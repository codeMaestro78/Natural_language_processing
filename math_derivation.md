# Mathematical Derivations for NLP Models

This file complements `theory.md`. The goal here is to write the objectives, probability models, gradients, and complexity arguments in a form close to what you would expect in graduate notes or in the appendix of a careful paper.

## Global notation and optimization reminders

Let a dataset be $\mathcal{D} = \{(x^{(n)}, y^{(n)})\}_{n=1}^N$. If a model produces logits $z \in \mathbb{R}^C$, the softmax is

$$
\operatorname{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}.
$$

For one-hot target $y$, the cross-entropy loss is

$$
\ell(z, y) = -\sum_{c=1}^C y_c \log \operatorname{softmax}(z)_c.
$$

A useful identity is

$$
\frac{\partial \ell}{\partial z_c} = \operatorname{softmax}(z)_c - y_c.
$$

This gradient pattern reappears all over NLP: BoW classifiers, RNN decoders, Transformer LMs, BERT MLM heads, and GPT next-token heads all eventually reduce to this template.

---

## 1. Text preprocessing and tokenization

### Formalization

A tokenizer is a measurable map

$$
\tau: \Sigma^* \to V^*,
$$

where $\Sigma$ is a character alphabet and $V$ is a finite symbol inventory. If $x \in \Sigma^*$ is raw text, the tokenized output is

$$
\tau(x) = (w_1, \dots, w_T), \qquad w_t \in V.
$$

The empirical token distribution is

$$
\hat{p}(w) = \frac{c(w)}{\sum_{v \in V} c(v)}.
$$

For a truncated vocabulary $V_K$ of size $K$, the OOV rate is

$$
\operatorname{OOV}(V_K) = 1 - \sum_{w \in V_K} \hat{p}(w).
$$

### Compression view

Subword tokenization can be seen through corpus coding length. If a corpus $x$ is segmented into tokens $w_1, \dots, w_T$, the code length under a token distribution $q$ is

$$
L(x; q, \tau) = -\sum_{t=1}^T \log q(w_t).
$$

BPE greedily alters $\tau$ to reduce this length while maintaining a manageable vocabulary.

### Complexity

- Vocabulary counting: $O(L)$ where $L$ is total corpus length.
- Lookup table memory: $O(|V|)$.
- Tokenization affects every downstream complexity term because it changes sequence length $T$.

---

## 2. Bag of Words

### Feature map

For vocabulary $V = \{v_1, \dots, v_{|V|}\}$ and document $d$, the BoW vector is

$$
\phi(d)_j = c(v_j, d).
$$

The binary variant is

$$
\phi_{\text{bin}}(d)_j = \mathbb{1}[c(v_j, d) > 0].
$$

### Softmax regression objective

Let $W \in \mathbb{R}^{|V| \times C}$ and $b \in \mathbb{R}^C$. Logits are

$$
z = \phi(d)^\top W + b.
$$

Then

$$
p(y=c \mid d) = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}.
$$

The empirical objective with $L_2$ regularization is

$$
\mathcal{L}(W,b) = -\frac{1}{N} \sum_{n=1}^{N} \log p(y^{(n)} \mid d^{(n)}) + \frac{\lambda}{2}\|W\|_F^2.
$$

### Gradients

For a single example with one-hot target $y$:

$$
\frac{\partial \ell}{\partial z} = \hat{y} - y,
$$

where $\hat{y}=\operatorname{softmax}(z)$. Therefore

$$
\frac{\partial \ell}{\partial W} = \phi(d)(\hat{y}-y)^\top + \lambda W,
$$

$$
\frac{\partial \ell}{\partial b} = \hat{y}-y.
$$

These formulas appear explicitly in `SoftmaxRegressionScratch.fit`.

### Complexity

- Building document-term matrix: $O(\text{nnz})$.
- Linear prediction: $O(|V|C)$ dense, or $O(\text{nnz} \cdot C)$ sparse.

---

## 3. TF-IDF

### Definitions

For term $t$ and document $d$:

$$
\operatorname{tf}(t,d) = \frac{c(t,d)}{\sum_{t'} c(t',d)}.
$$

Document frequency:

$$
\operatorname{df}(t) = \sum_{d=1}^{N} \mathbb{1}[t \in d].
$$

Smoothed inverse document frequency:

$$
\operatorname{idf}(t) = \log \frac{1+N}{1+\operatorname{df}(t)} + 1.
$$

Hence

$$
\operatorname{tfidf}(t,d) = \operatorname{tf}(t,d)\operatorname{idf}(t).
$$

### Information-theoretic reading

If $\operatorname{df}(t)$ is high, the term occurs in many documents and thus contributes less information about document identity. The logarithm moderates the penalty, preventing extremely rare terms from dominating too aggressively.

### Complexity

- Need one extra corpus pass to estimate $\operatorname{df}(t)$.
- Downstream gradient formulas for the classifier are identical to BoW because TF-IDF is just a different feature map.

---

## 4. Word2Vec

### Skip-gram with full softmax

For center word $c$ and outside word $o$:

$$
p(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w \in V} \exp(u_w^\top v_c)}.
$$

Loss for one pair:

$$
\ell = -\log p(o \mid c) = -u_o^\top v_c + \log \sum_{w \in V} \exp(u_w^\top v_c).
$$

Gradient with respect to the center vector:

$$
\frac{\partial \ell}{\partial v_c} = -u_o + \sum_{w \in V} p(w \mid c) u_w.
$$

This is expensive because it requires summation over the full vocabulary.

### Negative sampling objective

For negatives $n_1, \dots, n_K$:

$$
\ell_{\text{NS}} = -\log \sigma(u_o^\top v_c) - \sum_{k=1}^{K} \log \sigma(-u_{n_k}^\top v_c).
$$

Useful derivatives:

$$
\frac{d}{dx}[-\log \sigma(x)] = \sigma(x)-1,
$$

$$
\frac{d}{dx}[-\log \sigma(-x)] = \sigma(x).
$$

Therefore

$$
\frac{\partial \ell_{\text{NS}}}{\partial v_c} = (\sigma(u_o^\top v_c)-1)u_o + \sum_{k=1}^{K} \sigma(u_{n_k}^\top v_c)u_{n_k}.
$$

And for the positive output vector,

$$
\frac{\partial \ell_{\text{NS}}}{\partial u_o} = (\sigma(u_o^\top v_c)-1)v_c.
$$

For a negative vector $u_{n_k}$,

$$
\frac{\partial \ell_{\text{NS}}}{\partial u_{n_k}} = \sigma(u_{n_k}^\top v_c)v_c.
$$

These match `_skipgram_step` in the scratch implementation.

### CBOW objective

Let the context set be $C_t$. If

$$
\bar{v}_{C_t} = \frac{1}{|C_t|}\sum_{j \in C_t} v_j,
$$

then CBOW uses the same logistic loss with $\bar{v}_{C_t}$ replacing $v_c$. Each context embedding receives the average gradient contribution.

### PMI connection

Skip-gram with negative sampling can be shown to implicitly factorize a shifted PMI matrix:

$$
u_w^\top v_c \approx \operatorname{PMI}(w,c) - \log K.
$$

This explains why cosine neighborhoods encode semantic similarity.

### Complexity

- Full softmax: $O(|V|d)$ per training pair.
- Negative sampling: $O(Kd)$ per pair.

---

## 5. GloVe

### Objective

Given co-occurrence counts $X_{ij}$, optimize

$$
J = \sum_{i,j} f(X_{ij}) \left(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2.
$$

Define the residual

$$
r_{ij} = w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}.
$$

Then

$$
J = \sum_{i,j} f(X_{ij}) r_{ij}^2.
$$

### Gradients

$$
\frac{\partial J}{\partial w_i} = \sum_j 2 f(X_{ij}) r_{ij} \tilde{w}_j,
$$

$$
\frac{\partial J}{\partial \tilde{w}_j} = \sum_i 2 f(X_{ij}) r_{ij} w_i,
$$

$$
\frac{\partial J}{\partial b_i} = \sum_j 2 f(X_{ij}) r_{ij}, \qquad
\frac{\partial J}{\partial \tilde{b}_j} = \sum_i 2 f(X_{ij}) r_{ij}.
$$

The scratch code performs these updates entry by entry with adaptive learning rates.

### Why log counts?

Raw counts are too skewed. Taking $\log X_{ij}$ linearizes multiplicative frequency differences and makes relational structure more accessible. Much of the semantic story in GloVe comes from ratios like

$$
\frac{P(k \mid i)}{P(k \mid j)},
$$

which are captured by differences between dot products.

### Complexity

The cost is proportional to the number of nonzero co-occurrence entries, not $|V|^2$, assuming sparse storage.

---

## 6. N-gram language models

### Maximum-likelihood estimation

Under the $(n-1)$-order Markov assumption,

$$
p(w_t \mid w_{<t}) \approx p(w_t \mid h_t), \qquad h_t = w_{t-n+1:t-1}.
$$

MLE gives

$$
\hat{p}(w \mid h) = \frac{c(h,w)}{c(h)}.
$$

### Additive smoothing

To avoid zero probabilities,

$$
\hat{p}_\alpha(w \mid h) = \frac{c(h,w)+\alpha}{c(h)+\alpha |V|}.
$$

### Perplexity derivation

If a test corpus has total token count $T$ and log-likelihood $\log p(w_{1:T})$, average negative log-likelihood is

$$
-\frac{1}{T} \log p(w_{1:T}).
$$

Exponentiating yields perplexity:

$$
\operatorname{PPL} = \exp\left(-\frac{1}{T}\log p(w_{1:T})\right).
$$

Thus perplexity is the effective branching factor of the model.

### Complexity

- Estimation: one pass over all observed n-grams.
- Querying: hash-table lookup plus smoothing arithmetic.

---

## 7. Hidden Markov Models for POS tagging

### Probability model

Let observations be $x_{1:T}$ and hidden tags $z_{1:T}$. The HMM assumes

$$
p(x_{1:T}, z_{1:T}) = p(z_1) p(x_1\mid z_1) \prod_{t=2}^{T} p(z_t\mid z_{t-1}) p(x_t\mid z_t).
$$

This implies

$$
\log p(x_{1:T}, z_{1:T}) = \log p(z_1) + \log p(x_1\mid z_1) + \sum_{t=2}^{T} \left[\log p(z_t\mid z_{t-1}) + \log p(x_t\mid z_t)\right].
$$

### Smoothed estimators

With additive smoothing $\alpha$:

$$
\hat{p}(z_1 = k) = \frac{c_{\text{start}}(k)+\alpha}{N + \alpha K},
$$

$$
\hat{p}(z_t=j \mid z_{t-1}=i) = \frac{c(i,j)+\alpha}{c(i)+\alpha K},
$$

$$
\hat{p}(x=w \mid z=k) = \frac{c(k,w)+\alpha}{c(k)+\alpha(|V|+1)}.
$$

### Viterbi derivation

Define

$$
\delta_t(j) = \max_{z_{1:t-1}} \log p(z_{1:t-1}, z_t=j, x_{1:t}).
$$

Initialization:

$$
\delta_1(j) = \log p(z_1=j) + \log p(x_1\mid z_1=j).
$$

Recurrence:

$$
\delta_t(j) = \max_i \left[\delta_{t-1}(i) + \log p(z_t=j\mid z_{t-1}=i)\right] + \log p(x_t\mid z_t=j).
$$

Backpointers store the maximizing $i$ at each step.

### Complexity

$O(TK^2)$ time, $O(TK)$ memory with backpointers.

---

## 8. Recurrent Neural Networks

### Forward equations

Given one-hot or embedded input $x_t$:

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h),
$$

$$
z = W_{hy}h_T + b_y.
$$

### Backpropagation through time

Suppose the loss depends on the final state only. Then

$$
\frac{\partial \ell}{\partial h_T} = W_{hy}^\top \frac{\partial \ell}{\partial z}.
$$

For $t<T$,

$$
\frac{\partial \ell}{\partial h_t} = \left(W_{hh}^\top \operatorname{diag}(1-h_{t+1}^2)\right) \frac{\partial \ell}{\partial h_{t+1}}.
$$

This repeated multiplication explains vanishing/exploding gradients: the norm is controlled by the spectral properties of the recurrent Jacobian.

### Parameter gradients

At time $t$, define

$$
\delta_t = (1-h_t^2) \odot \frac{\partial \ell}{\partial h_t}.
$$

Then

$$
\frac{\partial \ell}{\partial W_{xh}} = \sum_t \delta_t x_t^\top,
$$

$$
\frac{\partial \ell}{\partial W_{hh}} = \sum_t \delta_t h_{t-1}^\top,
$$

$$
\frac{\partial \ell}{\partial b_h} = \sum_t \delta_t.
$$

This is exactly the logic mirrored in `RNNClassifierScratch.backward`.

### Complexity

One recurrent layer costs roughly $O(TH^2)$ plus embedding lookup and output projection.

---

## 9. LSTM

### Gate equations

Let $a_t = [x_t; h_{t-1}]$. Then

$$
i_t = \sigma(W_i a_t + b_i), \quad
f_t = \sigma(W_f a_t + b_f), \quad
o_t = \sigma(W_o a_t + b_o), \quad
g_t = \tanh(W_g a_t + b_g).
$$

State updates:

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t,
$$

$$
h_t = o_t \odot \tanh(c_t).
$$

### Why gradients survive better

Because

$$
\frac{\partial c_t}{\partial c_{t-1}} = f_t,
$$

and $f_t$ can stay near $1$, the model can preserve a nearly identity gradient path through time. Compare this to the plain RNN, where repeated multiplication by $W_{hh}$ is unavoidable.

### Parameter count

If the input dimension is $d$ and hidden size is $H$, then one LSTM layer uses roughly

$$
4H(d+H) + 4H
$$

parameters.

### Complexity

Same $O(TH^2)$ order as an RNN, but with about four times the gate-related affine work.

---

## 10. GRU

### Equations

$$
z_t = \sigma(W_z[x_t;h_{t-1}] + b_z),
$$

$$
r_t = \sigma(W_r[x_t;h_{t-1}] + b_r),
$$

$$
\tilde{h}_t = \tanh(W_h[x_t; r_t \odot h_{t-1}] + b_h),
$$

$$
h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t.
$$

### Gradient intuition

The direct term $(1-z_t) \odot h_{t-1}$ provides a skip-like path. If $z_t$ is small, the old state is copied almost unchanged. This creates a built-in interpolation between memory preservation and content overwrite.

### Parameter count

For input dimension $d$ and hidden size $H$:

$$
3H(d+H) + 3H.
$$

So GRUs are lighter than LSTMs while still gated.

---

## 11. Sequence-to-Sequence models

### Conditional likelihood

For source sequence $x$ and target sequence $y=(y_1,\dots,y_T)$,

$$
p_\theta(y \mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid y_{<t}, x).
$$

Teacher-forced training uses the gold history $y_{<t}^*$:

$$
\mathcal{L}(\theta) = -\sum_{(x,y) \in \mathcal{D}} \sum_{t=1}^{T} \log p_\theta(y_t^* \mid y_{<t}^*, x).
$$

### Exposure bias

At training time, the decoder conditions on gold history; at test time, it conditions on its own predictions. Formally, the state distribution at inference differs from the state distribution under teacher forcing. Scheduled sampling tries to narrow this gap, though it introduces bias.

### Complexity

If encoder and decoder are recurrent with hidden size $H$, the dominant cost is approximately $O(T_s H^2 + T_t H^2)$, plus attention if present.

---

## 12. Attention mechanism

### Additive attention

Scores:

$$
e_{t,i} = v^\top \tanh(W_h h_i + W_s s_t + b).
$$

Weights:

$$
\alpha_{t,i} = \frac{e^{e_{t,i}}}{\sum_j e^{e_{t,j}}}.
$$

Context:

$$
c_t = \sum_i \alpha_{t,i} h_i.
$$

### Softmax Jacobian

For attention logits $e$ and weights $\alpha = \operatorname{softmax}(e)$,

$$
\frac{\partial \alpha_i}{\partial e_j} = \alpha_i (\delta_{ij} - \alpha_j).
$$

This means increasing one score redistributes probability mass globally over the whole simplex.

### Dot-product attention

With query $q_i$, keys $k_j$, values $v_j$:

$$
\alpha_{ij} = \operatorname{softmax}_j(q_i^\top k_j),
$$

$$
o_i = \sum_j \alpha_{ij} v_j.
$$

### Complexity

For all-pairs attention over length $T$ and width $d$, the score matrix costs $O(T^2 d)$ to compute and $O(T^2)$ to store.

---

## 13. Transformer architecture

### Scaled dot-product attention

For matrices $Q \in \mathbb{R}^{T_q \times d_k}$, $K \in \mathbb{R}^{T_k \times d_k}$, $V \in \mathbb{R}^{T_k \times d_v}$:

$$
\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.
$$

The factor $1/\sqrt{d_k}$ stabilizes variance. If query/key components are mean-zero with unit variance, then $q^\top k$ has variance proportional to $d_k$; dividing by $\sqrt{d_k}$ keeps logits in a scale where softmax gradients do not saturate too early.

### Multi-head attention

For head $h$:

$$
Q_h = XW_h^Q, \quad K_h = XW_h^K, \quad V_h = XW_h^V,
$$

$$
H_h = \operatorname{Attn}(Q_h, K_h, V_h).
$$

Then

$$
\operatorname{MHA}(X) = \operatorname{Concat}(H_1, \dots, H_H)W^O.
$$

This allows different heads to attend to different subspaces and interaction patterns.

### Feed-forward block

Applied position-wise:

$$
\operatorname{FFN}(x) = W_2 \phi(W_1 x + b_1) + b_2,
$$

where $\phi$ is often ReLU or GELU.

### Positional encodings

For sinusoidal encoding,

$$
\operatorname{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right),
$$

$$
\operatorname{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right).
$$

These provide a deterministic basis from which relative offsets can be approximately recovered by linear operations.

### Complexity

For sequence length $T$, model width $d$, number of heads $H$:

- QKV projections: $O(Td^2)$,
- attention scores and weighted sum: $O(T^2 d)$,
- feed-forward: $O(T d d_{ff})$.

For long contexts, the $T^2$ term dominates.

### Causal masking

A decoder mask uses

$$
M_{ij} = \begin{cases}
0 & j \le i,\\
-\infty & j > i.
\end{cases}
$$

Then the softmax assigns zero probability to future positions.

---

## 14. BERT

### MLM objective

Given masked positions $M$,

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in M} \log p_\theta(x_t \mid x_{\setminus M}).
$$

Only masked positions contribute to the loss. If logits at position $t$ are $z_t$, then the local loss is

$$
\ell_t = -\log \operatorname{softmax}(z_t)_{x_t}.
$$

Thus

$$
\frac{\partial \ell_t}{\partial z_t} = \hat{y}_t - y_t.
$$

Unmasked positions still matter because they influence contextual hidden states that predict masked positions.

### Bidirectionality

The encoder attention mask allows tokens to attend to both left and right context, unlike GPT-style causal masks. This makes hidden states more suitable as contextual features for classification and tagging.

### Complexity

Same as an encoder Transformer: each layer is $O(T^2 d)$ in time and $O(T^2)$ in attention memory.

---

## 15. GPT architecture

### Autoregressive objective

For a sequence $x_{1:T}$,

$$
p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}),
$$

and the negative log-likelihood is

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).
$$

If logits at step $t$ are $z_t$, then the contribution is

$$
\ell_t = -\log \operatorname{softmax}(z_t)_{x_t}.
$$

### Generation

At inference, the model samples or selects

$$
\hat{x}_{t+1} \sim p_\theta(\cdot \mid \hat{x}_{\le t}).
$$

The sequence is extended iteratively. Greedy decoding selects $\arg\max$ at each step; top-$k$ or nucleus sampling truncates the support before sampling.

### KV-cache insight

During generation, re-computing keys and values for previous tokens is wasteful. KV caching stores them so each new step attends over cached past states plus the new token. This changes per-step complexity from reprocessing the full prefix to incremental updates, though total context attention is still linear in generated length per new token.

---

## 16. Modern LLM training pipeline

### Pretraining objective

The dominant objective is still token-level cross-entropy:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).
$$

### Optimization at scale

For distributed training with data parallelism, if worker gradients are $g_1, \dots, g_m$, the averaged gradient is

$$
\bar{g} = \frac{1}{m} \sum_{i=1}^{m} g_i.
$$

This preserves the expectation of the minibatch gradient while reducing variance as batch size grows.

A typical AdamW update is

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta \lambda \theta_{t-1}.
$$

### Compute and scaling

For a dense decoder Transformer, rough FLOP counts scale like

$$
\text{FLOPs} \propto L \cdot T \cdot d^2 + L \cdot T^2 \cdot d,
$$

where $L$ is number of layers. For typical LLM regimes, both width-dependent and attention-dependent costs matter, with the $T^2$ term dominating at long context lengths.

### Preference optimization sketch

If preference data says response $y^+$ is preferred over $y^-$ for prompt $x$, DPO-style objectives optimize a comparison of log-probability ratios instead of token likelihood alone. One simplified form is

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y^+\mid x)}{\pi_{\text{ref}}(y^+\mid x)} - \log \frac{\pi_\theta(y^-\mid x)}{\pi_{\text{ref}}(y^-\mid x)}\right]\right).
$$

This shows how post-training objectives are layered on top of the base language-model objective.

### Why the pipeline matters

The architecture provides capacity, the data provides signal, and optimization provides a path through parameter space. Large-scale NLP progress depends on all three.

---

## Suggested reading path

1. Start with Sections 2, 4, 6, and 7 to ground classical NLP.
2. Move to Sections 8 through 12 for sequence modeling.
3. Study Sections 13 through 16 carefully for modern Transformer and LLM work.
4. Then revisit `implementation_from_scratch.py` and derive the gradients yourself before trusting any library call.

That exercise is where research intuition starts to feel less mystical and more mechanical.
