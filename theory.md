# NLP Theory Notes: A Research-Style Roadmap

This document is the conceptual spine of the repository. It is written for a reader who already knows linear algebra, probability, machine learning, deep learning, and PyTorch, but wants a mathematically grounded view of NLP as a sequence of modeling assumptions over symbolic data.

The companion file `math_derivation.md` contains denser derivations. The code lives in `implementation_from_scratch.py`, `pytorch_model.py`, and `train.py`.

## How to use this file

For each topic, read the sections in order:

1. build the intuition,
2. connect it to the formal probabilistic or optimization view,
3. inspect the architecture and data flow,
4. trace the scratch implementation,
5. compare against the practical PyTorch implementation,
6. run the toy dataset demo,
7. inspect intermediate representations,
8. reason about metrics and complexity,
9. connect to research,
10. locate the relevant repository files,
11. extend toward modern papers,
12. solve the exercises.

## Common notation

A document is $x = (w_1, \dots, w_T)$, a vocabulary is $V$ with $|V|$ discrete symbols, and a token-id sequence is $(i_1, \dots, i_T)$, where $i_t \in \{1, \dots, |V|\}$. A model with parameters $\theta$ defines either a distribution $p_\theta$ or a score $f_\theta$.

Training usually minimizes empirical risk:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{n=1}^N \ell\big(f_\theta(x^{(n)}), y^{(n)}\big).
$$

For language modeling, perplexity is:

$$
\operatorname{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log p_\theta(w_t \mid w_{<t})\right).
$$

---

## 1. Text preprocessing and tokenization

### 1. INTUITION

Text preprocessing is the interface between messy human language and the rigid algebra of machine learning. Raw strings are not yet objects a model can reason about. Tokenization answers a deceptively deep question: what should count as an atomic unit of meaning?

An analogy: imagine trying to study traffic flow from satellite imagery. Before modeling anything, you need to decide whether the basic object is a pixel, a car, a lane, or an intersection. Tokenization plays the same role for language.

### 2. MATHEMATICAL FOUNDATIONS

Let $\Sigma$ be the character alphabet. A tokenizer is a mapping $\tau : \Sigma^* \to V^*$. A preprocessing pipeline is a composition of transformations:

$$
\tilde{x} = (\phi_m \circ \cdots \circ \phi_1)(x),
$$

where $\phi_k$ may lowercase, normalize punctuation, or collapse whitespace. After tokenization, the empirical unigram distribution is

$$
\hat{p}(w) = \frac{c(w)}{\sum_{v \in V} c(v)}.
$$

Vocabulary truncation induces an out-of-vocabulary rate:

$$
\operatorname{OOV}(V_K) = \frac{\sum_{w \notin V_K} c(w)}{\sum_{w} c(w)}.
$$

Subword tokenization can be seen as an optimization problem trading compression against vocabulary size. In BPE-style methods, merges are chosen greedily to reduce corpus encoding length.

### 3. ARCHITECTURE

The data flow is:

1. raw string,
2. normalization,
3. token segmentation,
4. vocabulary construction,
5. integer encoding,
6. padding and masking for batching.

The computational graph is trivial compared with neural models, but the design choice propagates everywhere: it determines sequence length, sparsity, memory usage, and what linguistic regularities can even be represented.

### 4. IMPLEMENTATION FROM SCRATCH

In `implementation_from_scratch.py`:

- `normalize_text` performs lowercasing and basic regex cleanup.
- `tokenize` converts normalized text into whitespace tokens.
- `Vocabulary.build` constructs token-to-id and id-to-token mappings.
- `ensure_tokenized` and `pad_sequences` prepare downstream models.

This is intentionally simple so you can inspect every assumption. No magic tokenizer goblins are hiding under the floorboards.

### 5. PRACTICAL IMPLEMENTATION

The practical pipeline appears in `train.py`:

- `build_classification_sequences` converts text into token id sequences.
- `build_torch_classification_tensors` pads sequences and builds tensors.
- PyTorch embedding layers in `pytorch_model.py` consume these ids directly.

In production systems, the analogous components are WordPiece, BPE, SentencePiece, or byte-level tokenizers.

### 6. DATASET EXAMPLE

- Sentiment classification uses `dataset/sentiment_toy.json`.
- Translation uses `dataset/translation_toy.json`.
- Language modeling uses `dataset/lm_tiny.txt`.

Try comparing vocabulary sizes and OOV behavior when using whitespace tokens versus a hypothetical byte-level tokenizer.

### 7. VISUALIZATION

Useful diagnostics:

- token frequency histogram,
- cumulative coverage curve versus vocabulary size,
- OOV rate on held-out text,
- sequence length distribution,
- examples of normalization collisions such as `don't -> don't` versus punctuation stripping.

### 8. PERFORMANCE ANALYSIS

- Time complexity is linear in corpus size, $O(L)$, where $L$ is the total number of characters or tokens scanned.
- Memory complexity is $O(|V|)$ for vocabulary storage.
- Key evaluation quantities are vocabulary coverage, OOV rate, average sequence length, and downstream accuracy.

Tokenization changes the effective context length. Finer tokens increase $T$ but reduce OOV; coarser tokens reduce $T$ but explode the vocabulary.

### 9. RESEARCH INSIGHT

Tokenization is an inductive bias, not a mere preprocessing chore. It decides which regularities are easy to learn. Word-level models struggle with morphology and rare words; byte-level models remove OOV but lengthen sequences; subword methods sit in the middle.

Modern research increasingly questions fixed tokenization, especially for multilingual and character-rich languages.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: preprocessing utilities and `Vocabulary`
- `train.py`: tensorization pipeline
- `dataset/`: toy corpora
- `experiments/`: notebooks for vocabulary ablations and OOV analysis

### 11. ADVANCED EXTENSIONS

- Byte Pair Encoding (BPE)
- Unigram language-model tokenization
- SentencePiece for raw text without whitespace assumptions
- Byte-level tokenization for robust multilingual and noisy text processing
- Learned tokenization and tokenizer-free models

### 12. EXERCISES

1. Prove that minimizing OOV rate alone does not minimize sequence length.
2. Implement BPE from scratch and compare average tokenized length with whitespace tokenization.
3. For a morphologically rich language, argue whether word-level or subword-level tokenization induces a better bias.

---

## 2. Bag of Words

### 1. INTUITION

Bag of Words discards word order and keeps only counts. It asks: if I forget syntax and remember only which words occurred, how much can I still predict? Surprisingly, quite a lot for topic classification, sentiment, and retrieval.

The analogy is a grocery receipt. The order of purchased items is gone, but the list still tells you a lot about the household.

### 2. MATHEMATICAL FOUNDATIONS

For document $d$, define the count vector $x_d \in \mathbb{R}^{|V|}$ by

$$
(x_d)_j = c(v_j, d).
$$

A binary BoW model uses $\mathbb{1}[c(v_j,d) > 0]$ instead. For classification with $C$ labels, softmax regression models

$$
p_\theta(y=c \mid x) = \frac{\exp(w_c^\top x + b_c)}{\sum_{c'=1}^C \exp(w_{c'}^\top x + b_{c'})}.
$$

The training objective is cross-entropy with optional $L_2$ regularization:

$$
\mathcal{L}(W,b) = -\frac{1}{N}\sum_{n=1}^N \log p_\theta(y^{(n)} \mid x^{(n)}) + \frac{\lambda}{2}\|W\|_F^2.
$$

### 3. ARCHITECTURE

Pipeline:

1. tokenize text,
2. build vocabulary,
3. map each document to a sparse count vector,
4. feed the vector to a linear classifier,
5. apply softmax for class probabilities.

The computational graph is linear: counts $\to$ logits $\to$ softmax $\to$ loss.

### 4. IMPLEMENTATION FROM SCRATCH

- `BagOfWordsVectorizer` builds the count matrix.
- `SoftmaxRegressionScratch` implements forward pass, softmax, cross-entropy gradients, and SGD updates.
- Demo: `python train.py --demo scratch-bow --epochs 150`

### 5. PRACTICAL IMPLEMENTATION

- `BOWClassifier` in `pytorch_model.py` is a single linear layer.
- Demo: `python train.py --demo torch-bow --epochs 100`

The PyTorch model is structurally the same as the scratch model. The difference is automation of gradient computation and optimization.

### 6. DATASET EXAMPLE

`dataset/sentiment_toy.json` is a small binary sentiment dataset. Positive documents contain words such as `wonderful`, `beautiful`, `great`; negative documents contain `dull`, `messy`, `tedious`, and similar cues.

BoW should separate the classes with high accuracy because the label signal is largely lexical.

### 7. VISUALIZATION

- Inspect the sparse document-term matrix.
- Plot top positive and negative learned weights.
- Highlight misclassified examples and identify missing lexical cues.

The most interpretable artifact is the classifier weight vector. It literally tells you which words push the log-odds up or down.

### 8. PERFORMANCE ANALYSIS

- Loss: cross-entropy.
- Metrics: accuracy, precision, recall, F1.
- Complexity: building the design matrix is $O(\text{nnz})$; dense prediction is $O(N|V|C)$, but sparse implementations are cheaper.

Failure mode: documents with identical bags but different order are indistinguishable.

### 9. RESEARCH INSIGHT

BoW is crude, but it establishes a powerful baseline. Many flashy models have been embarrassed by well-tuned sparse linear models on small and medium-sized datasets.

It works because topical and sentiment signals are often linearly separable in lexical count space.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `BagOfWordsVectorizer`, `SoftmaxRegressionScratch`
- `pytorch_model.py`: `BOWClassifier`
- `train.py`: `scratch-bow`, `torch-bow`
- `dataset/sentiment_toy.json`: toy labels

### 11. ADVANCED EXTENSIONS

- character n-grams,
- feature hashing,
- log-count ratio features as in NB-SVM,
- sparse regularization with $L_1$ penalties,
- calibrated linear classifiers for retrieval and ranking.

### 12. EXERCISES

1. Show that the softmax decision boundary in BoW space is linear.
2. Compare binary BoW and count BoW on the toy sentiment task.
3. Construct two sentences with opposite meanings but identical bag-of-words vectors.

---

## 3. TF-IDF

### 1. INTUITION

Bag of Words treats every token equally. TF-IDF corrects that by down-weighting globally common words and up-weighting locally informative words. The word `the` appears everywhere and says little; the word `cinematography` is rarer and more diagnostic.

### 2. MATHEMATICAL FOUNDATIONS

Term frequency is

$$
\operatorname{tf}(t,d) = \frac{c(t,d)}{\sum_{t'} c(t',d)}.
$$

Document frequency is

$$
\operatorname{df}(t) = \sum_{d} \mathbb{1}[t \in d].
$$

Inverse document frequency with smoothing is

$$
\operatorname{idf}(t) = \log \frac{1 + N}{1 + \operatorname{df}(t)} + 1.
$$

The TF-IDF feature is

$$
x_{t,d} = \operatorname{tf}(t,d)\operatorname{idf}(t).
$$

The downstream linear classifier uses the same cross-entropy objective as BoW.

### 3. ARCHITECTURE

Pipeline:

1. tokenize documents,
2. compute vocabulary and document frequencies,
3. compute normalized TF,
4. multiply by IDF,
5. classify with a linear model.

### 4. IMPLEMENTATION FROM SCRATCH

- `TfidfVectorizerScratch` extends `BagOfWordsVectorizer`.
- It computes smoothed IDF in `fit` and TF-IDF in `transform`.
- Demo: `python train.py --demo scratch-tfidf --epochs 150`

### 5. PRACTICAL IMPLEMENTATION

- `TFIDFClassifier` in `pytorch_model.py` reuses the BoW linear head.
- Demo: `python train.py --demo torch-tfidf --epochs 100`

### 6. DATASET EXAMPLE

Use the same sentiment dataset. Compare the learned weights with the BoW baseline. Common tokens should receive smaller effective influence because their feature values are down-weighted before classification.

### 7. VISUALIZATION

- Compare BoW and TF-IDF matrices for the same document.
- Plot tokens with the largest IDF values.
- Visualize cosine similarity between TF-IDF document vectors.

### 8. PERFORMANCE ANALYSIS

- Loss: cross-entropy after the linear layer.
- Metrics: accuracy and macro-F1.
- Complexity: same order as BoW plus one pass for document frequencies.

TF-IDF is especially effective for retrieval and linear text classification because it approximates how informative a term is for document identity.

### 9. RESEARCH INSIGHT

TF-IDF is not probabilistically pure in the way language models are, but it is a strong heuristic grounded in information content. In IR, it led to BM25 and related term saturation schemes.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `TfidfVectorizerScratch`
- `pytorch_model.py`: `TFIDFClassifier`
- `train.py`: `scratch-tfidf`, `torch-tfidf`

### 11. ADVANCED EXTENSIONS

- BM25,
- pivoted length normalization,
- class-based reweighting,
- dense-sparse hybrid retrieval.

### 12. EXERCISES

1. Derive when TF-IDF reduces exactly to normalized term frequency.
2. Explain why stopword removal is less critical under TF-IDF than under raw counts.
3. Implement BM25 and compare rankings with TF-IDF cosine similarity.

---

## 4. Word2Vec (Skip-gram + CBOW)

### 1. INTUITION

Word2Vec operationalizes the distributional hypothesis: words that occur in similar contexts should have similar representations. Instead of counting contexts explicitly, it learns dense vectors that are predictive of local neighborhoods.

Skip-gram predicts context words from a center word. CBOW predicts the center word from surrounding context.

### 2. MATHEMATICAL FOUNDATIONS

For skip-gram with center word $w_t$ and context word $w_o$, the full softmax objective is

$$
\log p(w_o \mid w_t) = \log \frac{\exp(u_{w_o}^\top v_{w_t})}{\sum_{w \in V} \exp(u_w^\top v_{w_t})}.
$$

The corpus loss sums this over all context windows. Negative sampling replaces the expensive softmax with

$$
\ell = -\log \sigma(u_o^\top v_c) - \sum_{k=1}^K \log \sigma(-u_{n_k}^\top v_c).
$$

CBOW averages or sums context embeddings before predicting the center word. The gradients update two embedding tables: input vectors $v_w$ and output vectors $u_w$.

### 3. ARCHITECTURE

Components:

1. vocabulary,
2. center-context pair generator,
3. input embedding matrix,
4. output embedding matrix,
5. negative sampler,
6. SGD loop.

Data flow for skip-gram: center id $\to$ input embedding $\to$ dot products with positive and negative outputs $\to$ logistic losses.

### 4. IMPLEMENTATION FROM SCRATCH

- `Word2VecScratch` supports `mode="skipgram"` and `mode="cbow"`.
- `_sample_negatives` draws noise words from the smoothed unigram distribution $p(w)^{0.75}$.
- `_skipgram_step` and `_cbow_step` implement the manual gradients.
- Demo: `python train.py --demo scratch-word2vec-skipgram --epochs 20`
- Demo: `python train.py --demo scratch-word2vec-cbow --epochs 20`

### 5. PRACTICAL IMPLEMENTATION

- `Word2VecNegativeSampling` in `pytorch_model.py` implements the same negative-sampling objective with autograd.
- It is easy to extend with larger corpora, mini-batching, and mixed precision if desired.

### 6. DATASET EXAMPLE

`dataset/lm_tiny.txt` is tiny, but enough to show that tokens like `language`, `models`, and `attention` start clustering based on context.

On a real corpus, you would evaluate with similarity, analogy, or downstream initialization.

### 7. VISUALIZATION

- nearest neighbors in cosine space,
- PCA or t-SNE of learned embeddings,
- similarity heatmaps,
- trajectory of one word vector over training epochs.

### 8. PERFORMANCE ANALYSIS

- Objective: negative-sampling logistic loss.
- Metrics: intrinsic similarity, analogy accuracy, downstream transfer performance.
- Complexity: full softmax is $O(|V|)$ per update; negative sampling is $O(K)$.

Word2Vec scales because it swaps exact normalization for cheap contrastive learning.

### 9. RESEARCH INSIGHT

Skip-gram with negative sampling approximately factorizes a shifted PMI matrix. This is one reason the embeddings are semantically meaningful: they encode co-occurrence structure in a low-dimensional geometry.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `Word2VecScratch`
- `pytorch_model.py`: `Word2VecNegativeSampling`
- `train.py`: `scratch-word2vec-skipgram`, `scratch-word2vec-cbow`
- `dataset/lm_tiny.txt`: context corpus

### 11. ADVANCED EXTENSIONS

- hierarchical softmax,
- subsampling frequent words,
- phrase embeddings,
- fastText with subword n-grams,
- contrastive sentence embedding methods.

### 12. EXERCISES

1. Derive the gradient of the negative-sampling loss with respect to the center vector.
2. Explain why the unigram distribution is raised to the power $0.75$.
3. Compare skip-gram and CBOW under a fixed budget of parameter updates.

---

## 5. GloVe embeddings

### 1. INTUITION

Word2Vec learns from predictive local windows. GloVe starts from the global co-occurrence matrix and asks for embeddings whose dot products reconstruct co-occurrence statistics. It is a matrix factorization view of meaning.

### 2. MATHEMATICAL FOUNDATIONS

Let $X_{ij}$ be the weighted co-occurrence count between word $i$ and context word $j$. GloVe fits vectors $w_i, \tilde{w}_j$ and biases $b_i, \tilde{b}_j$ via

$$
J = \sum_{i,j} f(X_{ij})\big(w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2.
$$

The weighting function is typically

$$
f(x) = \begin{cases}
\left(\frac{x}{x_{\max}}\right)^\alpha & x < x_{\max},\\
1 & x \ge x_{\max}.
\end{cases}
$$

The model aims to preserve log-count ratios, which encode semantic relations.

### 3. ARCHITECTURE

Pipeline:

1. build co-occurrence matrix from a context window,
2. initialize word and context vectors,
3. optimize weighted least squares,
4. combine word and context vectors into final embeddings.

### 4. IMPLEMENTATION FROM SCRATCH

- `GloVeScratch` constructs the co-occurrence map.
- It optimizes the weighted least-squares objective with AdaGrad-style accumulators.
- Demo: `python train.py --demo scratch-glove --epochs 50`

### 5. PRACTICAL IMPLEMENTATION

This repository keeps the pedagogical implementation in NumPy because GloVe is already a direct tensorized optimization problem. In larger projects, the exact same objective is usually written in PyTorch for batching and GPU acceleration.

### 6. DATASET EXAMPLE

Use `dataset/lm_tiny.txt`. On such a small corpus, the vectors are noisy, but the training dynamics and weighted reconstruction objective are still visible.

### 7. VISUALIZATION

- co-occurrence heatmap,
- residual histogram $w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij}$,
- nearest-neighbor plots in embedding space.

### 8. PERFORMANCE ANALYSIS

- Objective: weighted squared error.
- Metrics: intrinsic similarity or downstream transfer.
- Complexity: proportional to the number of nonzero co-occurrence entries.

GloVe can be memory-heavy if co-occurrence storage is naive, which is why sparse maps or sharded counting are essential on large corpora.

### 9. RESEARCH INSIGHT

GloVe succeeded because it reconciled count-based and predictive views. It explicitly exploits global corpus statistics while still yielding dense, compositional vectors.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `GloVeScratch`
- `train.py`: `scratch-glove`
- `dataset/lm_tiny.txt`: co-occurrence source corpus

### 11. ADVANCED EXTENSIONS

- subword-aware GloVe,
- dependency-based contexts,
- multilingual aligned embeddings,
- sparse or low-rank co-occurrence approximations.

### 12. EXERCISES

1. Show how GloVe connects to matrix factorization of log co-occurrence counts.
2. Study the effect of window size on the learned embeddings.
3. Compare GloVe and Word2Vec embeddings on a downstream classifier.

---

## 6. N-gram language models

### 1. INTUITION

An n-gram model predicts the next word from only the previous $n-1$ words. It is the simplest probabilistic language model that encodes local order.

You can think of it as a lookup table of short-range habits of a language.

### 2. MATHEMATICAL FOUNDATIONS

By the chain rule,

$$
p(w_{1:T}) = \prod_{t=1}^T p(w_t \mid w_{<t}).
$$

Under the Markov approximation,

$$
p(w_t \mid w_{<t}) \approx p(w_t \mid w_{t-n+1:t-1}).
$$

the maximum-likelihood estimate is

$$
\hat{p}(w \mid h) = \frac{c(h,w)}{c(h)}.
$$

With additive smoothing,

$$
\hat{p}_\alpha(w \mid h) = \frac{c(h,w)+\alpha}{c(h)+\alpha |V|}.
$$

### 3. ARCHITECTURE

Components:

1. sentence boundary markers,
2. context count table,
3. n-gram count table,
4. smoothed conditional probability function,
5. sampling or scoring routine.

Inference is table lookup, not gradient descent.

### 4. IMPLEMENTATION FROM SCRATCH

- `NGramLanguageModel` stores context and n-gram counters.
- `probability`, `sentence_log_probability`, `perplexity`, and `generate` expose the full workflow.
- Demo: `python train.py --demo scratch-ngram`

### 5. PRACTICAL IMPLEMENTATION

For classical n-gram models, the scratch implementation is already the practical one. Neural frameworks are unnecessary unless you are building a neural n-gram generalization.

### 6. DATASET EXAMPLE

Use `dataset/lm_tiny.txt`. Even on a toy corpus, you can compute perplexity and generate short samples.

### 7. VISUALIZATION

- table of the most probable next tokens for a fixed context,
- perplexity versus $n$,
- heatmap of conditional probabilities for selected histories.

### 8. PERFORMANCE ANALYSIS

- Objective: maximum likelihood of observed n-grams.
- Metric: perplexity.
- Complexity: training is linear in the number of observed n-grams; inference is $O(1)$ average for hash-table lookup.

The curse is sparsity: unseen contexts explode combinatorially as $n$ grows.

### 9. RESEARCH INSIGHT

N-gram models are historically foundational because they make the chain rule operational. Their limitations directly motivated neural language models: sparse counts cannot generalize across similar contexts.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `NGramLanguageModel`
- `train.py`: `scratch-ngram`
- `dataset/lm_tiny.txt`: language model corpus

### 11. ADVANCED EXTENSIONS

- Kneser-Ney smoothing,
- backoff and interpolation,
- class-based language models,
- neural n-gram models with embedding lookups and MLPs.

### 12. EXERCISES

1. Derive perplexity from average negative log-likelihood.
2. Explain why Kneser-Ney improves over naive Laplace smoothing.
3. Construct a corpus where a trigram model beats a bigram model and one where it overfits.

---

## 7. Hidden Markov Models for POS tagging

### 1. INTUITION

In an HMM tagger, the POS tags are hidden states and words are emissions. The model assumes language is generated by a latent grammatical process that transitions between tags and emits observed tokens.

### 2. MATHEMATICAL FOUNDATIONS

For tags $z_{1:T}$ and words $x_{1:T}$, the joint factorization is

$$
p(x_{1:T}, z_{1:T}) = p(z_1) p(x_1 \mid z_1) \prod_{t=2}^T p(z_t \mid z_{t-1}) p(x_t \mid z_t).
$$

The model uses start probabilities $p(z_1)$, transition probabilities $p(z_t \mid z_{t-1})$, and emission probabilities $p(x_t \mid z_t)$. Decoding seeks

$$
z_{1:T}^* = \arg\max_{z_{1:T}} p(z_{1:T} \mid x_{1:T}).
$$

Viterbi computes this efficiently with dynamic programming.

### 3. ARCHITECTURE

Training pipeline:

1. count tags, transitions, and emissions,
2. apply smoothing,
3. convert counts to log-probabilities.

Inference pipeline:

1. initialize first-token scores,
2. recursively compute best predecessor for each tag,
3. backtrack to recover the optimal tag sequence.

### 4. IMPLEMENTATION FROM SCRATCH

- `HMMPOSTagger.fit` estimates counts.
- `_start_log_prob`, `_transition_log_prob`, and `_emission_log_prob` produce smoothed log-probabilities.
- `viterbi` performs decoding.
- Demo: `python train.py --demo scratch-hmm`

### 5. PRACTICAL IMPLEMENTATION

For HMMs, the scratch dynamic program is already the practical method. Modern libraries would vectorize the same recurrences, but the algorithmic structure is identical.

### 6. DATASET EXAMPLE

`dataset/pos_toy.json` contains tiny tagged sentences such as `[the, blue, bird, sings]` with tags `[DET, ADJ, NOUN, VERB]`.

### 7. VISUALIZATION

- tag transition matrix,
- emission heatmap for common words,
- Viterbi trellis for one sentence,
- backpointer path.

### 8. PERFORMANCE ANALYSIS

- Objective: maximum likelihood of the joint sequence model.
- Metrics: token accuracy, sentence accuracy.
- Complexity of Viterbi: $O(TK^2)$ where $K$ is the number of tags.

### 9. RESEARCH INSIGHT

HMMs are beautiful because they make modeling assumptions explicit. They fail when those assumptions are too rigid: first-order Markov structure and conditional independence of emissions are linguistically crude.

Still, the HMM-to-CRF-to-neural-sequence-model progression is one of the cleanest narratives in NLP.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `HMMPOSTagger`
- `train.py`: `scratch-hmm`
- `dataset/pos_toy.json`: supervised tag sequences

### 11. ADVANCED EXTENSIONS

- higher-order HMMs,
- trigram taggers,
- CRFs,
- neural emissions with structured decoding,
- hidden semi-Markov models.

### 12. EXERCISES

1. Derive the Viterbi recurrence from the HMM factorization.
2. Compare greedy tagging with Viterbi decoding on the toy dataset.
3. Explain why unknown-word handling is especially important for POS tagging.

---

## 8. Recurrent Neural Networks

### 1. INTUITION

An RNN compresses the past into a hidden state that is updated one token at a time. Instead of storing explicit counts, it learns a continuous summary vector.

The promise is generalization: two different prefixes can map to nearby hidden states if they imply similar future behavior.

### 2. MATHEMATICAL FOUNDATIONS

The hidden recurrence is

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h).
$$

For classification, output logits are

$$
o = W_{hy} h_T + b_y.
$$

Softmax defines label probabilities. Training uses backpropagation through time (BPTT), where gradients flow through all previous hidden states.

Vanishing and exploding gradients arise because repeated Jacobian multiplication either shrinks or amplifies signals across time.

### 3. ARCHITECTURE

Components:

1. embedding lookup,
2. recurrent hidden update,
3. optional pooling or final-state readout,
4. classifier.

For language modeling, the readout occurs at every time step; for classification, often only the final state is used.

### 4. IMPLEMENTATION FROM SCRATCH

- `RNNClassifierScratch` implements forward propagation, manual BPTT, gradient clipping, and SGD.
- Demo: `python train.py --demo scratch-rnn --epochs 20`

### 5. PRACTICAL IMPLEMENTATION

- `RNNTextClassifier` in `pytorch_model.py`
- Demo: `python train.py --demo torch-rnn --epochs 40`

### 6. DATASET EXAMPLE

The toy sentiment dataset works because sentiment cues can often be captured by a sequential summary, even if long-range dependencies are limited.

### 7. VISUALIZATION

- hidden-state norms over time,
- gradient norms per time step,
- saliency over tokens,
- confusion matrix for predicted sentiment.

### 8. PERFORMANCE ANALYSIS

- Loss: cross-entropy.
- Metrics: accuracy, F1.
- Complexity: $O(T H^2)$ for hidden size $H$ if the recurrent matrix dominates.

### 9. RESEARCH INSIGHT

RNNs replaced sparse count tables with differentiable state. That was a conceptual leap: representation learning entered language modeling and sequence classification in earnest.

But plain RNNs struggle with long-term memory, motivating gated variants.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `RNNClassifierScratch`
- `pytorch_model.py`: `RNNTextClassifier`
- `train.py`: `scratch-rnn`, `torch-rnn`

### 11. ADVANCED EXTENSIONS

- bidirectional RNNs,
- stacked RNNs,
- layer normalization,
- recurrent dropout,
- orthogonal or unitary recurrent matrices.

### 12. EXERCISES

1. Derive the BPTT gradient for $W_{hh}$.
2. Experiment with gradient clipping thresholds and report the effect.
3. Construct a synthetic long-range dependency task where the vanilla RNN fails.

---

## 9. LSTM

### 1. INTUITION

The LSTM adds a cell state and gates that regulate information flow. Instead of forcing the network to rewrite its entire memory at every step, it learns when to forget, when to write, and when to expose memory.

### 2. MATHEMATICAL FOUNDATIONS

Given input $x_t$, hidden state $h_{t-1}$, and cell state $c_{t-1}$:

$$
\begin{aligned}
i_t &= \sigma(W_i[x_t;h_{t-1}] + b_i),\\
f_t &= \sigma(W_f[x_t;h_{t-1}] + b_f),\\
o_t &= \sigma(W_o[x_t;h_{t-1}] + b_o),\\
g_t &= \tanh(W_g[x_t;h_{t-1}] + b_g),\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t,\\
h_t &= o_t \odot \tanh(c_t).
\end{aligned}
$$

The additive cell-state path reduces gradient decay compared with a plain RNN.

### 3. ARCHITECTURE

The LSTM cell replaces the vanilla recurrent update. In stacked models, each layer emits a hidden sequence consumed by the next layer.

### 4. IMPLEMENTATION FROM SCRATCH

- `LSTMCellScratch` exposes one-step and sequence forward passes.
- This is useful for inspecting gates before worrying about full training machinery.

### 5. PRACTICAL IMPLEMENTATION

- `LSTMTextClassifier` in `pytorch_model.py`
- Demo: `python train.py --demo torch-lstm --epochs 40`

### 6. DATASET EXAMPLE

Train on the sentiment toy dataset and compare to the vanilla RNN. The difference is usually small on tiny corpora, but the architecture scales much better to longer dependencies.

### 7. VISUALIZATION

- input, forget, and output gate activations over time,
- cell-state trajectory norms,
- tokens that trigger forgetting or retention.

### 8. PERFORMANCE ANALYSIS

- Same classification metrics as RNNs.
- Complexity remains $O(T H^2)$ but with a larger constant due to four gates.

### 9. RESEARCH INSIGHT

LSTMs dominated NLP before Transformers because they preserved sequence order while mitigating long-range optimization problems. They were central to early neural machine translation, speech, and language modeling.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `LSTMCellScratch`
- `pytorch_model.py`: `LSTMTextClassifier`
- `train.py`: `torch-lstm`

### 11. ADVANCED EXTENSIONS

- peephole connections,
- coupled input-forget gates,
- bidirectional LSTMs,
- attention over LSTM hidden states.

### 12. EXERCISES

1. Show why the additive cell path helps gradient transport.
2. Compare gate saturation patterns for easy versus hard examples.
3. Replace $\tanh$ with ReLU in the candidate update and analyze stability.

---

## 10. GRU

### 1. INTUITION

The GRU is a streamlined gated recurrent unit. It merges the cell and hidden state and uses fewer gates than an LSTM while retaining much of the benefit.

### 2. MATHEMATICAL FOUNDATIONS

$$
\begin{aligned}
z_t &= \sigma(W_z[x_t;h_{t-1}] + b_z),\\
r_t &= \sigma(W_r[x_t;h_{t-1}] + b_r),\\
\tilde{h}_t &= \tanh(W_h[x_t; r_t \odot h_{t-1}] + b_h),\\
h_t &= (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t.
\end{aligned}
$$

The update gate interpolates between copying old state and writing new content.

### 3. ARCHITECTURE

GRUs drop the explicit cell state and use a more compact recurrence. This often yields faster training with competitive performance.

### 4. IMPLEMENTATION FROM SCRATCH

- `GRUCellScratch` provides single-step and sequence forward passes.
- Inspecting `update_gate`, `reset_gate`, and `candidate` clarifies the mechanism.

### 5. PRACTICAL IMPLEMENTATION

- `GRUTextClassifier` in `pytorch_model.py`
- Demo: `python train.py --demo torch-gru --epochs 40`

### 6. DATASET EXAMPLE

Again use the sentiment dataset. GRU often matches LSTM quality on small classification tasks with fewer parameters.

### 7. VISUALIZATION

- update gate trajectories,
- reset gate activations around phrase boundaries,
- hidden-state cosine similarity across time.

### 8. PERFORMANCE ANALYSIS

- Loss and metrics as above.
- Complexity is similar to LSTM but with fewer matrix multiplications.

### 9. RESEARCH INSIGHT

GRUs are a reminder that architecture design is often about finding the minimum mechanism needed for stable optimization. Many sequence tasks do not need the full LSTM machinery.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `GRUCellScratch`
- `pytorch_model.py`: `GRUTextClassifier`
- `train.py`: `torch-gru`

### 11. ADVANCED EXTENSIONS

- minimal GRU variants,
- bidirectional GRUs,
- attention-enhanced GRU encoders.

### 12. EXERCISES

1. Compare parameter counts for RNN, LSTM, and GRU at fixed hidden size.
2. Show that the GRU can implement identity memory when $z_t \approx 0$.
3. Design a synthetic task where a reset gate is essential.

---

## 11. Sequence-to-Sequence models

### 1. INTUITION

Sequence-to-sequence learning generalizes classification to structured output. Instead of predicting one label, the model emits an entire target sequence conditioned on an input sequence.

Machine translation is the canonical example: encode the source sentence, then decode the target one token at a time.

### 2. MATHEMATICAL FOUNDATIONS

For source sequence $x$ and target sequence $y=(y_1,\dots,y_T)$:

$$
p_\theta(y \mid x) = \prod_{t=1}^{T} p_\theta(y_t \mid y_{<t}, x).
$$

Training minimizes teacher-forced negative log-likelihood:

$$
\mathcal{L}(\theta) = -\sum_t \log p_\theta(y_t^* \mid y_{<t}^*, x).
$$

### 3. ARCHITECTURE

Components:

1. source embedding,
2. encoder recurrent network,
3. decoder recurrent network,
4. optional attention module,
5. output projection to target vocabulary.

Inference uses greedy decoding or beam search.

### 4. IMPLEMENTATION FROM SCRATCH

The scratch side of this repository exposes the critical mechanism through `AdditiveAttentionScratch`. The full practical seq2seq training loop is provided in PyTorch, where batching and teacher forcing are much easier to express.

### 5. PRACTICAL IMPLEMENTATION

- `Seq2SeqAttention` in `pytorch_model.py`
- Demo: `python train.py --demo torch-seq2seq --epochs 100`

### 6. DATASET EXAMPLE

`dataset/translation_toy.json` contains tiny English-to-French-like phrase pairs such as `i like nlp -> j aime nlp`.

### 7. VISUALIZATION

- decoder token probabilities at each step,
- attention alignment matrix,
- source and target length comparison,
- errors from greedy decoding.

### 8. PERFORMANCE ANALYSIS

- Loss: token-level cross-entropy.
- Metrics: exact match, BLEU on larger datasets, token accuracy on tiny datasets.
- Complexity for recurrent encoder-decoder: $O(T_s H^2 + T_t H^2)$, plus attention if used.

### 9. RESEARCH INSIGHT

Seq2seq reframed translation and generation as conditional language modeling. It was the bridge from recurrent architectures to attention and then to Transformers.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `AdditiveAttentionScratch`
- `pytorch_model.py`: `Seq2SeqAttention`
- `train.py`: `torch-seq2seq`
- `dataset/translation_toy.json`: paired sequences

### 11. ADVANCED EXTENSIONS

- beam search,
- coverage penalties,
- scheduled sampling,
- copy mechanisms,
- non-autoregressive sequence generation.

### 12. EXERCISES

1. Derive the teacher-forced training objective from conditional likelihood.
2. Explain exposure bias and evaluate scheduled sampling as a fix.
3. Compare greedy decoding and beam search on a toy translation task.

---

## 12. Attention mechanism

### 1. INTUITION

Attention lets a model look back selectively instead of compressing the entire past into one fixed-size vector. It is a learned, differentiable retrieval mechanism.

The clean analogy is reading with a highlighter: at each question, you glance back at the relevant lines instead of memorizing the whole page.

### 2. MATHEMATICAL FOUNDATIONS

Additive attention scores encoder state $h_i$ against decoder state $s_t$:

$$
e_{t,i} = v^\top \tanh(W_h h_i + W_s s_t + b).
$$

Attention weights are a simplex distribution:

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}, \qquad \sum_i \alpha_{t,i} = 1.
$$

The context vector is

$$
c_t = \sum_i \alpha_{t,i} h_i.
$$

### 3. ARCHITECTURE

Inputs: encoder states and a query state. Output: a context vector and the weight distribution over source positions.

This is a differentiable content-addressable memory.

### 4. IMPLEMENTATION FROM SCRATCH

- `AdditiveAttentionScratch.forward` computes energies, softmax weights, and context vectors.
- `scaled_dot_product_attention` implements the Transformer-style variant.

### 5. PRACTICAL IMPLEMENTATION

- `BahdanauAttention` in `pytorch_model.py`
- Transformer attention components also live in `scaled_dot_product_attention` and `MultiHeadAttention`

### 6. DATASET EXAMPLE

The seq2seq translation demo is the best place to watch attention align source and target positions.

### 7. VISUALIZATION

- attention heatmaps,
- entropy of the attention distribution,
- how weights move as the decoder advances.

### 8. PERFORMANCE ANALYSIS

- Attention adds roughly $O(T_s T_t H)$ work in encoder-decoder models.
- It improves alignment and gradient flow by removing the fixed-bottleneck constraint.

### 9. RESEARCH INSIGHT

Attention did not just improve performance. It changed the abstraction of sequence modeling from compression to selective interaction. That shift is the seed of the Transformer.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `AdditiveAttentionScratch`, `scaled_dot_product_attention`
- `pytorch_model.py`: `BahdanauAttention`, `MultiHeadAttention`
- `train.py`: `torch-seq2seq`, `torch-transformer`, `torch-bert`, `torch-gpt`

### 11. ADVANCED EXTENSIONS

- multiplicative attention,
- sparse attention,
- monotonic attention,
- retrieval and memory-augmented attention,
- linear-time attention approximations.

### 12. EXERCISES

1. Prove that the attention weights lie on the probability simplex.
2. Compare additive and dot-product attention in terms of cost and expressivity.
3. Explain when attention weights are interpretable and when they are not.

---

## 13. Transformer architecture

### 1. INTUITION

The Transformer discards recurrence and convolution, and instead models a sequence through repeated attention-based interactions between all token positions. Each token asks: which other tokens matter for my representation right now?

This is powerful because communication between distant positions becomes one hop rather than many recurrent steps.

### 2. MATHEMATICAL FOUNDATIONS

For input matrix $X \in \mathbb{R}^{T \times d}$:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V.
$$

Scaled dot-product attention is

$$
\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
$$

where $M$ encodes masks.

Multi-head attention uses several learned projections:

$$
\operatorname{MHA}(X) = \operatorname{Concat}(H_1,\dots,H_h)W_O,
$$

with $H_j = \operatorname{Attn}(Q_j,K_j,V_j)$.

Positional encodings inject order information because attention alone is permutation equivariant.

### 3. ARCHITECTURE

Encoder block:

1. layer norm,
2. multi-head self-attention,
3. residual addition,
4. layer norm,
5. feed-forward MLP,
6. residual addition.

Decoder blocks add causal masking. Full encoder-decoder Transformers also add cross-attention.

### 4. IMPLEMENTATION FROM SCRATCH

- `sinusoidal_positional_encoding`
- `causal_mask`
- `scaled_dot_product_attention`
- `MultiHeadSelfAttentionScratch`
- `FeedForwardScratch`
- `TransformerBlockScratch`
- `MiniTransformerLMScratch`

These components are enough to inspect the full forward pass of a mini language model without outsourcing the math.

### 5. PRACTICAL IMPLEMENTATION

- `MultiHeadAttention`
- `TransformerEncoderBlock`
- `TransformerDecoderBlock`
- `MiniTransformerEncoderClassifier`

Demo: `python train.py --demo torch-transformer --epochs 40`

### 6. DATASET EXAMPLE

The toy sentiment dataset is used for the encoder-style classifier. For generation, GPT-style decoding on `dataset/lm_tiny.txt` demonstrates the autoregressive side.

### 7. VISUALIZATION

- per-head attention maps,
- query-key similarity matrices,
- positional encodings,
- token probability distribution at the next-step prediction.

### 8. PERFORMANCE ANALYSIS

- Loss: cross-entropy.
- Metrics: accuracy for classification, perplexity for language modeling.
- Complexity per layer: self-attention costs $O(T^2 d)$ in time and $O(T^2)$ in memory for attention weights.

This quadratic sequence cost is the main computational bottleneck.

### 9. RESEARCH INSIGHT

Transformers work because they separate representation mixing from sequential computation. Depth replaces recurrence, and attention creates dynamic context-dependent routing.

Their main weaknesses are quadratic context cost, tokenization dependence, and data hunger.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: mini Transformer forward pass
- `pytorch_model.py`: reusable Transformer blocks
- `train.py`: `torch-transformer`
- `math_derivation.md`: full self-attention derivations

### 11. ADVANCED EXTENSIONS

- Pre-LN versus Post-LN variants,
- RoPE and ALiBi positional schemes,
- flash attention,
- sparse and linear attention,
- mixture-of-experts feed-forward layers.

### 12. EXERCISES

1. Derive the complexity of a Transformer layer and compare it with an RNN.
2. Show why self-attention without positions is permutation equivariant.
3. Implement rotary position embeddings and compare with sinusoidal encodings.

---

## 14. BERT

### 1. INTUITION

BERT is a bidirectional encoder pretrained to reconstruct masked tokens from both left and right context. Instead of learning to predict the next word only, it learns contextual representations useful for many downstream tasks.

### 2. MATHEMATICAL FOUNDATIONS

The masked language modeling (MLM) objective masks a subset $M$ of positions and optimizes

$$
\mathcal{L}_{\text{MLM}} = -\sum_{t \in M} \log p_\theta(x_t \mid x_{\setminus M}).
$$

In the original BERT, a next-sentence prediction objective was also added, though later work found it less essential.

### 3. ARCHITECTURE

BERT uses a stack of Transformer encoder blocks:

1. token embeddings,
2. positional embeddings,
3. repeated bidirectional self-attention,
4. MLM head projecting hidden states back to vocabulary logits.

### 4. IMPLEMENTATION FROM SCRATCH

The theoretical scratch decomposition is already present through the Transformer primitives in `implementation_from_scratch.py`. A full BERT trainer is more practical in PyTorch because masked-token batching and loss masking are cleaner there.

### 5. PRACTICAL IMPLEMENTATION

- `MiniBERTForMLM` in `pytorch_model.py`
- `build_mlm_batch` and `train_torch_bert` in `train.py`
- Demo: `python train.py --demo torch-bert --epochs 60`

### 6. DATASET EXAMPLE

`dataset/lm_tiny.txt` is turned into a masked-token prediction dataset by replacing selected tokens with `<mask>` and learning to recover them.

### 7. VISUALIZATION

- masked positions and recovered tokens,
- per-layer attention maps,
- hidden-state similarity across contexts for the same surface word.

### 8. PERFORMANCE ANALYSIS

- Loss: masked cross-entropy over masked positions only.
- Metrics: masked-token accuracy, downstream task transfer.
- Complexity: same quadratic attention cost as Transformer encoders.

### 9. RESEARCH INSIGHT

BERT works because bidirectional conditioning creates rich contextual embeddings. It was a major shift from task-specific architectures to pretrain-then-finetune pipelines.

Limitations include token-level masking mismatch, compute cost, and lack of native generative decoding.

### 10. GITHUB PROJECT STRUCTURE

- `pytorch_model.py`: `MiniBERTForMLM`
- `train.py`: `build_mlm_batch`, `train_torch_bert`
- `dataset/lm_tiny.txt`: MLM corpus

### 11. ADVANCED EXTENSIONS

- RoBERTa,
- ALBERT,
- DeBERTa,
- span masking,
- ELECTRA-style replaced-token detection,
- parameter-efficient fine-tuning such as LoRA.

### 12. EXERCISES

1. Compare MLM with autoregressive LM as pretraining objectives.
2. Analyze why only masked positions contribute to the loss.
3. Implement whole-word masking and study the effect.

---

## 15. GPT architecture

### 1. INTUITION

GPT is an autoregressive Transformer decoder trained to predict the next token. Every hidden state is causal: it may look only to the left. This makes generation straightforward.

### 2. MATHEMATICAL FOUNDATIONS

The objective is causal language modeling:

$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^T \log p_\theta(x_t \mid x_{<t}).
$$

With a causal mask $M$, the self-attention logits to future positions are set to $-\infty$ so the softmax probability is zero there.

### 3. ARCHITECTURE

Decoder-only Transformer:

1. token embeddings,
2. positional encodings,
3. repeated masked self-attention + feed-forward blocks,
4. tied output head projecting back to vocabulary.

Inference repeatedly feeds the growing prefix back through the model.

### 4. IMPLEMENTATION FROM SCRATCH

- `causal_mask`
- `scaled_dot_product_attention`
- `MiniTransformerLMScratch`

Together, these show the causal language-modeling forward pass and greedy generation mechanics.

### 5. PRACTICAL IMPLEMENTATION

- `MiniGPTLM` in `pytorch_model.py`
- `build_causal_lm_batch` and `train_torch_gpt` in `train.py`
- Demo: `python train.py --demo torch-gpt --epochs 80`

### 6. DATASET EXAMPLE

`dataset/lm_tiny.txt` is converted into prefix-target pairs: input tokens are shifted left relative to labels.

### 7. VISUALIZATION

- causal attention maps,
- next-token probability distribution,
- entropy of generated distributions,
- failure cases from greedy decoding.

### 8. PERFORMANCE ANALYSIS

- Loss: next-token cross-entropy.
- Metrics: perplexity, downstream instruction following after fine-tuning in larger systems.
- Complexity: same $O(T^2 d)$ attention bottleneck per layer.

### 9. RESEARCH INSIGHT

GPT-style models excel because next-token prediction is a surprisingly general supervision signal. With enough scale, they learn syntax, semantics, world knowledge, and useful latent structure.

Their limitations include hallucination, context-window cost, and imperfect grounding.

### 10. GITHUB PROJECT STRUCTURE

- `implementation_from_scratch.py`: `MiniTransformerLMScratch`
- `pytorch_model.py`: `MiniGPTLM`
- `train.py`: `build_causal_lm_batch`, `train_torch_gpt`
- `dataset/lm_tiny.txt`: autoregressive corpus

### 11. ADVANCED EXTENSIONS

- rotary embeddings,
- grouped-query attention,
- speculative decoding,
- mixture-of-experts decoders,
- RLHF, DPO, and instruction tuning.

### 12. EXERCISES

1. Derive why causal masking prevents information leakage during training.
2. Compare greedy, top-$k$, and nucleus sampling.
3. Tie and untie embedding weights and analyze parameter count and performance.

---

## 16. Modern LLM training pipeline

### 1. INTUITION

Modern LLM training is not one trick but a pipeline: data curation, tokenization, large-scale pretraining, checkpointing, optimization stabilization, alignment, evaluation, and deployment. The model architecture is only one layer of the system.

### 2. MATHEMATICAL FOUNDATIONS

Pretraining still minimizes token-level negative log-likelihood:

$$
\mathcal{L}_{\text{pretrain}} = -\mathbb{E}_{x \sim \mathcal{D}} \sum_t \log p_\theta(x_t \mid x_{<t}).
$$

But practical training adds:

- distributed gradient averaging,
- mixed-precision numerical stability,
- learning-rate schedules,
- weight decay,
- gradient clipping,
- post-training objectives such as supervised fine-tuning or preference optimization.

Scaling-law discussions model loss as a function of parameters $N$, data $D$, and compute $C$.

### 3. ARCHITECTURE

System pipeline:

1. data ingestion and filtering,
2. deduplication and tokenization,
3. pretraining on autoregressive or masked objectives,
4. evaluation on held-out and benchmark suites,
5. instruction tuning / SFT,
6. alignment or preference optimization,
7. serving with batching, KV cache, and safety layers.

### 4. IMPLEMENTATION FROM SCRATCH

This repository does not attempt to reproduce full industrial LLM training from scratch. Instead, it isolates the mathematical core: tokenization, causal language modeling, attention, masking, and optimization on small corpora.

### 5. PRACTICAL IMPLEMENTATION

- `MiniGPTLM` and `MiniBERTForMLM` show the two dominant pretraining families.
- `train.py` demonstrates tiny-scale MLM and CLM loops.
- The missing industrial pieces are distributed systems, large-scale data, checkpoint orchestration, and alignment stages.

### 6. DATASET EXAMPLE

`dataset/lm_tiny.txt` is intentionally tiny. The point is to understand the loop end-to-end before scaling to web-scale corpora.

### 7. VISUALIZATION

- training loss curves,
- validation perplexity,
- token frequency skew,
- attention entropy,
- generated sample drift over training,
- nearest-neighbor embedding evolution.

### 8. PERFORMANCE ANALYSIS

- Core metric: validation loss or perplexity.
- Secondary metrics: benchmark accuracy, safety metrics, latency, throughput, memory usage.
- Complexity is dominated by Transformer forward/backward passes, optimizer state, and communication overhead in distributed training.

### 9. RESEARCH INSIGHT

Modern LLM progress comes from a three-way interaction: better objectives, larger and cleaner data, and vastly improved optimization systems. Many research papers are really about moving one of these three levers.

### 10. GITHUB PROJECT STRUCTURE

- `theory.md`: conceptual pipeline
- `math_derivation.md`: objective and complexity analysis
- `implementation_from_scratch.py`: pedagogical building blocks
- `pytorch_model.py`: mini encoder and decoder Transformer models
- `train.py`: executable toy pretraining loops

### 11. ADVANCED EXTENSIONS

- instruction tuning,
- direct preference optimization,
- retrieval-augmented generation,
- long-context training,
- multimodal LLMs,
- model compression and distillation,
- synthetic data pipelines,
- inference-time scaling and tool use.

### 12. EXERCISES

1. Design a miniature but principled LLM pretraining recipe for a domain corpus of your choice.
2. Explain why data deduplication matters for both generalization and benchmark contamination.
3. Compare SFT, RLHF, and DPO from an objective-design perspective.

---

## Final outcomes

If you truly work through the notes, code, and experiments, you should be able to:

- read NLP papers without treating equations as decorative wallpaper,
- derive core objectives and debug their gradients,
- implement classical and neural NLP models from scratch,
- reason about why attention and Transformers work,
- build toy BERT- and GPT-style systems end to end,
- structure clean research repositories for NLP experiments,
- scale your understanding toward modern LLM training pipelines.

That is the real course objective: not memorizing model names, but acquiring mechanistic fluency.
