# NLP Learning: From Count-Based Models to Mini LLMs

This repository is a research-style NLP study lab. It is designed for readers who already know Python, linear algebra, probability, machine learning, neural networks, and PyTorch, but want a deeper and more mathematically grounded treatment of Natural Language Processing.

The goal is not just to *use* NLP models, but to understand why they work, how they are optimized, what their failure modes are, and how to implement their core mechanisms from scratch.

Each topic in `theory.md` follows the same research-oriented structure: intuition, mathematical foundations, architecture, scratch implementation, practical implementation, dataset example, visualization, performance analysis, research insight, repository mapping, advanced extensions, and exercises.

## What this repository covers

The material moves in a deliberate sequence:

1. Text preprocessing and tokenization
2. Bag of Words
3. TF-IDF
4. Word2Vec (Skip-gram and CBOW)
5. GloVe embeddings
6. N-gram language models
7. Hidden Markov Models for POS tagging
8. Recurrent Neural Networks
9. LSTM
10. GRU
11. Sequence-to-Sequence models
12. Attention mechanism
13. Transformer architecture
14. BERT
15. GPT
16. Modern LLM training pipeline

## Repository layout

```text
NLP_learning/
│
├── README.md
├── theory.md
├── math_derivation.md
├── implementation_from_scratch.py
├── pytorch_model.py
├── train.py
├── requirements.txt
├── vocab.json
├── dataset/
│   ├── sentiment_toy.json
│   ├── pos_toy.json
│   ├── translation_toy.json
│   └── lm_tiny.txt
└── experiments/
	├── oov_analysis.ipynb
	└── vocab_size_ablation.ipynb
```

## File guide

- `theory.md` — research-level conceptual notes for all major NLP topics.
- `math_derivation.md` — derivations of objectives, probabilities, gradients, and complexity.
- `implementation_from_scratch.py` — NumPy implementations of classical NLP and core neural building blocks.
- `pytorch_model.py` — practical PyTorch versions of RNNs, attention, Transformers, BERT-style MLM, and GPT-style causal LMs.
- `train.py` — small runnable demos on toy datasets.
- `vocab.json` — exported combined vocabulary for the toy corpora.
- `dataset/` — tiny datasets for sentiment classification, POS tagging, translation, and language modeling.

## Quick start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a few demos:

```bash
python train.py --demo scratch-bow --epochs 150
python train.py --demo scratch-word2vec-skipgram --epochs 20
python train.py --demo torch-word2vec --epochs 20
python train.py --demo scratch-hmm
python train.py --demo torch-lstm --epochs 40
python train.py --demo torch-transformer --epochs 40
python train.py --demo torch-gpt --epochs 80
```

## Topic-to-code map

| Topic | Scratch implementation | PyTorch implementation |
|---|---|---|
| Preprocessing + tokenization | `normalize_text`, `tokenize`, `Vocabulary` | data pipeline in `train.py` |
| Bag of Words | `BagOfWordsVectorizer`, `SoftmaxRegressionScratch` | `BOWClassifier` |
| TF-IDF | `TfidfVectorizerScratch` | `TFIDFClassifier` |
| Word2Vec | `Word2VecScratch` | `Word2VecNegativeSampling` |
| GloVe | `GloVeScratch` | concept linked from scratch code |
| N-gram LM | `NGramLanguageModel` | concept linked from scratch code |
| HMM POS tagging | `HMMPOSTagger` | classical model retained in scratch code |
| Vanilla RNN | `RNNClassifierScratch` | `RNNTextClassifier` |
| LSTM | `LSTMCellScratch` | `LSTMTextClassifier` |
| GRU | `GRUCellScratch` | `GRUTextClassifier` |
| Seq2Seq + attention | `AdditiveAttentionScratch` | `Seq2SeqAttention` |
| Attention | `scaled_dot_product_attention` | `scaled_dot_product_attention`, `MultiHeadAttention` |
| Transformer | `MiniTransformerLMScratch` | `MiniTransformerEncoderClassifier` |
| BERT | theory + math notes | `MiniBERTForMLM` |
| GPT | theory + math notes | `MiniGPTLM` |
| LLM training pipeline | theory + math notes | training patterns in `train.py` |

## Learning workflow

For each topic, work in this order:

1. Read the intuition and architecture in `theory.md`.
2. Study the objective and derivations in `math_derivation.md`.
3. Trace the NumPy implementation in `implementation_from_scratch.py`.
4. Compare it against the practical PyTorch version in `pytorch_model.py`.
5. Run the small demo from `train.py`.
6. Modify one design choice and measure the effect.

That loop is the difference between “I recognize this buzzword” and “I can debug a paper implementation at 2am with tea and stubbornness.”

## Research outcomes this repo targets

By the end, you should be able to:

- read modern NLP papers with less intimidation and more curiosity,
- derive and interpret common training objectives,
- implement major NLP components from scratch,
- run controlled toy experiments before scaling up,
- build clean GitHub repositories for NLP research projects,
- understand the design choices behind BERT-, GPT-, and Transformer-style systems.

## Suggested extensions

Once the toy versions are clear, scale toward:

- subword tokenization with BPE or SentencePiece,
- pretrained embeddings and transfer learning,
- encoder-decoder Transformers for translation,
- instruction tuning and preference optimization,
- retrieval-augmented generation,
- efficient attention and long-context modeling,
- distributed training, mixed precision, and checkpoint sharding.

## Notes

- The datasets here are intentionally tiny and pedagogical.
- The scratch implementations prioritize transparency over speed.
- The PyTorch models are small enough to inspect end-to-end.
- For serious research, replace the toy data with larger corpora and add proper experiment tracking.

If you want to turn this into a publication-quality project, the natural next steps are reproducible configs, logging, ablations, and evaluation scripts. In other words: fewer mysteries, more tables.

