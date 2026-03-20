"""Research-oriented NLP implementations from scratch using NumPy.

This module is intentionally educational: it implements the main ideas behind
classical and neural NLP models with minimal dependencies so that the math and
data flow remain visible.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import numpy as np


EPS = 1e-12
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def set_seed(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)


def normalize_text(text: str) -> str:
	text = text.lower().strip()
	text = re.sub(r"[^a-z0-9\s']", " ", text)
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def tokenize(text: str) -> List[str]:
	normalized = normalize_text(text)
	return normalized.split() if normalized else []


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
	shifted = x - np.max(x, axis=axis, keepdims=True)
	exp_x = np.exp(shifted)
	return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + EPS)


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
	x = np.clip(x, -50.0, 50.0)
	return 1.0 / (1.0 + np.exp(-x))


def ensure_tokenized(documents: Sequence[str | Sequence[str]]) -> List[List[str]]:
	tokenized: List[List[str]] = []
	for document in documents:
		if isinstance(document, str):
			tokenized.append(tokenize(document))
		else:
			cleaned = [normalize_text(str(token)) for token in document]
			tokenized.append([token for token in cleaned if token])
	return tokenized


def pad_sequences(
	sequences: Sequence[Sequence[int]],
	pad_value: int = 0,
	max_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	if not sequences:
		return np.zeros((0, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)
	lengths = np.array([len(sequence) for sequence in sequences], dtype=np.int64)
	target_length = max_length or int(lengths.max())
	padded = np.full((len(sequences), target_length), pad_value, dtype=np.int64)
	for row, sequence in enumerate(sequences):
		length = min(len(sequence), target_length)
		padded[row, :length] = sequence[:length]
	return padded, lengths


@dataclass
class Vocabulary:
	token_to_id: Dict[str, int]
	id_to_token: List[str]
	pad_token: str = "<pad>"
	unk_token: str = "<unk>"
	bos_token: str = "<bos>"
	eos_token: str = "<eos>"

	@classmethod
	def build(
		cls,
		tokenized_corpus: Sequence[Sequence[str]],
		min_freq: int = 1,
		max_size: Optional[int] = None,
		specials: Optional[Sequence[str]] = None,
	) -> "Vocabulary":
		specials = list(specials or SPECIAL_TOKENS)
		counts = Counter(token for sentence in tokenized_corpus for token in sentence)
		tokens = [
			token
			for token, count in counts.most_common()
			if count >= min_freq and token not in specials
		]
		if max_size is not None:
			tokens = tokens[: max(0, max_size - len(specials))]
		id_to_token = specials + tokens
		token_to_id = {token: index for index, token in enumerate(id_to_token)}
		return cls(token_to_id=token_to_id, id_to_token=id_to_token)

	@property
	def pad_id(self) -> int:
		return self.token_to_id.get(self.pad_token, 0)

	@property
	def unk_id(self) -> int:
		return self.token_to_id.get(self.unk_token, 0)

	@property
	def bos_id(self) -> Optional[int]:
		return self.token_to_id.get(self.bos_token)

	@property
	def eos_id(self) -> Optional[int]:
		return self.token_to_id.get(self.eos_token)

	def encode(
		self,
		tokens: Sequence[str],
		add_bos: bool = False,
		add_eos: bool = False,
	) -> List[int]:
		sequence: List[int] = []
		if add_bos and self.bos_id is not None:
			sequence.append(self.bos_id)
		sequence.extend(self.token_to_id.get(token, self.unk_id) for token in tokens)
		if add_eos and self.eos_id is not None:
			sequence.append(self.eos_id)
		return sequence

	def decode(self, token_ids: Sequence[int], skip_specials: bool = True) -> List[str]:
		specials = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
		decoded: List[str] = []
		for token_id in token_ids:
			token = self.id_to_token[int(token_id)]
			if skip_specials and token in specials:
				continue
			decoded.append(token)
		return decoded

	def __len__(self) -> int:
		return len(self.id_to_token)


class BagOfWordsVectorizer:
	def __init__(self, max_features: Optional[int] = None, min_freq: int = 1, binary: bool = False):
		self.max_features = max_features
		self.min_freq = min_freq
		self.binary = binary
		self.vocabulary_: Dict[str, int] = {}

	def fit(self, documents: Sequence[str | Sequence[str]]) -> "BagOfWordsVectorizer":
		tokenized = ensure_tokenized(documents)
		counts = Counter(token for document in tokenized for token in document)
		tokens = [token for token, count in counts.most_common() if count >= self.min_freq]
		if self.max_features is not None:
			tokens = tokens[: self.max_features]
		self.vocabulary_ = {token: index for index, token in enumerate(tokens)}
		return self

	def transform(self, documents: Sequence[str | Sequence[str]]) -> np.ndarray:
		tokenized = ensure_tokenized(documents)
		matrix = np.zeros((len(tokenized), len(self.vocabulary_)), dtype=np.float64)
		for row, document in enumerate(tokenized):
			counts = Counter(document)
			for token, count in counts.items():
				column = self.vocabulary_.get(token)
				if column is not None:
					matrix[row, column] = 1.0 if self.binary else float(count)
		return matrix

	def fit_transform(self, documents: Sequence[str | Sequence[str]]) -> np.ndarray:
		return self.fit(documents).transform(documents)


class TfidfVectorizerScratch(BagOfWordsVectorizer):
	def __init__(self, max_features: Optional[int] = None, min_freq: int = 1):
		super().__init__(max_features=max_features, min_freq=min_freq, binary=False)
		self.idf_: Optional[np.ndarray] = None

	def fit(self, documents: Sequence[str | Sequence[str]]) -> "TfidfVectorizerScratch":
		tokenized = ensure_tokenized(documents)
		super().fit(tokenized)
		num_docs = len(tokenized)
		df = np.zeros(len(self.vocabulary_), dtype=np.float64)
		for document in tokenized:
			seen = {self.vocabulary_[token] for token in set(document) if token in self.vocabulary_}
			for index in seen:
				df[index] += 1.0
		self.idf_ = np.log((1.0 + num_docs) / (1.0 + df)) + 1.0
		return self

	def transform(self, documents: Sequence[str | Sequence[str]]) -> np.ndarray:
		counts = super().transform(documents)
		tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
		if self.idf_ is None:
			raise ValueError("Vectorizer must be fitted before calling transform().")
		return tf * self.idf_


class SoftmaxRegressionScratch:
	def __init__(
		self,
		input_dim: int,
		num_classes: int,
		learning_rate: float = 0.1,
		l2: float = 1e-4,
		seed: int = 42,
	):
		rng = np.random.default_rng(seed)
		self.weights = rng.normal(0.0, 0.01, size=(input_dim, num_classes))
		self.bias = np.zeros(num_classes, dtype=np.float64)
		self.learning_rate = learning_rate
		self.l2 = l2

	def forward(self, features: np.ndarray) -> np.ndarray:
		return features @ self.weights + self.bias

	def predict_proba(self, features: np.ndarray) -> np.ndarray:
		return softmax(self.forward(features), axis=1)

	def predict(self, features: np.ndarray) -> np.ndarray:
		return np.argmax(self.predict_proba(features), axis=1)

	def fit(
		self,
		features: np.ndarray,
		labels: Sequence[int],
		epochs: int = 200,
		batch_size: int = 8,
	) -> List[float]:
		labels_array = np.asarray(labels, dtype=np.int64)
		num_examples = features.shape[0]
		history: List[float] = []
		identity = np.eye(self.bias.shape[0], dtype=np.float64)

		for _ in range(epochs):
			indices = np.arange(num_examples)
			np.random.shuffle(indices)
			epoch_loss = 0.0
			num_batches = 0

			for start in range(0, num_examples, batch_size):
				batch_indices = indices[start : start + batch_size]
				batch_x = features[batch_indices]
				batch_y = labels_array[batch_indices]
				logits = self.forward(batch_x)
				probabilities = softmax(logits, axis=1)
				targets = identity[batch_y]

				loss = -np.mean(np.sum(targets * np.log(probabilities + EPS), axis=1))
				loss += 0.5 * self.l2 * np.sum(self.weights * self.weights)
				epoch_loss += float(loss)
				num_batches += 1

				grad_logits = (probabilities - targets) / len(batch_indices)
				grad_weights = batch_x.T @ grad_logits + self.l2 * self.weights
				grad_bias = grad_logits.sum(axis=0)

				self.weights -= self.learning_rate * grad_weights
				self.bias -= self.learning_rate * grad_bias

			history.append(epoch_loss / max(1, num_batches))
		return history


class Word2VecScratch:
	def __init__(
		self,
		mode: str = "skipgram",
		embed_dim: int = 50,
		window_size: int = 2,
		negative_samples: int = 5,
		learning_rate: float = 0.025,
		epochs: int = 20,
		min_freq: int = 1,
		max_vocab_size: Optional[int] = None,
		seed: int = 42,
	):
		if mode not in {"skipgram", "cbow"}:
			raise ValueError("mode must be 'skipgram' or 'cbow'")
		self.mode = mode
		self.embed_dim = embed_dim
		self.window_size = window_size
		self.negative_samples = negative_samples
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.min_freq = min_freq
		self.max_vocab_size = max_vocab_size
		self.rng = np.random.default_rng(seed)
		self.vocab: Optional[Vocabulary] = None
		self.input_embeddings: Optional[np.ndarray] = None
		self.output_embeddings: Optional[np.ndarray] = None
		self.sampling_distribution: Optional[np.ndarray] = None

	def _build_distribution(self, tokenized_corpus: Sequence[Sequence[str]]) -> np.ndarray:
		if self.vocab is None:
			raise ValueError("Vocabulary must be initialized before building a distribution.")
		counts = np.zeros(len(self.vocab), dtype=np.float64)
		for sentence in tokenized_corpus:
			for token in sentence:
				counts[self.vocab.token_to_id.get(token, self.vocab.unk_id)] += 1.0
		counts = np.power(np.maximum(counts, EPS), 0.75)
		return counts / np.sum(counts)

	def _sample_negatives(self, excluded: set[int]) -> List[int]:
		if self.sampling_distribution is None:
			raise ValueError("Sampling distribution is not initialized.")
		negatives: List[int] = []
		while len(negatives) < self.negative_samples:
			candidate = int(self.rng.choice(len(self.sampling_distribution), p=self.sampling_distribution))
			if candidate not in excluded:
				negatives.append(candidate)
		return negatives

	def _skipgram_step(self, center_id: int, target_id: int, negative_ids: Sequence[int]) -> float:
		assert self.input_embeddings is not None and self.output_embeddings is not None
		center = self.input_embeddings[center_id].copy()
		target = self.output_embeddings[target_id].copy()
		negatives = self.output_embeddings[list(negative_ids)].copy()

		positive_score = float(sigmoid(np.dot(target, center)))
		negative_scores = np.asarray(sigmoid(negatives @ center), dtype=np.float64)

		grad_center = (positive_score - 1.0) * target
		grad_center += np.sum(negative_scores[:, None] * negatives, axis=0)

		self.input_embeddings[center_id] -= self.learning_rate * grad_center
		self.output_embeddings[target_id] -= self.learning_rate * ((positive_score - 1.0) * center)
		self.output_embeddings[list(negative_ids)] -= self.learning_rate * (
			negative_scores[:, None] * center[None, :]
		)

		return -math.log(positive_score + EPS) - float(np.sum(np.log(1.0 - negative_scores + EPS)))

	def _cbow_step(self, context_ids: Sequence[int], target_id: int, negative_ids: Sequence[int]) -> float:
		assert self.input_embeddings is not None and self.output_embeddings is not None
		context_vector = np.mean(self.input_embeddings[list(context_ids)], axis=0)
		target = self.output_embeddings[target_id].copy()
		negatives = self.output_embeddings[list(negative_ids)].copy()

		positive_score = float(sigmoid(np.dot(target, context_vector)))
		negative_scores = np.asarray(sigmoid(negatives @ context_vector), dtype=np.float64)

		grad_context = (positive_score - 1.0) * target
		grad_context += np.sum(negative_scores[:, None] * negatives, axis=0)

		for context_id in context_ids:
			self.input_embeddings[context_id] -= self.learning_rate * grad_context / len(context_ids)

		self.output_embeddings[target_id] -= self.learning_rate * ((positive_score - 1.0) * context_vector)
		self.output_embeddings[list(negative_ids)] -= self.learning_rate * (
			negative_scores[:, None] * context_vector[None, :]
		)

		return -math.log(positive_score + EPS) - float(np.sum(np.log(1.0 - negative_scores + EPS)))

	def fit(self, tokenized_corpus: Sequence[Sequence[str]]) -> List[float]:
		self.vocab = Vocabulary.build(
			tokenized_corpus,
			min_freq=self.min_freq,
			max_size=self.max_vocab_size,
		)
		self.input_embeddings = self.rng.normal(0.0, 0.1, size=(len(self.vocab), self.embed_dim))
		self.output_embeddings = np.zeros((len(self.vocab), self.embed_dim), dtype=np.float64)
		self.sampling_distribution = self._build_distribution(tokenized_corpus)

		history: List[float] = []
		encoded_corpus = [self.vocab.encode(sentence) for sentence in tokenized_corpus]
		for _ in range(self.epochs):
			total_loss = 0.0
			num_steps = 0
			for sentence in encoded_corpus:
				for center_index, center_id in enumerate(sentence):
					left = max(0, center_index - self.window_size)
					right = min(len(sentence), center_index + self.window_size + 1)
					context_ids = [sentence[index] for index in range(left, right) if index != center_index]
					if not context_ids:
						continue
					if self.mode == "skipgram":
						for target_id in context_ids:
							negatives = self._sample_negatives({center_id, target_id})
							total_loss += self._skipgram_step(center_id, target_id, negatives)
							num_steps += 1
					else:
						negatives = self._sample_negatives(set(context_ids) | {center_id})
						total_loss += self._cbow_step(context_ids, center_id, negatives)
						num_steps += 1
			history.append(total_loss / max(1, num_steps))
		return history

	def embeddings(self) -> np.ndarray:
		if self.input_embeddings is None or self.output_embeddings is None:
			raise ValueError("Model must be fitted before requesting embeddings.")
		return self.input_embeddings + self.output_embeddings

	def most_similar(self, token: str, top_k: int = 5) -> List[Tuple[str, float]]:
		if self.vocab is None:
			raise ValueError("Model must be fitted before querying neighbors.")
		token_id = self.vocab.token_to_id.get(token)
		if token_id is None:
			raise KeyError(f"Unknown token: {token}")
		vectors = self.embeddings()
		query = vectors[token_id]
		norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
		similarities = (vectors @ query) / np.maximum(norms, EPS)
		neighbors = np.argsort(-similarities)
		results: List[Tuple[str, float]] = []
		for neighbor_id in neighbors:
			if neighbor_id == token_id:
				continue
			results.append((self.vocab.id_to_token[int(neighbor_id)], float(similarities[neighbor_id])))
			if len(results) == top_k:
				break
		return results


class GloVeScratch:
	def __init__(
		self,
		embed_dim: int = 50,
		window_size: int = 4,
		learning_rate: float = 0.05,
		x_max: float = 100.0,
		alpha: float = 0.75,
		epochs: int = 50,
		seed: int = 42,
	):
		self.embed_dim = embed_dim
		self.window_size = window_size
		self.learning_rate = learning_rate
		self.x_max = x_max
		self.alpha = alpha
		self.epochs = epochs
		self.rng = np.random.default_rng(seed)
		self.vocab: Optional[Vocabulary] = None
		self.word_vectors: Optional[np.ndarray] = None
		self.context_vectors: Optional[np.ndarray] = None
		self.word_bias: Optional[np.ndarray] = None
		self.context_bias: Optional[np.ndarray] = None
		self.word_accumulator: Optional[np.ndarray] = None
		self.context_accumulator: Optional[np.ndarray] = None
		self.word_bias_accumulator: Optional[np.ndarray] = None
		self.context_bias_accumulator: Optional[np.ndarray] = None

	def _build_cooccurrence(self, tokenized_corpus: Sequence[Sequence[str]]) -> Dict[Tuple[int, int], float]:
		if self.vocab is None:
			raise ValueError("Vocabulary must be built before co-occurrence extraction.")
		cooccurrence: Dict[Tuple[int, int], float] = defaultdict(float)
		encoded = [self.vocab.encode(sentence) for sentence in tokenized_corpus]
		for sentence in encoded:
			for center_index, center_id in enumerate(sentence):
				left = max(0, center_index - self.window_size)
				right = min(len(sentence), center_index + self.window_size + 1)
				for context_index in range(left, right):
					if context_index == center_index:
						continue
					context_id = sentence[context_index]
					distance = abs(center_index - context_index)
					cooccurrence[(center_id, context_id)] += 1.0 / distance
		return cooccurrence

	def fit(self, tokenized_corpus: Sequence[Sequence[str]]) -> List[float]:
		self.vocab = Vocabulary.build(tokenized_corpus)
		vocab_size = len(self.vocab)
		self.word_vectors = self.rng.normal(0.0, 0.1, size=(vocab_size, self.embed_dim))
		self.context_vectors = self.rng.normal(0.0, 0.1, size=(vocab_size, self.embed_dim))
		self.word_bias = np.zeros(vocab_size, dtype=np.float64)
		self.context_bias = np.zeros(vocab_size, dtype=np.float64)
		self.word_accumulator = np.ones((vocab_size, self.embed_dim), dtype=np.float64)
		self.context_accumulator = np.ones((vocab_size, self.embed_dim), dtype=np.float64)
		self.word_bias_accumulator = np.ones(vocab_size, dtype=np.float64)
		self.context_bias_accumulator = np.ones(vocab_size, dtype=np.float64)

		cooccurrence_items = list(self._build_cooccurrence(tokenized_corpus).items())
		history: List[float] = []
		for _ in range(self.epochs):
			self.rng.shuffle(cooccurrence_items)
			epoch_loss = 0.0
			for (word_id, context_id), count in cooccurrence_items:
				assert self.word_vectors is not None and self.context_vectors is not None
				assert self.word_bias is not None and self.context_bias is not None
				assert self.word_accumulator is not None and self.context_accumulator is not None
				assert self.word_bias_accumulator is not None and self.context_bias_accumulator is not None

				weight = (count / self.x_max) ** self.alpha if count < self.x_max else 1.0
				word_vector = self.word_vectors[word_id].copy()
				context_vector = self.context_vectors[context_id].copy()
				log_count = math.log(count + EPS)
				residual = (
					float(np.dot(word_vector, context_vector))
					+ self.word_bias[word_id]
					+ self.context_bias[context_id]
					- log_count
				)
				scaled_residual = 2.0 * weight * residual
				grad_word = scaled_residual * context_vector
				grad_context = scaled_residual * word_vector
				grad_word_bias = scaled_residual
				grad_context_bias = scaled_residual

				self.word_accumulator[word_id] += grad_word * grad_word
				self.context_accumulator[context_id] += grad_context * grad_context
				self.word_bias_accumulator[word_id] += grad_word_bias * grad_word_bias
				self.context_bias_accumulator[context_id] += grad_context_bias * grad_context_bias

				self.word_vectors[word_id] -= self.learning_rate * grad_word / np.sqrt(
					self.word_accumulator[word_id] + EPS
				)
				self.context_vectors[context_id] -= self.learning_rate * grad_context / np.sqrt(
					self.context_accumulator[context_id] + EPS
				)
				self.word_bias[word_id] -= self.learning_rate * grad_word_bias / math.sqrt(
					float(self.word_bias_accumulator[word_id] + EPS)
				)
				self.context_bias[context_id] -= self.learning_rate * grad_context_bias / math.sqrt(
					float(self.context_bias_accumulator[context_id] + EPS)
				)
				epoch_loss += weight * residual * residual
			history.append(epoch_loss / max(1, len(cooccurrence_items)))
		return history

	def embeddings(self) -> np.ndarray:
		if self.word_vectors is None or self.context_vectors is None:
			raise ValueError("Model must be fitted before requesting embeddings.")
		return self.word_vectors + self.context_vectors


class NGramLanguageModel:
	def __init__(self, n: int = 3, alpha: float = 1.0):
		if n < 1:
			raise ValueError("n must be at least 1")
		self.n = n
		self.alpha = alpha
		self.context_counts: Counter[Tuple[str, ...]] = Counter()
		self.ngram_counts: Counter[Tuple[str, ...]] = Counter()
		self.vocabulary: List[str] = []

	def fit(self, tokenized_corpus: Sequence[Sequence[str]]) -> "NGramLanguageModel":
		vocabulary = set(SPECIAL_TOKENS)
		bos_context = ["<bos>"] * (self.n - 1)
		for sentence in tokenized_corpus:
			vocabulary.update(sentence)
			augmented = bos_context + list(sentence) + ["<eos>"]
			for index in range(self.n - 1, len(augmented)):
				context = tuple(augmented[index - self.n + 1 : index])
				token = augmented[index]
				self.context_counts[context] += 1
				self.ngram_counts[context + (token,)] += 1
		self.vocabulary = sorted(vocabulary)
		return self

	def probability(self, context: Sequence[str], token: str) -> float:
		context = tuple(context[-(self.n - 1) :]) if self.n > 1 else tuple()
		numerator = self.ngram_counts[context + (token,)] + self.alpha
		denominator = self.context_counts[context] + self.alpha * len(self.vocabulary)
		return numerator / denominator

	def sentence_log_probability(self, sentence: Sequence[str]) -> float:
		bos_context = ["<bos>"] * (self.n - 1)
		augmented = bos_context + list(sentence) + ["<eos>"]
		log_probability = 0.0
		for index in range(self.n - 1, len(augmented)):
			context = augmented[index - self.n + 1 : index]
			token = augmented[index]
			log_probability += math.log(self.probability(context, token) + EPS)
		return log_probability

	def perplexity(self, tokenized_corpus: Sequence[Sequence[str]]) -> float:
		total_log_probability = 0.0
		total_tokens = 0
		for sentence in tokenized_corpus:
			total_log_probability += self.sentence_log_probability(sentence)
			total_tokens += len(sentence) + 1
		return math.exp(-total_log_probability / max(1, total_tokens))

	def generate(self, max_length: int = 20) -> List[str]:
		context = ["<bos>"] * (self.n - 1)
		sentence: List[str] = []
		candidates = [token for token in self.vocabulary if token not in {"<pad>", "<bos>"}]
		for _ in range(max_length):
			probabilities = np.array([self.probability(context, token) for token in candidates], dtype=np.float64)
			probabilities /= probabilities.sum()
			token = str(np.random.choice(candidates, p=probabilities))
			if token == "<eos>":
				break
			sentence.append(token)
			if self.n > 1:
				context = (context + [token])[-(self.n - 1) :]
		return sentence


class HMMPOSTagger:
	def __init__(self, alpha: float = 1.0):
		self.alpha = alpha
		self.tags: List[str] = []
		self.words: List[str] = []
		self.start_counts: Counter[str] = Counter()
		self.transition_counts: Counter[Tuple[str, str]] = Counter()
		self.emission_counts: Counter[Tuple[str, str]] = Counter()
		self.tag_counts: Counter[str] = Counter()
		self.num_sentences = 0

	def fit(self, tagged_sentences: Sequence[Tuple[Sequence[str], Sequence[str]]]) -> "HMMPOSTagger":
		tag_set = set()
		word_set = set()
		for tokens, tags in tagged_sentences:
			if len(tokens) != len(tags):
				raise ValueError("Each token sequence must have the same length as its tag sequence.")
			self.num_sentences += 1
			if tags:
				self.start_counts[tags[0]] += 1
			previous_tag = None
			for token, tag in zip(tokens, tags):
				tag_set.add(tag)
				word_set.add(token)
				self.tag_counts[tag] += 1
				self.emission_counts[(tag, token)] += 1
				if previous_tag is not None:
					self.transition_counts[(previous_tag, tag)] += 1
				previous_tag = tag
		self.tags = sorted(tag_set)
		self.words = sorted(word_set)
		return self

	def _start_log_prob(self, tag: str) -> float:
		numerator = self.start_counts[tag] + self.alpha
		denominator = self.num_sentences + self.alpha * len(self.tags)
		return math.log(numerator / denominator + EPS)

	def _transition_log_prob(self, prev_tag: str, next_tag: str) -> float:
		numerator = self.transition_counts[(prev_tag, next_tag)] + self.alpha
		denominator = self.tag_counts[prev_tag] + self.alpha * len(self.tags)
		return math.log(numerator / denominator + EPS)

	def _emission_log_prob(self, tag: str, token: str) -> float:
		numerator = self.emission_counts[(tag, token)] + self.alpha
		denominator = self.tag_counts[tag] + self.alpha * (len(self.words) + 1)
		return math.log(numerator / denominator + EPS)

	def viterbi(self, tokens: Sequence[str]) -> List[str]:
		if not tokens:
			return []
		num_tags = len(self.tags)
		length = len(tokens)
		scores = np.full((length, num_tags), -np.inf, dtype=np.float64)
		backpointers = np.zeros((length, num_tags), dtype=np.int64)

		for tag_index, tag in enumerate(self.tags):
			scores[0, tag_index] = self._start_log_prob(tag) + self._emission_log_prob(tag, tokens[0])

		for time_index in range(1, length):
			token = tokens[time_index]
			for current_index, current_tag in enumerate(self.tags):
				emission_score = self._emission_log_prob(current_tag, token)
				transition_scores = np.array(
					[
						scores[time_index - 1, previous_index]
						+ self._transition_log_prob(self.tags[previous_index], current_tag)
						for previous_index in range(num_tags)
					],
					dtype=np.float64,
				)
				best_previous = int(np.argmax(transition_scores))
				scores[time_index, current_index] = transition_scores[best_previous] + emission_score
				backpointers[time_index, current_index] = best_previous

		best_last = int(np.argmax(scores[-1]))
		best_path = [best_last]
		for time_index in range(length - 1, 0, -1):
			best_path.append(int(backpointers[time_index, best_path[-1]]))
		best_path.reverse()
		return [self.tags[index] for index in best_path]


class RNNScratchCache(TypedDict):
	token_ids: List[int]
	inputs: List[np.ndarray]
	hidden_states: List[np.ndarray]
	probabilities: np.ndarray


class RNNClassifierScratch:
	def __init__(
		self,
		vocab_size: int,
		embed_dim: int,
		hidden_dim: int,
		num_classes: int,
		learning_rate: float = 0.05,
		seed: int = 42,
	):
		rng = np.random.default_rng(seed)
		self.embeddings = rng.normal(0.0, 0.1, size=(vocab_size, embed_dim))
		self.Wxh = rng.normal(0.0, 0.1, size=(hidden_dim, embed_dim))
		self.Whh = rng.normal(0.0, 0.1, size=(hidden_dim, hidden_dim))
		self.bh = np.zeros(hidden_dim, dtype=np.float64)
		self.Why = rng.normal(0.0, 0.1, size=(num_classes, hidden_dim))
		self.by = np.zeros(num_classes, dtype=np.float64)
		self.learning_rate = learning_rate

	def forward(self, token_ids: Sequence[int]) -> Tuple[np.ndarray, RNNScratchCache]:
		hidden_states = [np.zeros(self.Whh.shape[0], dtype=np.float64)]
		inputs: List[np.ndarray] = []
		for token_id in token_ids:
			x_t = self.embeddings[int(token_id)]
			h_t = np.tanh(self.Wxh @ x_t + self.Whh @ hidden_states[-1] + self.bh)
			inputs.append(x_t)
			hidden_states.append(h_t)
		logits = self.Why @ hidden_states[-1] + self.by
		probabilities = softmax(logits)
		cache = {
			"token_ids": list(token_ids),
			"inputs": inputs,
			"hidden_states": hidden_states,
			"probabilities": probabilities,
		}
		cache: RNNScratchCache
		return probabilities, cache

	def backward(self, cache: RNNScratchCache, label: int) -> Dict[str, np.ndarray]:
		probabilities = np.array(cache["probabilities"], dtype=np.float64)
		hidden_states = cache["hidden_states"]
		inputs = cache["inputs"]
		token_ids = cache["token_ids"]

		dlogits = probabilities.copy()
		dlogits[int(label)] -= 1.0

		grad_embeddings = np.zeros_like(self.embeddings)
		grad_Wxh = np.zeros_like(self.Wxh)
		grad_Whh = np.zeros_like(self.Whh)
		grad_bh = np.zeros_like(self.bh)
		grad_Why = np.outer(dlogits, hidden_states[-1])
		grad_by = dlogits.copy()

		dh = self.Why.T @ dlogits
		for time_index in reversed(range(len(token_ids))):
			h_t = hidden_states[time_index + 1]
			h_prev = hidden_states[time_index]
			dtanh = (1.0 - h_t * h_t) * dh
			grad_Wxh += np.outer(dtanh, inputs[time_index])
			grad_Whh += np.outer(dtanh, h_prev)
			grad_bh += dtanh
			grad_embeddings[int(token_ids[time_index])] += self.Wxh.T @ dtanh
			dh = self.Whh.T @ dtanh

		return {
			"embeddings": grad_embeddings,
			"Wxh": grad_Wxh,
			"Whh": grad_Whh,
			"bh": grad_bh,
			"Why": grad_Why,
			"by": grad_by,
		}

	def _apply_gradients(self, gradients: Dict[str, np.ndarray], clip_value: float = 5.0) -> None:
		for gradient in gradients.values():
			np.clip(gradient, -clip_value, clip_value, out=gradient)
		self.embeddings -= self.learning_rate * gradients["embeddings"]
		self.Wxh -= self.learning_rate * gradients["Wxh"]
		self.Whh -= self.learning_rate * gradients["Whh"]
		self.bh -= self.learning_rate * gradients["bh"]
		self.Why -= self.learning_rate * gradients["Why"]
		self.by -= self.learning_rate * gradients["by"]

	def fit(self, sequences: Sequence[Sequence[int]], labels: Sequence[int], epochs: int = 20) -> List[float]:
		history: List[float] = []
		for _ in range(epochs):
			total_loss = 0.0
			for sequence, label in zip(sequences, labels):
				probabilities, cache = self.forward(sequence)
				total_loss += -math.log(float(probabilities[int(label)]) + EPS)
				gradients = self.backward(cache, int(label))
				self._apply_gradients(gradients)
			history.append(total_loss / max(1, len(sequences)))
		return history

	def predict(self, sequence: Sequence[int]) -> int:
		probabilities, _ = self.forward(sequence)
		return int(np.argmax(probabilities))


class LSTMCellScratch:
	def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
		rng = np.random.default_rng(seed)
		self.hidden_dim = hidden_dim
		self.W = rng.normal(0.0, 0.1, size=(4 * hidden_dim, input_dim + hidden_dim))
		self.b = np.zeros(4 * hidden_dim, dtype=np.float64)

	def step(
		self,
		x_t: np.ndarray,
		h_prev: np.ndarray,
		c_prev: np.ndarray,
	) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
		joined = np.concatenate([x_t, h_prev])
		gates = self.W @ joined + self.b
		i, f, o, g = np.split(gates, 4)
		i = np.asarray(sigmoid(i), dtype=np.float64)
		f = np.asarray(sigmoid(f), dtype=np.float64)
		o = np.asarray(sigmoid(o), dtype=np.float64)
		g = np.tanh(g)
		c_t = f * c_prev + i * g
		h_t = o * np.tanh(c_t)
		return h_t, c_t, {"input_gate": i, "forget_gate": f, "output_gate": o, "candidate": g}

	def forward_sequence(self, inputs: Sequence[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
		h_t = np.zeros(self.hidden_dim, dtype=np.float64)
		c_t = np.zeros(self.hidden_dim, dtype=np.float64)
		hidden_states: List[np.ndarray] = []
		cell_states: List[np.ndarray] = []
		for x_t in inputs:
			h_t, c_t, _ = self.step(x_t, h_t, c_t)
			hidden_states.append(h_t)
			cell_states.append(c_t)
		return hidden_states, cell_states


class GRUCellScratch:
	def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
		rng = np.random.default_rng(seed)
		self.hidden_dim = hidden_dim
		self.W_z = rng.normal(0.0, 0.1, size=(hidden_dim, input_dim + hidden_dim))
		self.W_r = rng.normal(0.0, 0.1, size=(hidden_dim, input_dim + hidden_dim))
		self.W_h = rng.normal(0.0, 0.1, size=(hidden_dim, input_dim + hidden_dim))
		self.b_z = np.zeros(hidden_dim, dtype=np.float64)
		self.b_r = np.zeros(hidden_dim, dtype=np.float64)
		self.b_h = np.zeros(hidden_dim, dtype=np.float64)

	def step(self, x_t: np.ndarray, h_prev: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
		joined = np.concatenate([x_t, h_prev])
		z_t = np.asarray(sigmoid(self.W_z @ joined + self.b_z), dtype=np.float64)
		r_t = np.asarray(sigmoid(self.W_r @ joined + self.b_r), dtype=np.float64)
		candidate_input = np.concatenate([x_t, r_t * h_prev])
		h_tilde = np.tanh(self.W_h @ candidate_input + self.b_h)
		h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
		return h_t, {"update_gate": z_t, "reset_gate": r_t, "candidate": h_tilde}

	def forward_sequence(self, inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
		h_t = np.zeros(self.hidden_dim, dtype=np.float64)
		hidden_states: List[np.ndarray] = []
		for x_t in inputs:
			h_t, _ = self.step(x_t, h_t)
			hidden_states.append(h_t)
		return hidden_states


class AdditiveAttentionScratch:
	def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int, seed: int = 42):
		rng = np.random.default_rng(seed)
		self.W_h = rng.normal(0.0, 0.1, size=(attention_dim, encoder_dim))
		self.W_s = rng.normal(0.0, 0.1, size=(attention_dim, decoder_dim))
		self.v = rng.normal(0.0, 0.1, size=(attention_dim,))
		self.b = np.zeros(attention_dim, dtype=np.float64)

	def forward(self, encoder_states: np.ndarray, decoder_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		energies = np.tanh(encoder_states @ self.W_h.T + decoder_state @ self.W_s.T + self.b)
		scores = energies @ self.v
		weights = softmax(scores)
		context = weights @ encoder_states
		return context, weights


def sinusoidal_positional_encoding(max_length: int, model_dim: int) -> np.ndarray:
	positions = np.arange(max_length, dtype=np.float64)[:, None]
	div_term = np.exp(np.arange(0, model_dim, 2, dtype=np.float64) * (-math.log(10000.0) / model_dim))
	encoding = np.zeros((max_length, model_dim), dtype=np.float64)
	encoding[:, 0::2] = np.sin(positions * div_term)
	encoding[:, 1::2] = np.cos(positions * div_term)
	return encoding


def causal_mask(sequence_length: int) -> np.ndarray:
	return np.tril(np.ones((sequence_length, sequence_length), dtype=bool))


def scaled_dot_product_attention(
	query: np.ndarray,
	key: np.ndarray,
	value: np.ndarray,
	mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	scale = math.sqrt(query.shape[-1])
	scores = query @ np.swapaxes(key, -1, -2) / scale
	if mask is not None:
		scores = np.where(mask, scores, -1e9)
	weights = softmax(scores, axis=-1)
	output = weights @ value
	return output, weights


class LayerNormScratch:
	def __init__(self, model_dim: int, eps: float = 1e-5):
		self.gamma = np.ones(model_dim, dtype=np.float64)
		self.beta = np.zeros(model_dim, dtype=np.float64)
		self.eps = eps

	def __call__(self, inputs: np.ndarray) -> np.ndarray:
		mean = inputs.mean(axis=-1, keepdims=True)
		variance = ((inputs - mean) ** 2).mean(axis=-1, keepdims=True)
		normalized = (inputs - mean) / np.sqrt(variance + self.eps)
		return self.gamma * normalized + self.beta


class FeedForwardScratch:
	def __init__(self, model_dim: int, ff_dim: int, seed: int = 42):
		rng = np.random.default_rng(seed)
		self.W1 = rng.normal(0.0, 0.02, size=(model_dim, ff_dim))
		self.b1 = np.zeros(ff_dim, dtype=np.float64)
		self.W2 = rng.normal(0.0, 0.02, size=(ff_dim, model_dim))
		self.b2 = np.zeros(model_dim, dtype=np.float64)

	def forward(self, inputs: np.ndarray) -> np.ndarray:
		hidden = np.maximum(0.0, inputs @ self.W1 + self.b1)
		return hidden @ self.W2 + self.b2


class MultiHeadSelfAttentionScratch:
	def __init__(self, model_dim: int, num_heads: int, seed: int = 42):
		if model_dim % num_heads != 0:
			raise ValueError("model_dim must be divisible by num_heads")
		rng = np.random.default_rng(seed)
		self.model_dim = model_dim
		self.num_heads = num_heads
		self.head_dim = model_dim // num_heads
		self.W_q = rng.normal(0.0, 0.02, size=(model_dim, model_dim))
		self.W_k = rng.normal(0.0, 0.02, size=(model_dim, model_dim))
		self.W_v = rng.normal(0.0, 0.02, size=(model_dim, model_dim))
		self.W_o = rng.normal(0.0, 0.02, size=(model_dim, model_dim))

	def forward(self, inputs: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		length = inputs.shape[0]
		query = (inputs @ self.W_q).reshape(length, self.num_heads, self.head_dim).transpose(1, 0, 2)
		key = (inputs @ self.W_k).reshape(length, self.num_heads, self.head_dim).transpose(1, 0, 2)
		value = (inputs @ self.W_v).reshape(length, self.num_heads, self.head_dim).transpose(1, 0, 2)
		attention_mask = None
		if mask is not None:
			attention_mask = np.broadcast_to(mask[None, :, :], (self.num_heads, length, length))
		attended, attention_weights = scaled_dot_product_attention(query, key, value, attention_mask)
		merged = attended.transpose(1, 0, 2).reshape(length, self.model_dim)
		return merged @ self.W_o, attention_weights


class TransformerBlockScratch:
	def __init__(self, model_dim: int, num_heads: int, ff_dim: int, seed: int = 42):
		self.attention = MultiHeadSelfAttentionScratch(model_dim, num_heads, seed=seed)
		self.feed_forward = FeedForwardScratch(model_dim, ff_dim, seed=seed + 1)
		self.norm1 = LayerNormScratch(model_dim)
		self.norm2 = LayerNormScratch(model_dim)

	def forward(self, inputs: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
		attention_output, attention_weights = self.attention.forward(self.norm1(inputs), mask=mask)
		residual = inputs + attention_output
		output = residual + self.feed_forward.forward(self.norm2(residual))
		return output, attention_weights


class MiniTransformerLMScratch:
	def __init__(
		self,
		vocab_size: int,
		model_dim: int = 64,
		num_heads: int = 4,
		ff_dim: int = 128,
		num_layers: int = 2,
		max_length: int = 128,
		seed: int = 42,
	):
		rng = np.random.default_rng(seed)
		self.vocab_size = vocab_size
		self.model_dim = model_dim
		self.max_length = max_length
		self.token_embeddings = rng.normal(0.0, 0.02, size=(vocab_size, model_dim))
		self.position_encoding = sinusoidal_positional_encoding(max_length, model_dim)
		self.layers = [
			TransformerBlockScratch(model_dim, num_heads, ff_dim, seed=seed + layer_index * 7)
			for layer_index in range(num_layers)
		]
		self.norm = LayerNormScratch(model_dim)
		self.output_projection = rng.normal(0.0, 0.02, size=(model_dim, vocab_size))

	def forward(self, token_ids: Sequence[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
		if len(token_ids) > self.max_length:
			raise ValueError("Sequence length exceeds model maximum length.")
		inputs = self.token_embeddings[list(token_ids)] + self.position_encoding[: len(token_ids)]
		mask = causal_mask(len(token_ids))
		attention_maps: List[np.ndarray] = []
		hidden = inputs
		for layer in self.layers:
			hidden, attention_weights = layer.forward(hidden, mask=mask)
			attention_maps.append(attention_weights)
		logits = self.norm(hidden) @ self.output_projection
		return logits, attention_maps

	def next_token_distribution(self, token_ids: Sequence[int]) -> np.ndarray:
		logits, _ = self.forward(token_ids)
		return softmax(logits[-1])

	def generate(self, prefix_ids: Sequence[int], max_new_tokens: int = 20) -> List[int]:
		generated = list(prefix_ids)
		for _ in range(max_new_tokens):
			distribution = self.next_token_distribution(generated)
			next_id = int(np.argmax(distribution))
			generated.append(next_id)
			if len(generated) >= self.max_length:
				break
		return generated


def load_sentiment_dataset(path: str | Path) -> Tuple[List[str], List[int]]:
	records = json.loads(Path(path).read_text())
	texts = [str(record["text"]) for record in records]
	labels = [int(record["label"]) for record in records]
	return texts, labels


def load_pos_dataset(path: str | Path) -> List[Tuple[List[str], List[str]]]:
	records = json.loads(Path(path).read_text())
	return [(list(record["tokens"]), list(record["tags"])) for record in records]


def load_translation_pairs(path: str | Path) -> List[Tuple[str, str]]:
	records = json.loads(Path(path).read_text())
	return [(str(record["source"]), str(record["target"])) for record in records]


def load_language_model_corpus(path: str | Path) -> List[List[str]]:
	lines = Path(path).read_text().splitlines()
	return [tokenize(line) for line in lines if line.strip()]


def demo() -> None:
	base_dir = Path(__file__).resolve().parent
	texts, labels = load_sentiment_dataset(base_dir / "dataset" / "sentiment_toy.json")

	bow = BagOfWordsVectorizer(max_features=128)
	features = bow.fit_transform(texts)
	classifier = SoftmaxRegressionScratch(features.shape[1], 2, learning_rate=0.2)
	history = classifier.fit(features, labels, epochs=150, batch_size=4)
	predictions = classifier.predict(features)
	accuracy = float(np.mean(predictions == np.array(labels)))
	print(f"Scratch BoW accuracy: {accuracy:.3f} | final loss: {history[-1]:.4f}")

	corpus = load_language_model_corpus(base_dir / "dataset" / "lm_tiny.txt")
	skipgram = Word2VecScratch(mode="skipgram", embed_dim=16, epochs=10, window_size=2)
	skipgram.fit(corpus)
	print("Word2Vec nearest neighbors for 'language':", skipgram.most_similar("language", top_k=3))

	ngram = NGramLanguageModel(n=3, alpha=0.5).fit(corpus)
	print(f"Trigram perplexity on the toy corpus: {ngram.perplexity(corpus):.3f}")

	tagged_sentences = load_pos_dataset(base_dir / "dataset" / "pos_toy.json")
	hmm = HMMPOSTagger(alpha=0.5).fit(tagged_sentences)
	print("Viterbi tags for 'the bird sings':", hmm.viterbi(["the", "bird", "sings"]))

	tokenized = ensure_tokenized(texts)
	vocab = Vocabulary.build(tokenized)
	encoded = [vocab.encode(tokens) for tokens in tokenized]
	rnn = RNNClassifierScratch(len(vocab), embed_dim=16, hidden_dim=24, num_classes=2, learning_rate=0.05)
	rnn.fit(encoded, labels, epochs=8)
	rnn_predictions = np.array([rnn.predict(sequence) for sequence in encoded])
	rnn_accuracy = float(np.mean(rnn_predictions == np.array(labels)))
	print(f"Scratch RNN accuracy: {rnn_accuracy:.3f}")

	transformer = MiniTransformerLMScratch(vocab_size=len(vocab), model_dim=32, num_heads=4, ff_dim=64, num_layers=2)
	logits, attention_maps = transformer.forward(vocab.encode(tokenized[0], add_eos=True))
	print(f"Mini Transformer logits shape: {logits.shape}; heads in layer 1: {attention_maps[0].shape[0]}")


if __name__ == "__main__":
	demo()
