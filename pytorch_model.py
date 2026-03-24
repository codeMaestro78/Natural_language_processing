"""PyTorch implementations for the main neural NLP architectures in this repository."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple, cast


if TYPE_CHECKING:
	import torch as torch_typing
	import torch.nn as nn_typing
	Tensor = torch_typing.Tensor
	Device = torch_typing.device
	DropoutLayer = nn_typing.Dropout
	_TorchRequiredBase = nn_typing.Module

else:
	Tensor = object
	Device = object
	DropoutLayer = object
	_TorchRequiredBase = object


try:
	import torch as _torch
	import torch.nn as _nn
	import torch.nn.functional as _F

	TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard for environments without torch
	_torch = None
	_nn = None
	_F = None
	TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
	assert _torch is not None and _nn is not None and _F is not None
	torch = _torch
	nn = _nn
	F = _F

	def lengths_to_mask(lengths: Tensor, max_length: Optional[int] = None) -> Tensor:
		max_length = max_length or int(lengths.max().item())
		positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
		return positions < lengths.unsqueeze(1)


	def sinusoidal_positions(max_length: int, model_dim: int, device: Optional[Device] = None) -> Tensor:
		positions = torch.arange(max_length, device=device, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(
			torch.arange(0, model_dim, 2, device=device, dtype=torch.float32)
			* (-math.log(10000.0) / model_dim)
		)
		encoding = torch.zeros(max_length, model_dim, device=device, dtype=torch.float32)
		encoding[:, 0::2] = torch.sin(positions * div_term)
		encoding[:, 1::2] = torch.cos(positions * div_term)
		return encoding


	class PositionalEncoding(nn.Module):
		def __init__(self, model_dim: int, max_length: int = 512):
			super().__init__()
			self.register_buffer("encoding", sinusoidal_positions(max_length, model_dim), persistent=False)

		def forward(self, x: Tensor) -> Tensor:
			encoding = cast(Tensor, self.encoding)
			return x + encoding[: x.size(1)].unsqueeze(0)


	class BOWClassifier(nn.Module):
		def __init__(self, input_dim: int, num_classes: int):
			super().__init__()
			self.linear = nn.Linear(input_dim, num_classes)

		def forward(self, features: Tensor) -> Tensor:
			return self.linear(features)


	class TFIDFClassifier(BOWClassifier):
		pass


	class Word2VecNegativeSampling(nn.Module):
		def __init__(self, vocab_size: int, embed_dim: int):
			super().__init__()
			self.input_embeddings = nn.Embedding(vocab_size, embed_dim)
			self.output_embeddings = nn.Embedding(vocab_size, embed_dim)
			nn.init.normal_(self.input_embeddings.weight, mean=0.0, std=0.1)
			nn.init.zeros_(self.output_embeddings.weight)

		def forward(
			self,
			center_ids: Tensor,
			target_ids: Tensor,
			negative_ids: Tensor,
		) -> Tensor:
			center = self.input_embeddings(center_ids)
			target = self.output_embeddings(target_ids)
			negatives = self.output_embeddings(negative_ids)
			positive_logits = torch.sum(center * target, dim=-1)
			negative_logits = torch.einsum("bd,bkd->bk", center, negatives)
			positive_loss = -F.logsigmoid(positive_logits)
			negative_loss = -F.logsigmoid(-negative_logits).sum(dim=-1)
			return (positive_loss + negative_loss).mean()

		def embeddings(self) -> Tensor:
			return self.input_embeddings.weight + self.output_embeddings.weight


	class RecurrentTextClassifier(nn.Module):
		recurrent_cls = nn.RNN

		def __init__(
			self,
			vocab_size: int,
			embed_dim: int,
			hidden_dim: int,
			num_classes: int,
			num_layers: int = 1,
			dropout: float = 0.1,
			pad_id: int = 0,
		):
			super().__init__()
			effective_dropout = dropout if num_layers > 1 else 0.0
			self.pad_id = pad_id
			self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
			self.recurrent = self.recurrent_cls(
				embed_dim,
				hidden_dim,
				num_layers=num_layers,
				batch_first=True,
				dropout=effective_dropout,
			)
			self.dropout = nn.Dropout(dropout)
			self.classifier = nn.Linear(hidden_dim, num_classes)

		def forward(self, token_ids: Tensor, lengths: Optional[Tensor] = None) -> Tensor:
			embeddings = self.embedding(token_ids)
			outputs, hidden = self.recurrent(embeddings)
			if isinstance(hidden, tuple):
				hidden_state = hidden[0][-1]
			else:
				hidden_state = hidden[-1]
			hidden_state = self.dropout(hidden_state)
			return self.classifier(hidden_state)


	class RNNTextClassifier(RecurrentTextClassifier):
		recurrent_cls = nn.RNN


	class LSTMTextClassifier(RecurrentTextClassifier):
		recurrent_cls = nn.LSTM


	class GRUTextClassifier(RecurrentTextClassifier):
		recurrent_cls = nn.GRU


	class BahdanauAttention(nn.Module):
		def __init__(self, hidden_dim: int):
			super().__init__()
			self.encoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
			self.decoder_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
			self.energy = nn.Linear(hidden_dim, 1, bias=False)

		def forward(
			self,
			encoder_outputs: Tensor,
			decoder_state: Tensor,
			mask: Optional[Tensor] = None,
		) -> Tuple[Tensor, Tensor]:
			scores = self.energy(
				torch.tanh(self.encoder_projection(encoder_outputs) + self.decoder_projection(decoder_state).unsqueeze(1))
			).squeeze(-1)
			if mask is not None:
				scores = scores.masked_fill(~mask, -1e9)
			weights = torch.softmax(scores, dim=-1)
			context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
			return context, weights


	class Seq2SeqAttention(nn.Module):
		def __init__(
			self,
			src_vocab_size: int,
			tgt_vocab_size: int,
			embed_dim: int = 64,
			hidden_dim: int = 128,
			pad_id: int = 0,
		):
			super().__init__()
			self.pad_id = pad_id
			self.src_embedding = nn.Embedding(src_vocab_size, embed_dim, padding_idx=pad_id)
			self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=pad_id)
			self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
			self.decoder = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
			self.attention = BahdanauAttention(hidden_dim)
			self.output_projection = nn.Linear(hidden_dim * 2, tgt_vocab_size)

		def forward(
			self,
			src_tokens: Tensor,
			tgt_tokens: Tensor,
			teacher_forcing_ratio: float = 1.0,
		) -> Tuple[Tensor, Tensor]:
			src_mask = src_tokens.ne(self.pad_id)
			encoder_outputs, hidden = self.encoder(self.src_embedding(src_tokens))
			decoder_input = tgt_tokens[:, 0]
			decoder_hidden = hidden
			logits: List[Tensor] = []
			attention_maps: List[Tensor] = []

			for step in range(1, tgt_tokens.size(1)):
				embedded = self.tgt_embedding(decoder_input).unsqueeze(1)
				context, attention = self.attention(encoder_outputs, decoder_hidden[-1], src_mask)
				decoder_features = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
				decoder_output, decoder_hidden = self.decoder(decoder_features, decoder_hidden)
				step_logits = self.output_projection(torch.cat([decoder_output.squeeze(1), context], dim=-1))
				logits.append(step_logits)
				attention_maps.append(attention)

				if self.training and torch.rand(1, device=src_tokens.device).item() < teacher_forcing_ratio:
					decoder_input = tgt_tokens[:, step]
				else:
					decoder_input = step_logits.argmax(dim=-1)

			return torch.stack(logits, dim=1), torch.stack(attention_maps, dim=1)

		@torch.no_grad()
		def greedy_decode(
			self,
			src_tokens: Tensor,
			bos_id: int,
			eos_id: int,
			max_length: int = 20,
		) -> Tuple[Tensor, Tensor]:
			src_mask = src_tokens.ne(self.pad_id)
			encoder_outputs, hidden = self.encoder(self.src_embedding(src_tokens))
			decoder_input = torch.full((src_tokens.size(0),), bos_id, dtype=torch.long, device=src_tokens.device)
			decoder_hidden = hidden
			predictions: List[Tensor] = [decoder_input]
			attention_maps: List[Tensor] = []

			for _ in range(max_length):
				embedded = self.tgt_embedding(decoder_input).unsqueeze(1)
				context, attention = self.attention(encoder_outputs, decoder_hidden[-1], src_mask)
				decoder_features = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
				decoder_output, decoder_hidden = self.decoder(decoder_features, decoder_hidden)
				logits = self.output_projection(torch.cat([decoder_output.squeeze(1), context], dim=-1))
				decoder_input = logits.argmax(dim=-1)
				predictions.append(decoder_input)
				attention_maps.append(attention)
				if torch.all(decoder_input.eq(eos_id)):
					break
			return torch.stack(predictions, dim=1), torch.stack(attention_maps, dim=1) if attention_maps else torch.empty(0)


	def scaled_dot_product_attention(
		query: Tensor,
		key: Tensor,
		value: Tensor,
		attention_mask: Optional[Tensor] = None,
		causal: bool = False,
		dropout: Optional[DropoutLayer] = None,
	) -> Tuple[Tensor, Tensor]:
		head_dim = query.size(-1)
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
		if attention_mask is not None:
			scores = scores.masked_fill(~attention_mask, -1e9)
		if causal:
			length = scores.size(-1)
			causal_mask = torch.tril(torch.ones(length, length, device=scores.device, dtype=torch.bool))
			scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
		weights = torch.softmax(scores, dim=-1)
		if dropout is not None:
			weights = dropout(weights)
		return torch.matmul(weights, value), weights


	class MultiHeadAttention(nn.Module):
		def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
			super().__init__()
			if model_dim % num_heads != 0:
				raise ValueError("model_dim must be divisible by num_heads")
			self.model_dim = model_dim
			self.num_heads = num_heads
			self.head_dim = model_dim // num_heads
			self.q_proj = nn.Linear(model_dim, model_dim)
			self.k_proj = nn.Linear(model_dim, model_dim)
			self.v_proj = nn.Linear(model_dim, model_dim)
			self.o_proj = nn.Linear(model_dim, model_dim)
			self.dropout = nn.Dropout(dropout)

		def forward(
			self,
			x: Tensor,
			attention_mask: Optional[Tensor] = None,
			causal: bool = False,
		) -> Tuple[Tensor, Tensor]:
			batch_size, sequence_length, _ = x.shape
			query = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
			key = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
			value = self.v_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

			expanded_mask = None
			if attention_mask is not None:
				expanded_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
			attended, weights = scaled_dot_product_attention(
				query,
				key,
				value,
				attention_mask=expanded_mask,
				causal=causal,
				dropout=self.dropout,
			)
			attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.model_dim)
			return self.o_proj(attended), weights


	class PositionwiseFeedForward(nn.Module):
		def __init__(self, model_dim: int, ff_dim: int, dropout: float = 0.1):
			super().__init__()
			self.linear1 = nn.Linear(model_dim, ff_dim)
			self.linear2 = nn.Linear(ff_dim, model_dim)
			self.dropout = nn.Dropout(dropout)

		def forward(self, x: Tensor) -> Tensor:
			return self.linear2(self.dropout(F.gelu(self.linear1(x))))


	class TransformerEncoderBlock(nn.Module):
		def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
			super().__init__()
			self.attention = MultiHeadAttention(model_dim, num_heads, dropout=dropout)
			self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout=dropout)
			self.norm1 = nn.LayerNorm(model_dim)
			self.norm2 = nn.LayerNorm(model_dim)
			self.dropout = nn.Dropout(dropout)

		def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
			attention_output, weights = self.attention(self.norm1(x), attention_mask=attention_mask, causal=False)
			x = x + self.dropout(attention_output)
			x = x + self.dropout(self.feed_forward(self.norm2(x)))
			return x, weights


	class TransformerDecoderBlock(nn.Module):
		def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
			super().__init__()
			self.attention = MultiHeadAttention(model_dim, num_heads, dropout=dropout)
			self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout=dropout)
			self.norm1 = nn.LayerNorm(model_dim)
			self.norm2 = nn.LayerNorm(model_dim)
			self.dropout = nn.Dropout(dropout)

		def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
			attention_output, weights = self.attention(self.norm1(x), attention_mask=attention_mask, causal=True)
			x = x + self.dropout(attention_output)
			x = x + self.dropout(self.feed_forward(self.norm2(x)))
			return x, weights


	class MiniTransformerEncoderClassifier(nn.Module):
		def __init__(
			self,
			vocab_size: int,
			num_classes: int,
			model_dim: int = 128,
			num_heads: int = 4,
			ff_dim: int = 256,
			num_layers: int = 2,
			dropout: float = 0.1,
			pad_id: int = 0,
			max_length: int = 256,
		):
			super().__init__()
			self.pad_id = pad_id
			self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_id)
			self.position = PositionalEncoding(model_dim, max_length=max_length)
			self.layers = nn.ModuleList(
				[TransformerEncoderBlock(model_dim, num_heads, ff_dim, dropout=dropout) for _ in range(num_layers)]
			)
			self.norm = nn.LayerNorm(model_dim)
			self.classifier = nn.Linear(model_dim, num_classes)
			self.dropout = nn.Dropout(dropout)

		def forward(
			self,
			token_ids: Tensor,
			attention_mask: Optional[Tensor] = None,
		) -> Tuple[Tensor, List[Tensor]]:
			if attention_mask is None:
				attention_mask = token_ids.ne(self.pad_id)
			x = self.position(self.embedding(token_ids))
			attention_maps: List[Tensor] = []
			for layer in self.layers:
				x, weights = layer(x, attention_mask=attention_mask)
				attention_maps.append(weights)
			x = self.norm(x)
			pooled = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
			logits = self.classifier(self.dropout(pooled))
			return logits, attention_maps


	class MiniBERTForMLM(nn.Module):
		def __init__(
			self,
			vocab_size: int,
			model_dim: int = 128,
			num_heads: int = 4,
			ff_dim: int = 256,
			num_layers: int = 4,
			dropout: float = 0.1,
			pad_id: int = 0,
			max_length: int = 256,
		):
			super().__init__()
			self.pad_id = pad_id
			self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_id)
			self.position = PositionalEncoding(model_dim, max_length=max_length)
			self.layers = nn.ModuleList(
				[TransformerEncoderBlock(model_dim, num_heads, ff_dim, dropout=dropout) for _ in range(num_layers)]
			)
			self.norm = nn.LayerNorm(model_dim)
			self.mlm_head = nn.Sequential(
				nn.Linear(model_dim, model_dim),
				nn.GELU(),
				nn.LayerNorm(model_dim),
				nn.Linear(model_dim, vocab_size),
			)

		def forward(
			self,
			token_ids: Tensor,
			attention_mask: Optional[Tensor] = None,
		) -> Tuple[Tensor, List[Tensor]]:
			if attention_mask is None:
				attention_mask = token_ids.ne(self.pad_id)
			x = self.position(self.embedding(token_ids))
			attention_maps: List[Tensor] = []
			for layer in self.layers:
				x, weights = layer(x, attention_mask=attention_mask)
				attention_maps.append(weights)
			logits = self.mlm_head(self.norm(x))
			return logits, attention_maps


	class MiniGPTLM(nn.Module):
		def __init__(
			self,
			vocab_size: int,
			model_dim: int = 128,
			num_heads: int = 4,
			ff_dim: int = 256,
			num_layers: int = 4,
			dropout: float = 0.1,
			pad_id: int = 0,
			max_length: int = 256,
		):
			super().__init__()
			self.pad_id = pad_id
			self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=pad_id)
			self.position = PositionalEncoding(model_dim, max_length=max_length)
			self.layers = nn.ModuleList(
				[TransformerDecoderBlock(model_dim, num_heads, ff_dim, dropout=dropout) for _ in range(num_layers)]
			)
			self.norm = nn.LayerNorm(model_dim)
			self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
			self.lm_head.weight = self.embedding.weight

		def forward(
			self,
			token_ids: Tensor,
			attention_mask: Optional[Tensor] = None,
		) -> Tuple[Tensor, List[Tensor]]:
			if attention_mask is None:
				attention_mask = token_ids.ne(self.pad_id)
			x = self.position(self.embedding(token_ids))
			attention_maps: List[Tensor] = []
			for layer in self.layers:
				x, weights = layer(x, attention_mask=attention_mask)
				attention_maps.append(weights)
			logits = self.lm_head(self.norm(x))
			return logits, attention_maps

		@torch.no_grad()
		def generate(
			self,
			prefix_ids: Tensor,
			max_new_tokens: int = 20,
			temperature: float = 1.0,
		) -> Tensor:
			generated = prefix_ids.clone()
			for _ in range(max_new_tokens):
				logits, _ = self.forward(generated)
				next_logits = logits[:, -1, :] / max(temperature, 1e-6)
				next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
				generated = torch.cat([generated, next_token], dim=1)
			return generated


	def masked_language_model_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
		return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)


	def causal_language_model_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
		return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)


else:

	class _TorchRequired(_TorchRequiredBase):
		def __init__(self, *args, **kwargs):
			raise ImportError("PyTorch is required for `pytorch_model.py`. Install torch from requirements.txt.")

		def embeddings(self) -> Tensor:
			raise ImportError("PyTorch is required for `Word2VecNegativeSampling.embeddings()`.")

		def greedy_decode(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
			raise ImportError("PyTorch is required for `Seq2SeqAttention.greedy_decode()`.")

		def generate(self, *args, **kwargs) -> Tensor:
			raise ImportError("PyTorch is required for `MiniGPTLM.generate()`.")


	PositionalEncoding = _TorchRequired  # pyright: ignore[reportAssignmentType]
	BOWClassifier = _TorchRequired  # pyright: ignore[reportAssignmentType]
	TFIDFClassifier = _TorchRequired  # pyright: ignore[reportAssignmentType]
	Word2VecNegativeSampling = _TorchRequired  # pyright: ignore[reportAssignmentType]
	RNNTextClassifier = _TorchRequired  # pyright: ignore[reportAssignmentType]
	LSTMTextClassifier = _TorchRequired  # pyright: ignore[reportAssignmentType]
	GRUTextClassifier = _TorchRequired  # pyright: ignore[reportAssignmentType]
	BahdanauAttention = _TorchRequired  # pyright: ignore[reportAssignmentType]
	Seq2SeqAttention = _TorchRequired  # pyright: ignore[reportAssignmentType]
	MultiHeadAttention = _TorchRequired  # pyright: ignore[reportAssignmentType]
	TransformerEncoderBlock = _TorchRequired  # pyright: ignore[reportAssignmentType]
	TransformerDecoderBlock = _TorchRequired  # pyright: ignore[reportAssignmentType]
	MiniTransformerEncoderClassifier = _TorchRequired  # pyright: ignore[reportAssignmentType]
	MiniBERTForMLM = _TorchRequired  # pyright: ignore[reportAssignmentType]
	MiniGPTLM = _TorchRequired  # pyright: ignore[reportAssignmentType]


	def _torch_required_function(*args, **kwargs):
		raise ImportError("PyTorch is required for `pytorch_model.py`. Install torch from requirements.txt.")


	lengths_to_mask = _torch_required_function
	sinusoidal_positions = _torch_required_function
	scaled_dot_product_attention = _torch_required_function
	masked_language_model_loss = _torch_required_function
	causal_language_model_loss = _torch_required_function

