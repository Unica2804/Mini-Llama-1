import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

from src import model


@pytest.fixture
def tiny_config():
	return model.LLamaConfig(
		d_model=32,
		n_layers=2,
		n_heads=4,
		num_kv_groups=2,
		vocab_size=64,
		max_seq_len=16,
		seq_len=16,
		multiple_of=8,
		norm_eps=1e-6,
		rope_base=10000.0,
	)


def _patch_attention(monkeypatch):
	def causal_attention(query, key, value, attn_bias, p):
		del attn_bias, p
		query_heads = query.transpose(1, 2)
		key_heads = key.transpose(1, 2)
		value_heads = value.transpose(1, 2)
		scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
		scores = scores / query.shape[-1] ** 0.5
		sequence_length = query.shape[1]
		causal_mask = torch.triu(
			torch.ones(
				sequence_length,
				sequence_length,
				dtype=torch.bool,
				device=query.device,
			),
			diagonal=1,
		)
		scores = scores.masked_fill(causal_mask, float("-inf"))
		weights = torch.softmax(scores, dim=-1)
		return torch.matmul(weights, value_heads).transpose(1, 2)

	monkeypatch.setattr(model.xops, "memory_efficient_attention", causal_attention)


def test_config_computes_head_and_feed_forward_dimensions(tiny_config):
	assert tiny_config.head_dim == 8
	assert tiny_config.d_ff == 88
	assert tiny_config.d_ff % tiny_config.multiple_of == 0


def test_model_uses_supplied_config(tiny_config):
	model_instance = model.make_model(tiny_config)

	assert len(model_instance.layers) == tiny_config.n_layers
	assert model_instance.embedding.embedding.num_embeddings == tiny_config.vocab_size
	assert model_instance.embedding.embedding.embedding_dim == tiny_config.d_model
	assert model_instance.output.proj.out_features == tiny_config.vocab_size


def test_model_forward_returns_language_model_logits(tiny_config, monkeypatch):
	_patch_attention(monkeypatch)
	model_instance = model.make_model(tiny_config)
	tokens = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.seq_len))

	logits = model_instance(tokens)

	assert logits.shape == (2, tiny_config.seq_len, tiny_config.vocab_size)
	assert torch.isfinite(logits).all()


def test_model_backward_produces_gradients(tiny_config, monkeypatch):
	_patch_attention(monkeypatch)
	model_instance = model.make_model(tiny_config)
	tokens = torch.randint(0, tiny_config.vocab_size, (2, 8))

	model_instance(tokens).mean().backward()

	assert model_instance.embedding.embedding.weight.grad is not None
	assert model_instance.layers[0].self_attn.wqkv.weight.grad is not None
	assert model_instance.output.proj.weight.grad is not None


def test_rope_preserves_q_and_k_shapes(tiny_config):
	rope = model.RotaryPositionalEncoding(tiny_config)
	query = torch.randn(2, 5, tiny_config.n_heads, tiny_config.head_dim)
	key = torch.randn_like(query)
	positions = torch.arange(5).expand(2, 5)
	cos, sin = rope(query, 5)

	rotated_query, rotated_key = model.apply_rotary_pos_emb(
		query,
		key,
		cos,
		sin,
		positions,
	)

	assert rotated_query.shape == query.shape
	assert rotated_key.shape == key.shape
	assert torch.isfinite(rotated_query).all()
	assert torch.isfinite(rotated_key).all()
