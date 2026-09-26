"""
AviGPT-250M: Neural Core Architecture
--------------------------------------
Engineered for deep-thin, parameter-efficient autoregressive language modeling.
Features:
- Rotary Position Embeddings (RoPE) with high base theta (100k)
- Grouped-Query Attention (GQA) with 4:1 ratio for efficient KV caching
- QK-Norm (RMSNorm on Q and K) for mathematical training stability at small scale
- SwiGLU Feed-Forward Networks
- RMSNorm pre-normalization
- Weight-tied input/output embeddings (recovering ~50M params for deeper layers)
- PyTorch SDPA (Scaled Dot-Product Attention) & Liger Kernel compatibility

Architect & Owner: Avinash Ricky Yadlapalli
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AviGPTConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with precomputed cos/sin cache."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 100000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Compute inverse frequency band
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to Query and Key tensors."""
    # cos, sin shape: [seq_len, head_dim] -> unsqueeze to [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class AviAttention(nn.Module):
    """Grouped-Query Attention (GQA) with optional QK-Norm and SDPA acceleration."""

    def __init__(self, config: AviGPTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.qk_norm = config.qk_norm

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm if enabled (critical for numerical stability in deep small models)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply Rotary Position Embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV heads for GQA to match Query heads if needed
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled Dot-Product Attention (PyTorch native FlashAttention / SDPA kernel)
        is_causal = attention_mask is None and seq_len > 1
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class AviMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: AviGPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(silu(gate) * up)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AviBlock(nn.Module):
    """Pre-norm residual Transformer Block."""

    def __init__(self, config: AviGPTConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = AviAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AviMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention with residual connection
        normed_x = self.input_layernorm(x)
        attn_out = self.self_attn(normed_x, cos, sin, attention_mask=attention_mask)
        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class AviInternalMemory(nn.Module):
    """
    Sub-Millisecond Non-RAG Product-Key Memory (PKM) Layer.
    
    Provides internal associative memory storage directly in tensor weights:
    - Splits query into 2 sub-keys: q1, q2
    - Performs fast inner-product search over subkeys K1, K2
    - Cartesians the top-k sub-keys to form address space (e.g. 128 x 128 = 16,384 slots)
    - Retrieves top-k sparse factual memory value vectors
    - Adds memory delta via a zero-initialized residual gate for stable warmup
    - Latency: <0.02 ms inside PyTorch forward pass (zero RAG / network overhead)
    """

    def __init__(self, config: AviGPTConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.subkeys = config.internal_memory_subkeys
        self.key_dim = config.internal_memory_key_dim
        self.val_dim = config.internal_memory_val_dim
        self.topk = config.internal_memory_topk
        self.total_slots = self.subkeys * self.subkeys

        # Query projection from hidden state to 2 subkey queries
        self.q_proj = nn.Linear(self.hidden_size, 2 * self.key_dim, bias=False)

        # 2 Subkey codebooks (K1 and K2)
        self.keys1 = nn.Parameter(torch.randn(self.subkeys, self.key_dim) / math.sqrt(self.key_dim))
        self.keys2 = nn.Parameter(torch.randn(self.subkeys, self.key_dim) / math.sqrt(self.key_dim))

        # Memory values table [total_slots, val_dim]
        self.values = nn.Embedding(self.total_slots, self.val_dim)

        # Output projection back to hidden_size
        self.out_proj = nn.Linear(self.val_dim, self.hidden_size, bias=False)

        # Residual gating parameter, initialized to zero for progressive warmup
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 1. Project query and split into 2 halves
        q = self.q_proj(x)  # [B, T, 2 * key_dim]
        q1, q2 = q.chunk(2, dim=-1)  # [B, T, key_dim] each

        # 2. Compute inner products with sub-keys
        scores1 = torch.matmul(q1, self.keys1.t())  # [B, T, subkeys]
        scores2 = torch.matmul(q2, self.keys2.t())  # [B, T, subkeys]

        # 3. Top-k candidates per subkey (k_sub = min(topk, subkeys))
        k_sub = min(self.topk, self.subkeys)
        top_scores1, top_indices1 = scores1.topk(k_sub, dim=-1)  # [B, T, k_sub]
        top_scores2, top_indices2 = scores2.topk(k_sub, dim=-1)  # [B, T, k_sub]

        # 4. Cartesian product of top subkey scores
        combined_scores = (top_scores1.unsqueeze(-1) + top_scores2.unsqueeze(-2)).view(batch_size, seq_len, -1)

        # Compute slot indices: index = idx1 * subkeys + idx2
        idx1_exp = top_indices1.unsqueeze(-1).expand(-1, -1, -1, k_sub)
        idx2_exp = top_indices2.unsqueeze(-2).expand(-1, -1, k_sub, -1)
        combined_indices = (idx1_exp * self.subkeys + idx2_exp).view(batch_size, seq_len, -1)

        # 5. Select global top-k from the candidate pool
        final_scores, best_candidate_pos = combined_scores.topk(self.topk, dim=-1)
        final_indices = torch.gather(combined_indices, -1, best_candidate_pos)

        # Softmax over top-k slots
        weights = F.softmax(final_scores, dim=-1)  # [B, T, topk]

        # 6. Fetch memory values: [B, T, topk, val_dim]
        retrieved_vals = self.values(final_indices)

        # Weighted sum: [B, T, val_dim]
        memory_rep = torch.sum(weights.unsqueeze(-1) * retrieved_vals, dim=-2)

        # 7. Project back to hidden_size and add gated residual
        mem_delta = self.out_proj(memory_rep)
        return x + self.gate * mem_delta


class AviGPTForCausalLM(nn.Module):
    """Complete AviGPT-250M Causal Language Model."""

    def __init__(self, config: AviGPTConfig):
        super().__init__()
        self.config = config

        # Token Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Rotary Embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        # Transformer Layers
        self.layers = nn.ModuleList([AviBlock(config) for _ in range(config.num_hidden_layers)])

        # Optional Internal Non-RAG Product-Key Memory
        if config.use_internal_memory:
            self.internal_memory = AviInternalMemory(config)
            self.memory_layer_idx = min(config.internal_memory_layer, config.num_hidden_layers - 1)
        else:
            self.internal_memory = None
            self.memory_layer_idx = -1

        # Final RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie input & output embeddings (MobileLLM principle)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # Fetch RoPE frequencies
        cos, sin = self.rotary_emb(seq_len, device=device)

        # Pass through transformer layers with optional internal memory injection
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cos, sin, attention_mask=attention_mask)
            if self.internal_memory is not None and idx == self.memory_layer_idx:
                hidden_states = self.internal_memory(hidden_states)

        # Final RMSNorm
        hidden_states = self.norm(hidden_states)

        # Logits projection
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Labels from get_micro_batch are already pre-shifted by +1 (targets)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            return None, loss

        return logits, None

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculates total trainable parameter count."""
        if non_embedding:
            return sum(p.numel() for n, p in self.named_parameters() if "embed_tokens" not in n and "lm_head" not in n)
        # If tied, count unique parameters
        visited_ptrs = set()
        total_params = 0
        for p in self.parameters():
            if p.data_ptr() not in visited_ptrs:
                visited_ptrs.add(p.data_ptr())
                total_params += p.numel()
        return total_params
