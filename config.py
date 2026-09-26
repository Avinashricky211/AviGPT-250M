"""
AviGPT-250M: Architectural Configuration & Special Token Registry
------------------------------------------------------------------
Engineered for 250M parameter Small Language Model (SLM) training.
Incorporates:
- Modern high-efficiency transformer blocks (RoPE, RMSNorm, SwiGLU, GQA, QK-Norm)
- Tied input-output embeddings for maximum depth-to-parameter ratio
- Native 9-token hardware bus protocol for sub-millisecond NVMe SSD memory retrieval
- Cognitive reasoning and self-correction tokens (<think>, </think>, <correct>, </correct>)

Architect & Owner: Avinash Ricky Yadlapalli
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Special Tokens: Native Hardware NVMe Memory Bus + Cognitive Reasoning Tokens
SPECIAL_TOKENS = {
    # 1. Cognitive Reasoning Tokens
    "think_start": "<think>",
    "think_end": "</think>",
    "correct_start": "<correct>",
    "correct_end": "</correct>",
    
    # 2. Native NVMe SSD Hardware Memory Bus Protocol
    "intent_start": "<|intent_start|>",
    "intent_end": "<|intent_end|>",
    "mem_query_start": "<|mem_query|>",
    "mem_query_end": "<|mem_query_end|>",
    "mem_payload_start": "<|mem_payload|>",
    "mem_payload_end": "<|mem_payload_end|>",
    "calc_start": "<|calc|>",
    "calc_end": "<|calc_end|>",
    "synthesize": "<|synthesize|>",
    
    # 3. Chat Alignment / Dialogue Framing Tokens
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}


@dataclass
class AviGPTConfig:
    """Configuration class for the AviGPT-250M model."""
    
    # Vocabulary & Sequence
    vocab_size: int = 32000               # 100% Custom AviGPT Byte-Level BPE Vocabulary
    max_position_embeddings: int = 2048   # Pretraining context window (extensible to 4096+)
    
    # Architectural Dimensions (MobileLLM Deep-Thin Principle)
    hidden_size: int = 1024              # d_model: balanced width
    intermediate_size: int = 2096        # SwiGLU hidden dimension (calibrated for exact 250.3M params with 32k vocab)
    num_hidden_layers: int = 24          # Deep 24 layers for strong multi-step reasoning
    
    # Attention Setup
    num_attention_heads: int = 16        # 16 Query heads (head_dim = 64)
    num_key_value_heads: int = 4         # 4 KV heads (4:1 Grouped-Query Attention -> 75% smaller KV cache)
    head_dim: int = 64                   # 1024 / 16
    
    # Rotary Positional Embeddings
    rope_theta: float = 100000.0         # High base frequency for long context stability
    
    # Normalization & Stability
    rms_norm_eps: float = 1e-5           # Epsilon for RMSNorm stability
    qk_norm: bool = True                 # RMSNorm on Q and K vectors before attention (prevents logit blowup)
    
    # Embedding Tying
    tie_word_embeddings: bool = True     # Shares input embedding weights with lm_head (saves 50.3M params!)
    
    # Acceleration & Kernels
    use_liger_kernel: bool = True        # Uses LinkedIn Triton Liger Kernels on Linux/CUDA (60% VRAM saving)
    attn_implementation: str = "sdpa"    # PyTorch Scaled Dot-Product Attention / FlashAttention
    
    # Internal Non-RAG Product-Key Memory Layer (Sub-millisecond Tensor Memory)
    use_internal_memory: bool = False    # When enabled, injects parametric associative memory layer
    internal_memory_layer: int = 12      # Middle layer (Layer 12) where semantic abstraction peaks
    internal_memory_subkeys: int = 128   # 128 x 128 = 16,384 addressable compressed factual slots
    internal_memory_key_dim: int = 64    # Key projection dimension per sub-key
    internal_memory_val_dim: int = 256   # Compressed factual memory value dimension
    internal_memory_topk: int = 4        # Top-K sparse associative slots fetched per token

    # Special Token Registry
    special_tokens: Dict[str, str] = field(default_factory=lambda: SPECIAL_TOKENS)
    
    # Initialization
    initializer_range: float = 0.02
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# Training & Smoke Test Presets
PRESETS = {
    # Micro validation preset (runs instantly on any laptop or CPU)
    "smoke": AviGPTConfig(
        vocab_size=32000,
        max_position_embeddings=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    ),
    
    # Full production 250M parameter target
    "real_250m": AviGPTConfig(
        vocab_size=32000,
        max_position_embeddings=2048,
        hidden_size=1024,
        intermediate_size=2096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        tie_word_embeddings=True,
        qk_norm=True,
        rope_theta=100000.0,
    ),
    
    # Memory-Augmented 250M parameter target (Internal Non-RAG Product-Key Memory)
    "real_250m_memory": AviGPTConfig(
        vocab_size=32000,
        max_position_embeddings=2048,
        hidden_size=1024,
        intermediate_size=2096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        tie_word_embeddings=True,
        qk_norm=True,
        rope_theta=100000.0,
        use_internal_memory=True,
        internal_memory_layer=12,
        internal_memory_subkeys=128,  # 16,384 addressable internal factual slots
        internal_memory_key_dim=64,
        internal_memory_val_dim=256,
        internal_memory_topk=4,
    ),
}
