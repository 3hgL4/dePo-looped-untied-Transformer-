"""
Transformer and LoopedTransformer models.

Base transformer is LLaMA-v2-style: RoPE, SwiGLU, RMSNorm, causal attention.

LoopedTransformer adds:
- Weight-shared recurrent layers iterated K times
- Optional prelude/coda layers (ETD-style)
- Learned timestep embeddings per iteration (TMLT-style)
- Optional NoPE (no positional embeddings) for depth extrapolation
- Zero-init output projections for stable deep unrolling
- K configurable at inference time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (from picoGPT)
# ---------------------------------------------------------------------------

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_sequence_len: int, theta=10000.0):
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        t = torch.arange(max_sequence_len, device=frequencies.device)
        frequencies = torch.outer(t, frequencies).float()
        self.register_buffer('_cos', torch.cos(frequencies), persistent=False)
        self.register_buffer('_sin', torch.sin(frequencies), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (..., sequence_len, dim)
        :return: (..., sequence_len, dim)
        """
        assert x.shape[-2] <= self._sin.shape[0]
        seq_len = x.shape[-2]
        sin, cos = self._sin[:seq_len], self._cos[:seq_len]
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x_real, x_imag = x_pairs[..., 0], x_pairs[..., 1]
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        return torch.stack([out_real, out_imag], dim=-1).flatten(-2).type_as(x)


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 1):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1_w2 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out, w2_out = self.w1_w2(x).chunk(2, dim=-1)
        return self.w3(F.silu(w1_out) * w2_out)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        causal: bool = False,
        positional_embedding_func=None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.causal = causal
        self.positional_embedding_func = positional_embedding_func
        self.qkv_proj = nn.Linear(dim, 3 * n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(head_dim * n_heads, dim, bias=False)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        batch_size, sequence_len, _ = x.shape
        qkv = self.qkv_proj(x).view(
            batch_size, sequence_len, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        if self.positional_embedding_func is not None:
            query = self.positional_embedding_func(query)
            key = self.positional_embedding_func(key)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]

        attn_output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            is_causal=self.causal,
            attn_mask=attn_mask,
        )
        attn_output = attn_output.permute(0, 2, 1, 3).flatten(-2)
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        causal: bool = False,
        feed_forward_multiple_of: int = 1,
        positional_embedding_func=None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim=dim,
            head_dim=dim // n_heads,
            n_heads=n_heads,
            causal=causal,
            positional_embedding_func=positional_embedding_func,
        )
        self.feed_forward = FeedForwardSwiGLU(
            dim=dim, hidden_dim=4 * dim, multiple_of=feed_forward_multiple_of
        )
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, attention_mask=None):
        x = x + self.attention(self.attention_norm(x), attention_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Standard Transformer (unlooped baseline)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Standard transformer with unique layers. Used as unlooped baseline."""

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        max_sequence_len: int,
        causal: bool = True,
        feed_forward_multiple_of: int = 1,
        pad_token_idx: int = None,
        norm_eps: float = 1e-5,
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, dim, pad_token_idx)
        if use_positional_embedding:
            self.pos_embed_func = RotaryPositionEmbedding(
                dim=dim // n_heads, max_sequence_len=max_sequence_len
            )
        else:
            self.pos_embed_func = None
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                causal=causal,
                feed_forward_multiple_of=feed_forward_multiple_of,
                positional_embedding_func=self.pos_embed_func,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, attention_mask=None):
        x = self.token_embeddings(tokens)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        return self.out_proj(x)


# ---------------------------------------------------------------------------
# Looped Transformer
# ---------------------------------------------------------------------------

class LoopedTransformer(nn.Module):
    """Transformer with weight-shared recurrent layers iterated K times.

    Architecture: embed -> prelude -> K x recurrent -> coda -> norm -> output

    Ouro-style (all shared): n_prelude_layers=0, n_coda_layers=0
    ETD-style (prelude/recurrent/coda): non-zero prelude and coda

    Features:
    - Learned timestep embedding per iteration (TMLT-style, critical for expressivity)
    - Optional input injection (Parcae-style, off by default for Kohli comparability)
    - Optional NoPE (no positional embeddings, recommended for depth extrapolation)
    - Zero-init output projections for stable deep unrolling (Kohli et al.)
    - K configurable at inference time
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_recurrent_layers: int,
        n_heads: int,
        max_sequence_len: int,
        K_max: int,
        n_prelude_layers: int = 0,
        n_coda_layers: int = 0,
        causal: bool = True,
        feed_forward_multiple_of: int = 1,
        pad_token_idx: int = None,
        norm_eps: float = 1e-5,
        use_timestep_embedding: bool = True,
        use_input_injection: bool = False,
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.K_max = K_max
        self.n_recurrent_layers = n_recurrent_layers
        self.use_timestep_embedding = use_timestep_embedding
        self.use_input_injection = use_input_injection

        # Embedding
        self.token_embeddings = nn.Embedding(vocab_size, dim, pad_token_idx)

        # Positional encoding (NoPE when use_positional_embedding=False)
        if use_positional_embedding:
            self.pos_embed_func = RotaryPositionEmbedding(
                dim=dim // n_heads, max_sequence_len=max_sequence_len
            )
        else:
            self.pos_embed_func = None

        # Prelude layers (unique, not looped)
        self.prelude_layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                causal=causal,
                feed_forward_multiple_of=feed_forward_multiple_of,
                positional_embedding_func=self.pos_embed_func,
                norm_eps=norm_eps,
            )
            for _ in range(n_prelude_layers)
        ])

        # Recurrent layers (weight-shared, iterated K times)
        self.recurrent_layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                causal=causal,
                feed_forward_multiple_of=feed_forward_multiple_of,
                positional_embedding_func=self.pos_embed_func,
                norm_eps=norm_eps,
            )
            for _ in range(n_recurrent_layers)
        ])

        # Coda layers (unique, not looped)
        self.coda_layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                causal=causal,
                feed_forward_multiple_of=feed_forward_multiple_of,
                positional_embedding_func=self.pos_embed_func,
                norm_eps=norm_eps,
            )
            for _ in range(n_coda_layers)
        ])

        # Timestep embeddings (TMLT-style): one learned vector per iteration
        if use_timestep_embedding:
            self.timestep_embeddings = nn.Embedding(K_max, dim)
            nn.init.normal_(self.timestep_embeddings.weight, std=0.02)

        # Output
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def zero_init_output_projections(self):
        """Zero-init o_proj (attention) and w3 (FFN) for stable deep unrolling.

        Following Kohli et al. (2604.07822): ensures each recurrent iteration
        is initially an identity-like operation, stabilizing the input-output
        Jacobian even when unrolled to many iterations.
        """
        for layer in self.recurrent_layers:
            nn.init.zeros_(layer.attention.o_proj.weight)
            nn.init.zeros_(layer.feed_forward.w3.weight)

    def effective_depth(self, K: int = None) -> int:
        """Total sequential layers in a forward pass."""
        K = K or self.K_max
        return (
            len(self.prelude_layers)
            + K * self.n_recurrent_layers
            + len(self.coda_layers)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask=None,
        K: int = None,
    ):
        """
        Args:
            tokens: (batch_size, seq_len) long tensor
            attention_mask: (batch_size, seq_len) bool tensor
            K: number of loop iterations (default: K_max)
        """
        K = K or self.K_max

        x = self.token_embeddings(tokens)

        # Prelude
        for layer in self.prelude_layers:
            x = layer(x, attention_mask)

        # Save initial state for input injection (off by default)
        if self.use_input_injection:
            x_init = x

        # Recurrent block: K iterations of the shared layers (full BPTT)
        for k in range(K):
            # Input injection: add initial embedding back (stabilizes fixed points)
            if self.use_input_injection and k > 0:
                x = x + x_init

            # Add timestep embedding for this iteration
            if self.use_timestep_embedding:
                t_idx = min(k, self.K_max - 1)
                x = x + self.timestep_embeddings.weight[t_idx]

            # Apply all recurrent layers
            for layer in self.recurrent_layers:
                x = layer(x, attention_mask)

        # Coda
        for layer in self.coda_layers:
            x = layer(x, attention_mask)

        x = self.norm(x)
        return self.out_proj(x)

    def get_intermediate_states(
        self,
        tokens: torch.Tensor,
        attention_mask=None,
        K: int = None,
    ) -> list[torch.Tensor]:
        """Run forward pass and return hidden state after each loop iteration.

        Useful for linear probing and trajectory analysis (H2 experiments).
        Returns list of K tensors, each (batch_size, seq_len, dim).
        """
        K = K or self.K_max
        states = []

        x = self.token_embeddings(tokens)

        for layer in self.prelude_layers:
            x = layer(x, attention_mask)

        if self.use_input_injection:
            x_init = x

        for k in range(K):
            if self.use_input_injection and k > 0:
                x = x + x_init

            if self.use_timestep_embedding:
                t_idx = min(k, self.K_max - 1)
                x = x + self.timestep_embeddings.weight[t_idx]

            for layer in self.recurrent_layers:
                x = layer(x, attention_mask)

            states.append(x.detach().clone())

        return states
