"""
Untied (Transformer) and looped (LoopedTransformer) decoder-only models.

LLaMA-v2-style blocks: RoPE, SwiGLU, pre-norm RMSNorm, causal attention.

LoopedTransformer adds:
- one block of L weight-shared layers iterated K times (no prelude or coda)
- a learned iteration embedding tau_k added before each iteration
- zero-initialized output projections for stable deep unrolling
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
# Standard Transformer (untied baseline)
# ---------------------------------------------------------------------------

class Transformer(nn.Module):
    """Standard transformer with N distinct layers, each applied once (deep-N)."""

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

    def zero_init_output_projections(self):
        """Zero-init o_proj (attention) and w3 (FFN), as LoopedTransformer does.

        The paper's untied runs keep the default initialization (--no_zero_init);
        the option exists so that the choice is explicit and recorded in args.json.
        """
        for layer in self.layers:
            nn.init.zeros_(layer.attention.o_proj.weight)
            nn.init.zeros_(layer.feed_forward.w3.weight)

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
    """Transformer with a block of L weight-shared layers iterated K times.

    Architecture: embed -> K x (add tau_k, block of L layers) -> norm -> output.
    Every layer is in the loop; there is no prelude, coda or input re-injection.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        n_recurrent_layers: int,
        n_heads: int,
        max_sequence_len: int,
        K_max: int,
        causal: bool = True,
        feed_forward_multiple_of: int = 1,
        pad_token_idx: int = None,
        norm_eps: float = 1e-5,
        use_timestep_embedding: bool = True,
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.K_max = K_max
        self.n_recurrent_layers = n_recurrent_layers
        self.use_timestep_embedding = use_timestep_embedding

        self.token_embeddings = nn.Embedding(vocab_size, dim, pad_token_idx)

        # Rotary positional encoding (NoPE when use_positional_embedding=False)
        if use_positional_embedding:
            self.pos_embed_func = RotaryPositionEmbedding(
                dim=dim // n_heads, max_sequence_len=max_sequence_len
            )
        else:
            self.pos_embed_func = None

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

        # Iteration embeddings: one learned vector per iteration
        if use_timestep_embedding:
            self.timestep_embeddings = nn.Embedding(K_max, dim)
            nn.init.normal_(self.timestep_embeddings.weight, std=0.02)

        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.out_proj = nn.Linear(dim, vocab_size, bias=False)

    def zero_init_output_projections(self):
        """Zero-init o_proj (attention) and w3 (FFN) for stable deep unrolling.

        Following Kohli et al. (arXiv 2604.07822): each iteration starts as an
        identity-like map, which keeps the input-output Jacobian stable even
        when unrolled for many iterations.
        """
        for layer in self.recurrent_layers:
            nn.init.zeros_(layer.attention.o_proj.weight)
            nn.init.zeros_(layer.feed_forward.w3.weight)

    def effective_depth(self, K: int = None) -> int:
        """Layer evaluations in a forward pass (L * K)."""
        K = K or self.K_max
        return K * self.n_recurrent_layers

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
        for k in range(K):
            if self.use_timestep_embedding:
                t_idx = min(k, self.K_max - 1)
                x = x + self.timestep_embeddings.weight[t_idx]
            for layer in self.recurrent_layers:
                x = layer(x, attention_mask)
        x = self.norm(x)
        return self.out_proj(x)

    def get_intermediate_states(
        self,
        tokens: torch.Tensor,
        attention_mask=None,
        K: int = None,
        per_layer: bool = False,
    ) -> list[torch.Tensor]:
        """Run a forward pass and return the hidden state after each iteration.

        Returns K tensors, each (batch_size, seq_len, dim). With per_layer=True,
        returns K * n_recurrent_layers tensors, one after every layer inside
        every iteration (block boundaries every n_recurrent_layers entries), for
        depth-matched comparison with untied stacks.
        """
        K = K or self.K_max
        states = []
        x = self.token_embeddings(tokens)
        for k in range(K):
            if self.use_timestep_embedding:
                t_idx = min(k, self.K_max - 1)
                x = x + self.timestep_embeddings.weight[t_idx]
            for layer in self.recurrent_layers:
                x = layer(x, attention_mask)
                if per_layer:
                    states.append(x.detach().clone())
            if not per_layer:
                states.append(x.detach().clone())
        return states
