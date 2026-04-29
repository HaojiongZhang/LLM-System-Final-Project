import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .tensor_functions import flash_attention
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .paged_attention import BlockManager

# Import the C kernel wrapper.  paged_attn.so may not be compiled yet on
# first run; falls back to gather_kv in that case.
try:
    from .cuda_kernel_ops import CudaKernelOps, PAGED_ATTN_AVAILABLE
except Exception:
    CudaKernelOps = None          # type: ignore[assignment,misc]
    PAGED_ATTN_AVAILABLE = False

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_flash_attn: bool=False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
            use_flash_attn: If True, use FA2 backward via FlashAttentionFunc

        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head
        self.use_flash_attn = use_flash_attn

        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        # Use -1e9 instead of -inf for better numerical stability
        mask = -1e9 * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        # Project to Q, K, V: each of shape (batch_size, seq_len, n_embd)
        q = self.q_projection(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        k = self.k_projection(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        v = self.v_projection(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd)
        
        # Reshape Q: (batch_size, seq_len, n_embd) -> (batch_size, seq_len, n_head, attn_hidden_dim)
        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        # Permute to (batch_size, n_head, seq_len, attn_hidden_dim)
        q = q.permute(0, 2, 1, 3)
        
        # Reshape K and permute similarly
        k = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        k = k.permute(0, 2, 1, 3)
        # Transpose last two dimensions for K: (batch_size, n_head, attn_hidden_dim, seq_len)
        kT = k.permute(0, 1, 3, 2)
        
        # Reshape V similarly to Q
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)
        v = v.permute(0, 2, 1, 3)
        
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None

        # Flash path: use the backend flash op when available, otherwise fall
        # back to FlashAttentionFunc which keeps the FA2-style backward path.
        # Keep semantics aligned with the dense path by only using it when
        # attention dropout is effectively inactive.
        if self.use_flash_attn and (self.dropout.p_dropout == 0.0 or not self.training):
            k = kT.permute(0, 1, 3, 2)
            out_bhsd = flash_attention(
                q,
                k,
                v,
                causal=self.causal,
                softmax_scale=1.0 / (self.attn_hidden_dim ** 0.5),
            )
            return out_bhsd.permute(0, 2, 1, 3).contiguous().view(
                batch_size, queries_len, self.n_embd
            )

        # Compute q @ kT: (batch, num_head, seq_len, q_dim) @ (batch, num_head, q_dim, seq_len)
        #              = (batch, num_head, seq_len, seq_len)
        scores = q @ kT / (self.attn_hidden_dim ** 0.5)
        
        # Apply causal mask if needed
        if self.causal:
            mask = self.create_causal_mask(queries_len)
            scores = scores + mask
        
        # Apply softmax
        attn_weights = softmax(scores, dim=3)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Multiply by V: (batch, num_head, seq_len, seq_len) @ (batch, num_head, seq_len, v_dim)
        #             = (batch, num_head, seq_len, v_dim)
        attn_output = attn_weights @ v
        
        # Permute back: (batch, num_head, seq_len, v_dim) -> (batch, seq_len, num_head, v_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        
        # Reshape: (batch, seq_len, num_head, v_dim) -> (batch, seq_len, n_embd)
        # Ensure contiguous after permute before viewing
        result = attn_output.contiguous().view(batch_size, queries_len, self.n_embd)
        
        return result

        return result

    def forward(self, x, block_manager=None, layer_idx=None, seq_id=None):
        """
        Compute multi-head attention with optional causal masking.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            block_manager: BlockManager instance, or None during training.
            layer_idx (int): Which transformer layer this is (for indexing the cache).
            seq_id (int): Sequence identifier used to look up the block table.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        if block_manager is not None:
            return self._forward_paged(x, block_manager, layer_idx, seq_id)

        batch_size, seq_len, n_embd = x.shape
        # Project to Q, K, V
        q, kT, v = self.project_to_query_key_value(x)

        # Compute self-attention
        attn_output = self.self_attention(q, kT, v)

        # Apply output projection
        # Shape: (batch_size, seq_len, n_embd) -> flattened to (batch_size * seq_len, n_embd)
        output = self.out_projection(attn_output.contiguous().view(batch_size * seq_len, n_embd))
        # Reshape back to (batch_size, seq_len, n_embd)
        output = output.view(batch_size, seq_len, n_embd)

        return output

    def _forward_paged(self, x, block_manager, layer_idx, seq_id):
        """
        Attention forward pass for cached inference.

        seq_len > 1  (prefill):  runs standard causal self-attention and writes
                                  every token's K/V into the block manager.
        seq_len == 1, seq_id int (single decode):  writes the new token's K/V,
                                  then attends over the full history via the
                                  C kernel (or gather fallback).
        seq_len == 1, seq_id list (batched decode):  same but for B sequences
                                  at different positions in one forward pass.
        """
        batch_size, seq_len, n_embd = x.shape

        q, kT, v = self.project_to_query_key_value(x)
        # kT: (batch, n_head, head_dim, seq_len)
        # v:  (batch, n_head, seq_len,  head_dim)

        # Un-transpose K for cache writes. No grad needed during inference.
        k_np = kT.to_numpy().transpose(0, 1, 3, 2)  # (batch, n_head, seq_len, head_dim)
        v_np = v.to_numpy()

        # start_pos: current cache length for single-sequence modes.
        # Batched decode handles positions per-sequence inside its branch.
        start_pos = block_manager.seq_lengths[seq_id] if not isinstance(seq_id, list) else 0
        use_paged_kernel = PAGED_ATTN_AVAILABLE and block_manager.has_device_mirror()

        # Write K/V for prefill and single-sequence decode.
        # Batched decode writes inside its own branch below.
        if not isinstance(seq_id, list):
            for t in range(seq_len):
                block_manager.write_kv(
                    layer_idx, seq_id, start_pos + t, k_np[0, :, t, :], v_np[0, :, t, :]
                )

        if seq_len > 1:
            # Prefill: causal attention, same math as training.
            attn_output = self.self_attention(q, kT, v)

        elif isinstance(seq_id, list):
            # Batched decode: seq_id is a list of B integers.
            # k_np / v_np shape: (B, n_head, 1, head_dim).
            seq_ids = seq_id
            B = len(seq_ids)
            positions = [block_manager.seq_lengths[sid] for sid in seq_ids]

            for b, sid in enumerate(seq_ids):
                block_manager.write_kv(
                    layer_idx, sid, positions[b], k_np[b, :, 0, :], v_np[b, :, 0, :]
                )

            counts = [p + 1 for p in positions]

            if use_paged_kernel:
                q_kernel = q.contiguous().view(B, self.n_head, self.attn_hidden_dim)
                attn_out_t = CudaKernelOps.paged_attention(
                    q_kernel,
                    block_manager.device_ptr_k(layer_idx),
                    block_manager.device_ptr_v(layer_idx),
                    block_manager.get_block_tables_np(seq_ids),
                    np.array(counts, dtype=np.int32),
                    layer_idx,
                    block_manager.block_size,
                    scale=1.0 / (self.attn_hidden_dim ** 0.5),
                    n_blocks=block_manager.num_blocks,
                )  # (B, n_head, head_dim)
                attn_output = attn_out_t.view(B, 1, self.n_embd)
            else:
                k_batch_np, v_batch_np = block_manager.gather_kv_padded(
                    layer_idx, seq_ids, counts
                )
                max_len = max(counts)

                k_batch_t = tensor_from_numpy(k_batch_np, backend=self.backend)
                v_batch_t = tensor_from_numpy(v_batch_np, backend=self.backend)
                kT_batch  = k_batch_t.permute(0, 1, 3, 2)  # (B, n_head, head_dim, max_len)

                # q (B,H,1,d) @ kT_batch (B,H,d,max_len) -> (B,H,1,max_len)
                scores = q @ kT_batch / (self.attn_hidden_dim ** 0.5)

                # Mask padding positions to -inf before softmax.
                mask_np = np.full((B, 1, 1, max_len), -1e9, dtype=np.float32)
                for b, cnt in enumerate(counts):
                    mask_np[b, 0, 0, :cnt] = 0.0
                scores = scores + tensor_from_numpy(mask_np, backend=self.backend)

                attn_weights = softmax(scores, dim=3)
                attn_weights = self.dropout(attn_weights)

                attn_output = (attn_weights @ v_batch_t).permute(0, 2, 1, 3).contiguous().view(
                    B, 1, self.n_embd
                )

        else:
            # Single-sequence decode.
            total_len = start_pos + 1

            if use_paged_kernel:
                q_kernel = q.contiguous().view(1, self.n_head, self.attn_hidden_dim)
                attn_out_t = CudaKernelOps.paged_attention(
                    q_kernel,
                    block_manager.device_ptr_k(layer_idx),
                    block_manager.device_ptr_v(layer_idx),
                    block_manager.get_block_tables_np([seq_id]),
                    np.array([total_len], dtype=np.int32),
                    layer_idx,
                    block_manager.block_size,
                    scale=1.0 / (self.attn_hidden_dim ** 0.5),
                    n_blocks=block_manager.num_blocks,
                )  # (1, n_head, head_dim)
                attn_output = attn_out_t.view(1, 1, self.n_embd)
            else:
                k_full_np, v_full_np = block_manager.gather_kv(layer_idx, seq_id, total_len)
                k_full_t = tensor_from_numpy(k_full_np, backend=self.backend)
                v_full_t = tensor_from_numpy(v_full_np, backend=self.backend)
                kT_full  = k_full_t.permute(0, 1, 3, 2)  # (1, n_head, head_dim, total_len)

                scores = q @ kT_full / (self.attn_hidden_dim ** 0.5)
                attn_weights = softmax(scores, dim=3)
                attn_weights = self.dropout(attn_weights)

                attn_output = (attn_weights @ v_full_t).permute(0, 2, 1, 3).contiguous().view(
                    1, 1, self.n_embd
                )

        output = self.out_projection(attn_output.contiguous().view(batch_size * seq_len, n_embd))
        output = output.view(batch_size, seq_len, n_embd)
        return output


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        """
        Forward pass through feed-forward network with  activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None, use_flash_attn: bool=False):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.

        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            use_flash_attn (bool): Use FA2 backward in attention, default False

        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd, n_head, p_dropout=p_dropout, bias=bias, backend=backend, use_flash_attn=use_flash_attn)
        self.ff = FeedForward(n_embd, p_dropout=p_dropout, bias=bias, backend=backend)

    def forward(self, x, block_manager=None, layer_idx=None, seq_id=None):
        """
        Forward pass through transformer layer with pre-layer normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
            block_manager: BlockManager instance, or None during training.
            layer_idx (int): Layer index, used to index into the KV cache.
            seq_id (int): Sequence identifier for the block table lookup.

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        # Pre-LN architecture:
        # 1. Apply LN before attention, then attention + residual
        # 2. Apply LN before FF, then FF + residual
        attn_output = self.attention(
            self.ln_1(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd),
            block_manager=block_manager, layer_idx=layer_idx, seq_id=seq_id,
        )
        x = x + attn_output

        ff_output = self.ff(self.ln_2(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd))
        x = x + ff_output

        return x


class DecoderLM(Module):
    def __init__(
        self,
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5,
        bias: bool=True,
        backend: TensorBackend=None,
        use_flash_attn: bool=False,
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_flash_attn=use_flash_attn)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_flash_attn=use_flash_attn)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_flash_attn=use_flash_attn)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend, use_flash_attn=use_flash_attn)
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        token_emb = self.token_embeddings(idx)
        
        # 2. Create positional embeddings
        # Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        pos_ids = tensor_from_numpy(
            np.arange(seq_len, dtype=datatype)[np.newaxis, :],
            backend=self.backend
        ).contiguous().view(1, seq_len)
        # Convert position ids to int for indexing
        pos_ids_int = tensor_from_numpy(
            np.arange(seq_len, dtype=np.int32)[np.newaxis, :],
            backend=self.backend
        ).contiguous().view(1, seq_len)
        pos_emb = self.position_embeddings(pos_ids_int)  # (1, seq_len, n_embd)
        
        # 3. Add embeddings and apply dropout
        x = token_emb + pos_emb  # Broadcasting: (batch_size, seq_len, n_embd)
        x = self.dropout(x)
        
        # 4. Pass through transformer layers
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        
        # 5. Apply final layer normalization
        x = self.ln(x.view(batch_size * seq_len, self.n_embd)).view(batch_size, seq_len, self.n_embd)
        
        # 6. Project to vocabulary size
        logits = self.lm_head(x.view(batch_size * seq_len, self.n_embd)).view(batch_size, seq_len, self.n_vocab)

        return logits

    def prefill(self, idx, seq_id, block_manager):
        """
        Process the full prompt (first forward pass for a sequence).

        Runs identically to forward() but passes the block manager down to
        each transformer layer so attention can write the prompt's K/V vectors
        into the cache.  After this call, block_manager.seq_lengths[seq_id]
        is set to the prompt length, ready for decode_step.

        Args:
            idx (Tensor): Token indices of shape (1, prompt_len).
            seq_id (int): Identifier for this sequence.
            block_manager (BlockManager): The cache to populate.

        Returns:
            Tensor: Logits of shape (1, prompt_len, n_vocab).
        """
        block_manager.allocate_seq(seq_id)

        batch_size, seq_len = idx.shape

        token_emb = self.token_embeddings(idx)

        pos_ids = tensor_from_numpy(
            np.arange(seq_len, dtype=np.int32)[np.newaxis, :],
            backend=self.backend,
        ).contiguous().view(1, seq_len)
        pos_emb = self.position_embeddings(pos_ids)

        x = self.dropout(token_emb + pos_emb)

        layers = [self.t_layer_1, self.t_layer_2, self.t_layer_3, self.t_layer_4]
        for layer_idx, layer in enumerate(layers):
            x = layer(x, block_manager=block_manager, layer_idx=layer_idx, seq_id=seq_id)

        block_manager.seq_lengths[seq_id] = seq_len

        x = self.ln(x.view(batch_size * seq_len, self.n_embd)).view(batch_size, seq_len, self.n_embd)
        logits = self.lm_head(x.view(batch_size * seq_len, self.n_embd)).view(
            batch_size, seq_len, self.n_vocab
        )
        return logits

    def decode_step(self, token_id, seq_id, block_manager):
        """
        Process one new token during autoregressive generation.

        Reads the full K/V history from the cache (via gather_kv inside
        attention) and writes the new token's K/V into it.  Only the new
        token's logit is returned, so the full sequence is not recomputed.

        Args:
            token_id (Tensor): Shape (1, 1), the single new token index.
            seq_id (int): Sequence identifier.
            block_manager (BlockManager): The live cache for this sequence.

        Returns:
            Tensor: Logits of shape (1, 1, n_vocab).
        """
        cur_pos = block_manager.seq_lengths[seq_id]

        token_emb = self.token_embeddings(token_id)
        pos_ids = tensor_from_numpy(
            np.array([[cur_pos]], dtype=np.int32),
            backend=self.backend,
        )
        pos_emb = self.position_embeddings(pos_ids)

        x = self.dropout(token_emb + pos_emb)

        layers = [self.t_layer_1, self.t_layer_2, self.t_layer_3, self.t_layer_4]
        for layer_idx, layer in enumerate(layers):
            x = layer(x, block_manager=block_manager, layer_idx=layer_idx, seq_id=seq_id)

        block_manager.seq_lengths[seq_id] += 1

        x = self.ln(x.view(1, self.n_embd)).view(1, 1, self.n_embd)
        logits = self.lm_head(x.view(1, self.n_embd)).view(1, 1, self.n_vocab)
        return logits

    def decode_step_batch(self, token_ids, seq_ids, block_manager):
        """
        Batched version of decode_step: process one new token per active
        sequence in a single forward pass.

        Each sequence can be at a completely different position in the KV cache.
        The attention kernel pads shorter histories and masks the padding out,
        so the results are identical to calling decode_step individually.

        Args:
            token_ids (Tensor): Shape (B, 1), one new token index per sequence.
            seq_ids (list[int]): B sequence identifiers.
            block_manager (BlockManager): The shared KV cache.

        Returns:
            Tensor: Logits of shape (B, 1, n_vocab).
        """
        B = len(seq_ids)
        positions = [block_manager.seq_lengths[sid] for sid in seq_ids]

        token_emb = self.token_embeddings(token_ids)
        pos_ids = tensor_from_numpy(
            np.array([[p] for p in positions], dtype=np.int32),
            backend=self.backend,
        )
        pos_emb = self.position_embeddings(pos_ids)

        x = self.dropout(token_emb + pos_emb)

        # seq_id=seq_ids (a list) routes each layer to the batched decode branch.
        layers = [self.t_layer_1, self.t_layer_2, self.t_layer_3, self.t_layer_4]
        for layer_idx, layer in enumerate(layers):
            x = layer(x, block_manager=block_manager, layer_idx=layer_idx, seq_id=seq_ids)

        for sid in seq_ids:
            block_manager.seq_lengths[sid] += 1

        x = self.ln(x.view(B, self.n_embd)).view(B, 1, self.n_embd)
        logits = self.lm_head(x.view(B, self.n_embd)).view(B, 1, self.n_vocab)
        return logits
