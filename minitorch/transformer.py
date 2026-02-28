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
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
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

        ### BEGIN ASSIGN3_3
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3

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
        ### BEGIN ASSIGN3_3
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
        ### END ASSIGN3_3
    
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
        
        ### BEGIN ASSIGN3_3
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
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
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
        ### END ASSIGN3_3


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
        ### BEGIN ASSIGN3_3
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with  activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
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
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.attention = MultiHeadAttention(n_embd, n_head, p_dropout=p_dropout, bias=bias, backend=backend)
        self.ff = FeedForward(n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        # Pre-LN architecture:
        # 1. Apply LN before attention, then attention + residual
        # 2. Apply LN before FF, then FF + residual
        attn_output = self.attention(self.ln_1(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd))
        x = x + attn_output
        
        ff_output = self.ff(self.ln_2(x.view(batch_size * seq_len, n_embd)).view(batch_size, seq_len, n_embd))
        x = x + ff_output
        
        return x
        ### END YOUR SOLUTION


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
        backend: TensorBackend=None
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
        ### BEGIN ASSIGN3_3
        self.token_embeddings = Embedding(n_vocab, n_embd, backend=backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend=backend)
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout=p_dropout, ln_eps=ln_eps, bias=bias, backend=backend)
        self.dropout = Dropout(p_dropout)
        self.ln = LayerNorm1d(n_embd, ln_eps, backend=backend)
        self.lm_head = Linear(n_embd, n_vocab, bias=bias, backend=backend)
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
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
        ### END ASSIGN3_3
