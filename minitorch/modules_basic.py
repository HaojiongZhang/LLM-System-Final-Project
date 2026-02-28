"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weights : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        self.weights = Parameter(
            rand((num_embeddings, embedding_dim), backend=backend),
            name="embedding_weight"
        )
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        # Flatten x to (bs * seq_len,) for one-hot encoding
        x_flat = x.view(bs * seq_len)
        # Convert to one-hot: (bs * seq_len, num_embeddings)
        one_hot_x = one_hot(x_flat, self.num_embeddings)
        # Project to embeddings: (bs * seq_len, embedding_dim)
        embeddings = one_hot_x @ self.weights.value
        # Reshape to (bs, seq_len, embedding_dim)
        output = embeddings.view(bs, seq_len, self.embedding_dim)
        return output
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)

        Note: If p_dropout is 0, directly return the input tensor. Otherwise, the random seed may cause problems
        """
        ### BEGIN ASSIGN3_2
        if self.p_dropout == 0 or not self.training:
            return x
        # Use np.random.binomial to generate mask
        # mask shape should be same as x, with values 0 or 1
        mask_np = np.random.binomial(1, 1 - self.p_dropout, size=x.shape)
        mask = tensor_from_numpy(mask_np.astype(np.float32), backend=x.backend)
        # Scale by (1 - p_dropout) to maintain expected value
        return x * mask / (1 - self.p_dropout)
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        ### BEGIN ASSIGN3_2
        self.in_size = in_size
        self.use_bias = bias
        self.backend = backend
        
        # Initialize weights from Uniform(-1/sqrt(in_size), 1/sqrt(in_size))
        bound = 1.0 / np.sqrt(in_size)
        weight_data = np.random.uniform(-bound, bound, (in_size, out_size))
        self.weights = Parameter(
            tensor_from_numpy(weight_data.astype(np.float32), backend=backend),
            name="weight"
        )
        
        # Initialize bias if needed
        if bias:
            bias_data = np.random.uniform(-bound, bound, (out_size,))
            self.bias = Parameter(
                tensor_from_numpy(bias_data.astype(np.float32), backend=backend),
                name="bias"
            )
        else:
            self.bias = None
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size)
        """
        batch, in_size = x.shape
        ### BEGIN ASSIGN3_2
        # x @ weights: (batch, in_size) @ (in_size, out_size) = (batch, out_size)
        output = x @ self.weights.value
        if self.use_bias:
            output = output + self.bias.value.view(1, self.out_size)
        return output
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        self.backend = backend
        # Initialize weights to 1 (gamma)
        self.weights = Parameter(
            ones((dim,), backend=backend),
            name="weight"
        )
        # Initialize bias to 0 (beta)
        self.bias = Parameter(
            zeros((dim,), backend=backend),
            name="bias"
        )
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        # Calculate mean over dimension 1 (features): shape (bs, 1)
        mean = x.mean(dim=1)
        # Calculate variance over dimension 1: shape (bs, 1)
        # Use the more stable formula: var = E[(x - mean)^2]
        centered = x - mean
        variance = (centered ** 2).mean(dim=1)
        # Normalize: add eps before sqrt for numerical stability
        std = (variance + self.eps) ** 0.5
        x_normalized = centered / std
        # Scale and shift with learnable parameters
        output = x_normalized * self.weights.value + self.bias.value
        return output
        ### END ASSIGN3_2
