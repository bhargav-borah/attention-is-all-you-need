import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in "Attention Is All You Need" paper.
    
    This implementation supports both self-attention (same Q, K, V) and
    cross-attention (different sources for Q, K, V).
    
    Args:
        seq_len (int): Maximum sequence length
        d_model (int): Model dimension
        hidden_dim (int): Hidden dimension
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        causal (bool, optional): Whether to use causal mask for decoder. Defaults to False.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, seq_len, d_model, hidden_dim, num_heads=8, causal=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, 'hidden_dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.seq_len = seq_len
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections
        self.W_q = nn.Linear(d_model, hidden_dim)
        self.W_k = nn.Linear(d_model, hidden_dim)
        self.W_v = nn.Linear(d_model, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, d_model)
        
        # Initialize parameters with Xavier uniform
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        
        # Create causal mask if needed
        self.register_buffer(
            'causal_mask', 
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() if causal else None
        )
        
    def forward(self, q, k, v, padding_mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model]
            padding_mask (torch.Tensor, optional): Boolean mask for padding tokens
                                                  (True for pad tokens). Defaults to None.
                                                  
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        
        # Linear projections
        Q = self.W_q(q)  # [batch_size, seq_len, hidden_dim]
        K = self.W_k(k)  # [batch_size, seq_len, hidden_dim]
        V = self.W_v(v)  # [batch_size, seq_len, hidden_dim]
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if specified (for decoder self-attention)
        if self.causal:
            scores = scores.masked_fill(self.causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back to [batch_size, seq_len, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Final linear projection
        attn_output = self.W_o(attn_output)  # [batch_size, seq_len, d_model]
        
        return attn_output