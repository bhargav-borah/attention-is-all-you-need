import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForwardNetwork


class Encoder(nn.Module):
    """
    Transformer Encoder Layer as described in "Attention Is All You Need" paper.
    
    Each encoder layer consists of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Position-wise Feed-Forward Network
    4. Add & Norm
    
    Args:
        seq_len (int): Maximum sequence length
        d_model (int): Model dimension
        hidden_dim (int): Hidden dimension for attention
        proj_dim (int): Projection dimension for feed-forward network
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, seq_len, d_model, hidden_dim, proj_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Multi-head attention (self-attention)
        self.mha = MultiHeadAttention(
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=hidden_dim,
            causal=False
        )
        
        # Position-wise Feed-Forward Network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            proj_dim=proj_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask=None):
        """
        Forward pass for encoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            padding_mask (torch.Tensor, optional): Boolean mask for padding tokens
                                                  (True for pad tokens). Defaults to None.
                                                  
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Apply dropout to input
        x = self.dropout(x)
        
        # Multi-head self-attention with residual connection and layer normalization
        # Note: Apply layer normalization to the output of attention + residual
        attn_output = self.mha(q=x, k=x, v=x, padding_mask=padding_mask)
        x = self.norm1(attn_output + x)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(x)
        x = self.norm2(ffn_output + x)
        
        return x


class EncoderStack(nn.Module):
    """
    Stack of Transformer Encoder layers.
    
    Args:
        num_layers (int): Number of encoder layers
        seq_len (int): Maximum sequence length
        d_model (int): Model dimension
        hidden_dim (int): Hidden dimension for attention
        proj_dim (int): Projection dimension for feed-forward network
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, num_layers, seq_len, d_model, hidden_dim, proj_dim, dropout=0.1):
        super(EncoderStack, self).__init__()
        self.layers = nn.ModuleList([
            Encoder(
                seq_len=seq_len,
                d_model=d_model,
                hidden_dim=hidden_dim,
                proj_dim=proj_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, padding_mask=None):
        """
        Forward pass for encoder stack.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            padding_mask (torch.Tensor, optional): Boolean mask for padding tokens
                                                  (True for pad tokens). Defaults to None.
                                                  
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x