import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForwardNetwork


class Decoder(nn.Module):
    """
    Transformer Decoder Layer as described in "Attention Is All You Need" paper.
    
    Each decoder layer consists of:
    1. Masked Multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention with encoder output
    4. Add & Norm
    5. Position-wise Feed-Forward Network
    6. Add & Norm
    
    Args:
        seq_len (int): Maximum sequence length
        d_model (int): Model dimension
        hidden_dim (int): Hidden dimension for attention
        proj_dim (int): Projection dimension for feed-forward network
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, seq_len, d_model, hidden_dim, proj_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Masked multi-head self-attention (with causal masking)
        self.mha1 = MultiHeadAttention(
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=hidden_dim,
            causal=True,
            dropout=dropout
        )
        
        # Multi-head cross-attention with encoder output
        self.mha2 = MultiHeadAttention(
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=hidden_dim,
            causal=False,
            dropout=dropout
        )
        
        # Position-wise Feed-Forward Network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            proj_dim=proj_dim,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass for decoder layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            enc_output (torch.Tensor): Encoder output tensor of shape [batch_size, src_seq_len, d_model]
            src_padding_mask (torch.Tensor, optional): Boolean mask for source padding tokens
                                                      (True for pad tokens). Defaults to None.
            tgt_padding_mask (torch.Tensor, optional): Boolean mask for target padding tokens
                                                      (True for pad tokens). Defaults to None.
                                                      
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Apply dropout to input
        x = self.dropout(x)
        
        # Masked multi-head self-attention with residual connection and layer normalization
        attn_output1 = self.mha1(q=x, k=x, v=x, padding_mask=tgt_padding_mask)
        x = self.norm1(attn_output1 + x)
        
        # Multi-head cross-attention with encoder output
        attn_output2 = self.mha2(q=x, k=enc_output, v=enc_output, padding_mask=src_padding_mask)
        x = self.norm2(attn_output2 + x)
        
        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(x)
        x = self.norm3(ffn_output + x)
        
        return x


class DecoderStack(nn.Module):
    """
    Stack of Transformer Decoder layers.
    
    Args:
        num_layers (int): Number of decoder layers
        seq_len (int): Maximum sequence length
        d_model (int): Model dimension
        hidden_dim (int): Hidden dimension for attention
        proj_dim (int): Projection dimension for feed-forward network
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, num_layers, seq_len, d_model, hidden_dim, proj_dim, dropout=0.1):
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList([
            Decoder(
                seq_len=seq_len,
                d_model=d_model,
                hidden_dim=hidden_dim,
                proj_dim=proj_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, enc_output, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass for decoder stack.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            enc_output (torch.Tensor): Encoder output tensor of shape [batch_size, src_seq_len, d_model]
            src_padding_mask (torch.Tensor, optional): Boolean mask for source padding tokens
                                                     (True for pad tokens). Defaults to None.
            tgt_padding_mask (torch.Tensor, optional): Boolean mask for target padding tokens
                                                     (True for pad tokens). Defaults to None.
                                                     
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, enc_output, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        return x