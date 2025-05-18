import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network as described in "Attention Is All You Need" paper.
    
    This is a two-layer feed-forward network with ReLU activation in between.
    
    Args:
        d_model (int): Model dimension
        proj_dim (int): Projection dimension (hidden layer size)
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, d_model, proj_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, proj_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(proj_dim, d_model)
        
        # Initialize parameters with Xavier uniform
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        """
        Forward pass for feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)  # [batch_size, seq_len, proj_dim]
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)  # [batch_size, seq_len, d_model]
        return x