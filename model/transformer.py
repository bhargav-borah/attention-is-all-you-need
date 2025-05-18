import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len, d_model, hidden_dim, num_heads=8, causal=False, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0, 'hidden_dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.seq_len = seq_len
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.W_q = nn.Linear(d_model, hidden_dim)
        self.W_k = nn.Linear(d_model, hidden_dim)
        self.W_v = nn.Linear(d_model, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, d_model)
        self.register_buffer('causal_mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool() if causal else None)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

    def forward(self, q, k, v, padding_mask=None):
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.causal:
            scores = scores.masked_fill(self.causal_mask, float('-inf'))
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(q.size(0), q.size(1), -1)
        attn_output = self.W_o(attn_output)
        return attn_output

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, proj_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, proj_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(proj_dim, d_model)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class Encoder(nn.Module):
    def __init__(self, seq_len, d_model, hidden_dim, proj_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(seq_len=seq_len, d_model=d_model, hidden_dim=hidden_dim, causal=False)
        self.ffn = FeedForwardNetwork(d_model=d_model, proj_dim=proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        x = self.dropout(x)
        x = self.norm1(self.mha(q=x, k=x, v=x, padding_mask=padding_mask) + x)
        x = self.norm2(self.ffn(x) + x)
        return x

class Decoder(nn.Module):
    def __init__(self, seq_len, d_model, hidden_dim, proj_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.mha1 = MultiHeadAttention(seq_len=seq_len, d_model=d_model, hidden_dim=hidden_dim, causal=True, dropout=dropout)
        self.mha2 = MultiHeadAttention(seq_len=seq_len, d_model=d_model, hidden_dim=hidden_dim, causal=False, dropout=dropout)
        self.ffn = FeedForwardNetwork(d_model=d_model, proj_dim=proj_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_padding_mask=None, tgt_padding_mask=None):
        x = self.dropout(x)
        x = self.norm1(self.mha1(q=x, k=x, v=x, padding_mask=tgt_padding_mask) + x)
        x = self.norm2(self.mha2(q=x, k=enc_output, v=enc_output, padding_mask=src_padding_mask) + x)
        x = self.norm3(self.ffn(x) + x)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len=50, d_model=512, hidden_dim=512, proj_dim=2048, num_encoders=6, num_decoders=6):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.vocab_size = vocab_size + 3
        self.token_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embeddings = self.get_sinusoidal_position_embeddings()
        self.encoders = nn.ModuleList([Encoder(seq_len=seq_len, d_model=d_model, hidden_dim=hidden_dim, proj_dim=proj_dim) for _ in range(num_encoders)])
        self.decoders = nn.ModuleList([Decoder(seq_len=seq_len, d_model=d_model, hidden_dim=hidden_dim, proj_dim=proj_dim) for _ in range(num_decoders)])
        self.W_o = nn.Linear(d_model, self.vocab_size)

    def forward(self, encoder_input, decoder_input, src_padding_mask=None, tgt_padding_mask=None):
        encoder_input_embeddings = self.token_embeddings(encoder_input) + self.position_embeddings.to(encoder_input.device)
        decoder_input_embeddings = self.token_embeddings(decoder_input) + self.position_embeddings.to(decoder_input.device)
        encoder_output = encoder_input_embeddings
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, padding_mask=src_padding_mask)
        decoder_output = decoder_input_embeddings
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, encoder_output, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
        output_logits = self.W_o(decoder_output)
        return output_logits

    def get_sinusoidal_position_embeddings(self):
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe