import torch
import torch.nn as nn
import math
from .encoder import EncoderStack
from .decoder import DecoderStack


class Transformer(nn.Module):
    """
    Transformer model as described in "Attention Is All You Need" paper.
    
    Args:
        vocab_size (int): Size of the vocabulary
        seq_len (int, optional): Maximum sequence length. Defaults to 50.
        d_model (int, optional): Model dimension. Defaults to 512.
        hidden_dim (int, optional): Hidden dimension for attention. Defaults to 512.
        proj_dim (int, optional): Projection dimension for feed-forward network. Defaults to 2048.
        num_encoders (int, optional): Number of encoder layers. Defaults to 6.
        num_decoders (int, optional): Number of decoder layers. Defaults to 6.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    def __init__(self, vocab_size, seq_len=50, d_model=512, hidden_dim=512, 
                 proj_dim=2048, num_encoders=6, num_decoders=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # Add 3 to vocab_size for special tokens (SOS, EOS, PAD)
        self.vocab_size = vocab_size + 3
        
        # Embeddings
        self.token_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embeddings = self.get_sinusoidal_position_embeddings()
        
        # Encoder and Decoder stacks
        self.encoder_stack = EncoderStack(
            num_layers=num_encoders,
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            dropout=dropout
        )
        
        self.decoder_stack = DecoderStack(
            num_layers=num_decoders,
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            dropout=dropout
        )
        
        # Output linear layer
        self.W_o = nn.Linear(d_model, self.vocab_size)
        
    def forward(self, encoder_input, decoder_input, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass for the Transformer model.
        
        Args:
            encoder_input (torch.Tensor): Source sequence tensor of shape [batch_size, src_seq_len]
            decoder_input (torch.Tensor): Target sequence tensor of shape [batch_size, tgt_seq_len]
            src_padding_mask (torch.Tensor, optional): Boolean mask for source padding tokens
                                                    (True for pad tokens). Defaults to None.
            tgt_padding_mask (torch.Tensor, optional): Boolean mask for target padding tokens
                                                    (True for pad tokens). Defaults to None.
                                                    
        Returns:
            torch.Tensor: Output logits of shape [batch_size, tgt_seq_len, vocab_size]
        """
        # Encoder
        encoder_input_embeddings = self.token_embeddings(encoder_input) + self.position_embeddings.to(encoder_input.device)
        encoder_output = self.encoder_stack(encoder_input_embeddings, padding_mask=src_padding_mask)
        
        # Decoder
        decoder_input_embeddings = self.token_embeddings(decoder_input) + self.position_embeddings.to(decoder_input.device)
        decoder_output = self.decoder_stack(
            decoder_input_embeddings,
            encoder_output,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        
        # Output projection
        output_logits = self.W_o(decoder_output)
        
        return output_logits
    
    def get_sinusoidal_position_embeddings(self):
        """
        Generate sinusoidal positional embeddings as described in the paper.
        
        Returns:
            torch.Tensor: Position embeddings of shape [seq_len, d_model]
        """
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def translate(self, src_text, tokenizer, max_len=None):
        """
        Translate a source text using the trained model.
        
        Args:
            src_text (str): Source text to translate
            tokenizer (Tokenizer): Tokenizer instance
            max_len (int, optional): Maximum target sequence length. Defaults to None.
                
        Returns:
            str: Translated text
        """
        if max_len is None:
            max_len = self.seq_len
            
        self.eval()
        device = next(self.parameters()).device
        
        # Tokenize source text
        src_ids = torch.tensor(tokenizer.encode(src_text, add_sos=False, add_eos=True), dtype=torch.long).unsqueeze(0).to(device)
        src_padding_mask = (src_ids == tokenizer.special_tokens[tokenizer.pad_token])
        
        # Initialize target with SOS token
        tgt_ids = torch.tensor([[tokenizer.special_tokens[tokenizer.sos_token]]], dtype=torch.long).to(device)
        
        # Generate translation token by token
        for _ in range(max_len - 1):
            # Create padding mask for target
            tgt_padding_mask = (tgt_ids == tokenizer.special_tokens[tokenizer.pad_token])
            
            # Get model predictions
            with torch.no_grad():
                output = self(src_ids, tgt_ids, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            
            # Get the next token prediction
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to target sequence
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.special_tokens[tokenizer.eos_token]:
                break
                
        # Decode the generated token ids
        translation = tokenizer.decode(tgt_ids.squeeze().tolist())
        
        return translation