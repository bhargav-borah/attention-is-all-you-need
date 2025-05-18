import tiktoken

class Tokenizer:
    """
    Tokenizer class for encoding and decoding text using OpenAI's tiktoken.
    
    Args:
        encoding_scheme (str, optional): Tiktoken encoding scheme. Defaults to 'cl100k_base'.
        max_len (int, optional): Maximum sequence length. Defaults to None.
    """
    def __init__(self, encoding_scheme='cl100k_base', max_len=None):
        self.tokenizer = tiktoken.get_encoding(encoding_scheme)
        
        # Special tokens
        self.sos_token = '<sos>'  # Start of sequence
        self.eos_token = '<eos>'  # End of sequence
        self.pad_token = '<pad>'  # Padding
        
        # Assign special token IDs after the base vocabulary
        n_vocab = self.vocab_size
        self.special_tokens = {
            self.sos_token: n_vocab,
            self.eos_token: n_vocab + 1,
            self.pad_token: n_vocab + 2
        }
        
        self.max_len = max_len

    def encode(self, text, add_sos=True, add_eos=True):
        """
        Encode text to token IDs.
        
        Args:
            text (str): Input text to encode
            add_sos (bool, optional): Whether to add start-of-sequence token. Defaults to True.
            add_eos (bool, optional): Whether to add end-of-sequence token. Defaults to True.
            
        Returns:
            list: List of token IDs
        """
        # Encode using tiktoken
        token_ids = self.tokenizer.encode(text, allowed_special='all')
        
        # Add special tokens
        if add_sos:
            token_ids = [self.special_tokens[self.sos_token]] + token_ids
        if add_eos:
            token_ids = token_ids + [self.special_tokens[self.eos_token]]
        
        # Handle maximum length constraint
        if self.max_len is not None:
            if len(token_ids) > self.max_len:
                # Truncate if too long
                token_ids = token_ids[:self.max_len]
                # Ensure last token is EOS if specified
                if add_eos:
                    token_ids[-1] = self.special_tokens[self.eos_token]
            elif len(token_ids) < self.max_len:
                # Pad if too short
                token_ids = token_ids + [self.special_tokens[self.pad_token]] * (self.max_len - len(token_ids))
                
        return token_ids

    def decode(self, token_ids):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (list): List of token IDs
            
        Returns:
            str: Decoded text
        """
        # Filter out special tokens
        token_ids = [id_ for id_ in token_ids if id_ not in self.special_tokens.values()]
        
        # Decode using tiktoken
        text = self.tokenizer.decode(token_ids)
        return text

    @property
    def vocab_size(self):
        """
        Get the base vocabulary size (without special tokens).
        
        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.n_vocab