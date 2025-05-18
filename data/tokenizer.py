import tiktoken
from config.default import CFG

class Tokenizer:
    def __init__(self, encoding_scheme=CFG.encoding_scheme, max_len=None):
        self.tokenizer = tiktoken.get_encoding(encoding_scheme)
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        n_vocab = self.vocab_size
        self.special_tokens = {
            self.sos_token: n_vocab,
            self.eos_token: n_vocab + 1,
            self.pad_token: n_vocab + 2
        }
        self.max_len = max_len

    def encode(self, text, add_sos=True, add_eos=True):
        token_ids = self.tokenizer.encode(text, allowed_special='all')
        if add_sos:
            token_ids = [self.special_tokens[self.sos_token]] + token_ids
        if add_eos:
            token_ids = token_ids + [self.special_tokens[self.eos_token]]

        if self.max_len is not None:
            if len(token_ids) > self.max_len:
                token_ids = token_ids[:self.max_len]
                if add_eos:
                    token_ids[-1] = self.special_tokens[self.eos_token]
            elif len(token_ids) < self.max_len:
                token_ids = token_ids + [self.special_tokens[self.pad_token]] * (self.max_len - len(token_ids))
        return token_ids

    def decode(self, token_ids):
        token_ids = [id_ for id_ in token_ids if id_ not in self.special_tokens.values()]
        text = self.tokenizer.decode(token_ids)
        return text

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab