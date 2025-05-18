import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from .tokenizer import Tokenizer
from config.default import CFG

def load_data(dataset='eng2de', cfg=CFG):
    nrows = 18000
    train = pd.read_csv(cfg.eng2de['train'], lineterminator='\n', nrows=nrows)
    test = pd.read_csv(cfg.eng2de['test'], lineterminator='\n', nrows=nrows)
    val = pd.read_csv(cfg.eng2de['val'], lineterminator='\n', nrows=nrows)
    return train, val, test

class WMT2014EnglishToGermanData(Dataset):
    def __init__(self, df, tokenizer=Tokenizer, max_seq_len=50):
        self.data = df.dropna(subset=['en', 'de']).reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer(max_len=self.max_seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]['en']
        tgt_text = self.data.iloc[idx]['de']
        src_ids = self.tokenizer.encode(src_text, add_sos=False, add_eos=True)
        target_ids = self.tokenizer.encode(tgt_text, add_sos=False, add_eos=True)
        decoder_input_ids = self.tokenizer.encode(tgt_text, add_sos=True, add_eos=False)
        if len(decoder_input_ids) > self.max_seq_len:
            decoder_input_ids = decoder_input_ids[:self.max_seq_len]
        elif len(decoder_input_ids) < self.max_seq_len:
            decoder_input_ids = decoder_input_ids + [self.tokenizer.special_tokens[self.tokenizer.pad_token]] * (self.max_seq_len - len(decoder_input_ids))
        return {
            'source_ids': torch.tensor(src_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

def get_dataloaders(train_df, val_df, test_df, cfg=CFG):
    tokenizer = Tokenizer(max_len=cfg.max_seq_len)
    train_dataset = WMT2014EnglishToGermanData(train_df, tokenizer)
    val_dataset = WMT2014EnglishToGermanData(val_df, tokenizer)
    test_dataset = WMT2014EnglishToGermanData(test_df, tokenizer)

    train_dl = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_dl = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    test_dl = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    
    return train_dl, val_dl, test_dl, tokenizer