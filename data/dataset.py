import torch
from torch.utils.data import Dataset
import pandas as pd

class WMT2014EnglishToGermanData(Dataset):
    """
    Dataset class for WMT 2014 English-to-German translation task.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 'en' and 'de' columns
        tokenizer (Tokenizer): Tokenizer instance
        max_seq_len (int, optional): Maximum sequence length. Defaults to 50.
    """
    def __init__(self, df, tokenizer, max_seq_len=50):
        # Drop rows with NaN in either English or German columns
        self.data = df.dropna(subset=['en', 'de']).reset_index(drop=True)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def __len__(self):
        """
        Get the dataset size.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing 'source_ids', 'decoder_input_ids', and 'target_ids'
        """
        # Get source and target texts
        src_text = self.data.iloc[idx]['en']
        tgt_text = self.data.iloc[idx]['de']
        
        # Encode source sequence (no SOS token, but include EOS token)
        src_ids = self.tokenizer.encode(src_text, add_sos=False, add_eos=True)
        
        # Encode target sequence for loss calculation (no SOS token, but include EOS token)
        target_ids = self.tokenizer.encode(tgt_text, add_sos=False, add_eos=True)
        
        # Encode decoder input (include SOS token, but no EOS token)
        decoder_input_ids = self.tokenizer.encode(tgt_text, add_sos=True, add_eos=False)
        
        # Handle maximum length for decoder input
        if len(decoder_input_ids) > self.max_seq_len:
            decoder_input_ids = decoder_input_ids[:self.max_seq_len]
        elif len(decoder_input_ids) < self.max_seq_len:
            decoder_input_ids = decoder_input_ids + [self.tokenizer.special_tokens[self.tokenizer.pad_token]] * (self.max_seq_len - len(decoder_input_ids))
        
        return {
            'source_ids': torch.tensor(src_ids, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }


def load_wmt2014_data(train_path, val_path, test_path, nrows=None):
    """
    Load WMT 2014 English-to-German data from CSV files.
    
    Args:
        train_path (str): Path to the training data CSV
        val_path (str): Path to the validation data CSV
        test_path (str): Path to the test data CSV
        nrows (int, optional): Number of rows to load. Defaults to None (load all).
        
    Returns:
        tuple: (train_df, val_df, test_df) DataFrames
    """
    train = pd.read_csv(train_path, lineterminator='\n', nrows=nrows)
    val = pd.read_csv(val_path, lineterminator='\n', nrows=nrows)
    test = pd.read_csv(test_path, lineterminator='\n', nrows=nrows)
    
    return train, val, test