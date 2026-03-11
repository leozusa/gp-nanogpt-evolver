import os
import torch
import numpy as np

# Constants for injection
MAX_SEQ_LEN = 2048
TIME_BUDGET = 300

def make_dataloader(tokenizer, B, T, split):
    data_dir = os.path.join('nanoGPT', 'data', 'shakespeare_char')
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Simple random sampling dataloader
    def get_batch():
        while True:
            ix = torch.randint(len(data) - T, (B,))
            x = torch.stack([torch.from_numpy((data[i:i+T]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+T+1]).astype(np.int64)) for i in ix])
            # Use mps if available
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            x, y = x.to(device), y.to(device)
            yield x, y, 0 # 0 as dummy epoch
            
    return get_batch()

class Tokenizer:
    def __init__(self):
        import pickle
        meta_path = os.path.join('nanoGPT', 'data', 'shakespeare_char', 'meta.pkl')
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        self.stoi = meta['stoi']
        self.itos = meta['itos']

    @staticmethod
    def from_directory():
        return Tokenizer()

    def get_vocab_size(self):
        return self.vocab_size

def evaluate_bpb(model, tokenizer, B):
    # Simplified evaluation for smoke test
    model.eval()
    T = model.config.sequence_len
    loader = make_dataloader(tokenizer, B, T, 'val')
    x, y, _ = next(loader)
    with torch.no_grad():
        # autoresearch train.py returns only loss if targets is not None
        loss = model(x, y)
    return loss.item() / np.log(2) # bpw (bits per word/char)
