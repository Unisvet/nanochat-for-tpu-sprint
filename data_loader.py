import os
import requests
import jax.numpy as jnp
import numpy as np

class TinyShakespeareLoader:
    def __init__(self, batch_size=32, seq_len=64, split='train'):
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Download TinyShakespeare if not present
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_path = "tinyshakespeare.txt"
        
        if not os.path.exists(data_path):
            print(f"Downloading TinyShakespeare from {data_url}...")
            r = requests.get(data_url)
            with open(data_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        print(f"Dataset length: {len(self.text)} characters")
        
        # Basic character-level tokenizer
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
        print(f"Vocab size: {self.vocab_size}")
        
        # Encode the whole dataset
        self.data = np.array([self.stoi[c] for c in self.text], dtype=np.int32)
        
        # Split into train/val
        n = int(0.9 * len(self.data))
        if split == 'train':
            self.data = self.data[:n]
        else:
            self.data = self.data[n:]
            
    def get_batch(self, key):
        # Sample random indices for the batch
        # Note: keys in JAX are handled externally in the training loop
        # We can use numpy for basic batching if we want to keep it simple
        ix = np.random.randint(0, len(self.data) - self.seq_len, (self.batch_size,))
        x = np.stack([self.data[i:i+self.seq_len] for i in ix])
        y = np.stack([self.data[i+1:i+self.seq_len+1] for i in ix])
        
        return jnp.array(x), jnp.array(y)

if __name__ == "__main__":
    # Test loader
    loader = TinyShakespeareLoader(batch_size=4, seq_len=8)
    x, y = loader.get_batch(None)
    print("X sample:", x[0])
    print("Y sample:", y[0])
    print("Decoded X:", "".join([loader.itos[int(i)] for i in x[0]]))
    print("Decoded Y:", "".join([loader.itos[int(i)] for i in y[0]]))
