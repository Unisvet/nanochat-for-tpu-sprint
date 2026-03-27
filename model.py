import jax
import jax.numpy as jnp
from flax import nnx
import math

class CausalSelfAttention(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, decode: bool = False, rngs: nnx.Rngs = None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query = nnx.Linear(d_model, d_model, rngs=rngs)
        self.key = nnx.Linear(d_model, d_model, rngs=rngs)
        self.value = nnx.Linear(d_model, d_model, rngs=rngs)
        self.out = nnx.Linear(d_model, d_model, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None):
        b, s, d = x.shape
        
        q = self.query(x).reshape(b, s, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(b, s, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(b, s, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # [b, h, s, s]
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
            
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        out = jnp.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, d)
        return self.out(out)

class MLP(nnx.Module):
    def __init__(self, d_model: int, dim_feedforward: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = nnx.Linear(dim_feedforward, d_model, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray):
        x = self.linear1(x)
        x = jax.nn.gelu(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, rngs: nnx.Rngs):
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.attn = CausalSelfAttention(d_model, n_heads, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.mlp = MLP(d_model, dim_feedforward, rngs=rngs)
        
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray = None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x

class Nanochat(nnx.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int, rngs: nnx.Rngs):
        self.token_embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.pos_embed = nnx.Variable(jnp.zeros((max_len, d_model)), name="pos_embed", type="Param") # Simple learnable pos embedding
        
        self.blocks = nnx.List([
            TransformerBlock(d_model, n_heads, d_model * 4, rngs=rngs)
            for _ in range(n_layers)
        ])
        self.ln_f = nnx.LayerNorm(d_model, rngs=rngs)
        self.lm_head = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)
        
    def __call__(self, ids: jnp.ndarray):
        b, s = ids.shape
        x = self.token_embed(ids)
        
        # Add positional embedding
        x = x + self.pos_embed.value[:s]
        
        # Causal mask - [1, 1, s, s]
        mask = jnp.tril(jnp.ones((s, s)))
        mask = mask.reshape(1, 1, s, s)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
