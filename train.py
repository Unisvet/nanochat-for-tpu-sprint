import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from model import Nanochat
from data_loader import TinyShakespeareLoader

# Constants
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
MAX_LEN = 128
BATCH_SIZE = 16
SEQ_LEN = 64
LEARNING_RATE = 5e-4
STEPS = 500

def loss_fn(model, x, y, vocab_size):
    # x: [b, s], y: [b, s]
    logits = model(x) # [b, s, v]
    
    # labels: [b, s]
    # logits: [b, s, vocab_size]
    one_hot = jax.nn.one_hot(y, vocab_size)
    loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    return jnp.mean(loss)

@nnx.jit(static_argnames="vocab_size")
def train_step(model, optimizer, x, y, vocab_size):
    def loss_handler(model, x, y):
        return loss_fn(model, x, y, vocab_size)

    grad_fn = nnx.value_and_grad(loss_handler)
    loss, grads = grad_fn(model, x, y)
    
    optimizer.update(grads)
    return loss

def main():
    rngs = nnx.Rngs(0)
    
    # Initialize DataLoader
    loader = TinyShakespeareLoader(batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    vocab_size = loader.vocab_size
    
    # Initialize the model
    model = Nanochat(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        rngs=rngs,
    )
    
    # Initialize optimizer (AdamW)
    optimizer = nnx.ModelAndOptimizer(model, optax.adamw(LEARNING_RATE))
    
    # Initial loss sanity check
    # Random model should have loss ~ log(vocab_size)
    expected_initial_loss = np.log(vocab_size)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Expected initial loss: {expected_initial_loss:.4f}")
    
    print("Beginning training loop on CPU...")
    
    losses = []
    for step in range(STEPS + 1):
        x, y = loader.get_batch(None)
        
        loss = train_step(model, optimizer, x, y, vocab_size)
        losses.append(float(loss))
        
        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            
    avg_early = np.mean(losses[:20])
    avg_late = np.mean(losses[-20:])
    print(f"\nFinal Statistics:")
    print(f"Initial Average Loss (first 20): {avg_early:.4f}")
    print(f"Final Average Loss (last 20): {avg_late:.4f}")
    
    if avg_late < avg_early:
        print("SUCCESS: Loss is decreasing! The model is learning.")
    else:
        print("WARNING: Loss did not decrease significantly. Please check hyperparameters.")

if __name__ == "__main__":
    main()
