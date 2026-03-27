import jax
import jax.numpy as jnp
from flax import nnx
import optax
from model import Nanochat

# Constants
VOCAB_SIZE = 1000
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
MAX_LEN = 64
BATCH_SIZE = 8
SEQ_LEN = 32
LEARNING_RATE = 1e-3

def loss_fn(model, batch):
    ids = batch[:, :-1]
    targets = batch[:, 1:]
    
    logits = model(ids)
    
    # Calculate cross-entropy
    # labels: [b, s-1]
    # logits: [b, s-1, vocab_size]
    one_hot = jax.nn.one_hot(targets, VOCAB_SIZE)
    loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    return jnp.mean(loss)

@nnx.jit
def train_step(model, optimizer, batch):
    def wrap_loss_fn(model):
        return loss_fn(model, batch)
        
    grad_fn = nnx.value_and_grad(wrap_loss_fn)
    loss, grads = grad_fn(model)
    
    optimizer.update(grads)
    return loss

def main():
    rngs = nnx.Rngs(0)
    
    # Initialize the model
    model = Nanochat(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        rngs=rngs,
    )
    
    # Initialize optimizer
    # optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE))
    # Wait, nnx.Optimizer simplifies things
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE))
    
    print("Beginning training loop...")
    
    # Training Loop
    for step in range(101):
        # Generate dummy batch [batch, seq_len]
        batch = jax.random.randint(rngs.next(), (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)
        
        loss = train_step(model, optimizer, batch)
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
            
    print("Training complete.")

if __name__ == "__main__":
    main()
