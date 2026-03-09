import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

from core.model import MiniLlama
from data.dataset import get_batch


def main():
    print("Initializating tokenizer and data......")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    text_data = (
        "In 2026, AI Engineers build models from scratch. They don't just use APIs. "
        * 100
    )
    tokens = tokenizer.encode(text_data)
    data_tensor = torch.tensor(data=tokens, dtype=torch.long)

    VOCAB_SIZE = tokenizer.vocab_size
    B, T = 4, 16

    print("Building Model......")
    model = MiniLlama(
        vocab_size=VOCAB_SIZE, embed_dim=128, num_heads=4, hidden_dim=512, num_layers=2
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting learn in 100 steps......")

    for step in range(100):
        x, y = get_batch(data_tensor=data_tensor, seq_len=T, batch_size=B)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    print("Training finished, saving weights.......")
    torch.save(model.state_dict(), "mini_llama_weights.pth")
    print("File mini_llama_weights.pth successfully created!")


if __name__ == "__main__":
    main()
