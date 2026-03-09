import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from core.model import MiniLlama


@torch.no_grad()
def generate_text(
    model, prompt, tokenizer, max_new_tokens=30, temperature=0.8, seq_len=16
):
    model.eval()

    idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    print(f"\n[ ПРОМПТ ]: {prompt}\n[ ГЕНЕРАЦИЯ ]:", end=" ")

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -seq_len:]
        logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        next_token_logits = next_token_logits / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)
        word = tokenizer.decode(next_token[0].tolist())
        print(word, end="", flush=True)

    print("\n\n--- Инференс завершен ---")
    return idx


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    VOCAB_SIZE = tokenizer.vocab_size

    model = MiniLlama(
        vocab_size=VOCAB_SIZE, embed_dim=128, num_heads=4, hidden_dim=512, num_layers=2
    )

    weight_path = "./mini_llama_weights.pth"
    model.load_state_dict(torch.load(weight_path, weights_only=True))

    prompt_text = "In 2026, AI Engineers"
    generate_text(model=model, prompt=prompt_text, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
