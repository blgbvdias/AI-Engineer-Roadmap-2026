import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from core.model import MiniLlama


@torch.no_grad()
def generate_text_with_cache(
    model, prompt, tokenizer, max_new_tokens=30, temperature=0.8
):
    model.eval()
    # 1. Токенизируем промпт. Например, "In 2026, AI Engineers" (5 токенов)
    # Размер input_ids: [Batch=1, Time=5]
    input_ids = torch.tensor(data=tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0)
    # Эта переменная будет хранить наш сгенерированный текст для вывода
    generate_sequence = input_ids.clone()
    # Изначально у нас нет никакого кэша (коробка пуста)
    past_key_values = None

    print(f"\n[ ПРОМПТ ]: {prompt}\n[ ГЕНЕРАЦИЯ ]:", end=" ")

    # Цикл генерации
    for _ in range(max_new_tokens):

        # 2. ПРОХОД ЧЕРЕЗ МОДЕЛЬ (С КЭШЕМ)
        # Обрати внимание: теперь model.forward возвращает ДВА значения!
        # logits: вероятности следующего слова.
        # past_key_values: обновленная коробка с кэшем от всех слоев.
        logits, past_key_values = model(input_ids, past_key_values=past_key_values)
        # 3. Берем логиты ТОЛЬКО для последнего слова.
        # Почему [:, -1, :]?
        # На самом первом шаге (Prefill) мы прогнали 5 слов, и логитов 5 штук. Нам нужен последний.
        # На всех последующих шагах мы прогоняем всего 1 слово, и логит всего 1. Срез [:, -1, :] работает в обоих случаях!
        next_token_logits = logits[:, -1, :] / temperature
        # 4. Сэмплирование (выбор следующего слова)
        probs =  F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        # Вывод в консоль
        word = tokenizer.decode(next_token[0].tolist())
        print(word, end="", flush=True)

        # 5. ПРЕДМЕТ ГОРДОСТИ ИНЖЕНЕРА (Обновление входа)
        # На следующем шаге цикла нам НЕ НУЖНО подавать весь текст заново!
        # Мы скармливаем модели ТОЛЬКО ОДИН новый сгенерированный токен.
        # Размер input_ids становится [Batch=1, Time=1].
        input_ids = next_token
        # (Для красоты мы всё равно приклеиваем токен к общей строке, чтобы потом вернуть весь текст,
        # но в саму модель эта строка больше не идет!)
        generate_sequence = torch.cat((generate_sequence, next_token), dim=1)
    print("\n\n--- Инференс с KV-Cache завершен ---")
    return generate_sequence
def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    VOCAB_SIZE = tokenizer.vocab_size

    model = MiniLlama(
        vocab_size=VOCAB_SIZE, embed_dim=128, num_heads=4, hidden_dim=512, num_layers=2
    )

    weight_path = "./mini_llama_weights.pth"
    model.load_state_dict(torch.load(weight_path, weights_only=True))

    prompt_text = "In 2026, AI Engineers"
    generate_text_with_cache(model=model, prompt=prompt_text, tokenizer=tokenizer)


if __name__ == "__main__":
    main()
