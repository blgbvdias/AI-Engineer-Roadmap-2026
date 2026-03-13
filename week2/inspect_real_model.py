import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "Qwen/Qwen2.5-0.5B"

    print(f"Downloading Tokenizator {model_id}........")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"2. Загружаем веса модели в память (это может занять минуту)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map='auto'
    )

    print("\n--- АНАЛИЗ АРХИТЕКТУРЫ ---")


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Всего параметров: {total_params:,}")

    memory_mb = (total_params*2)/(1024**2)
    print(f"Оценка потребления VRAM (чистые веса): {memory_mb:.2f} MB")

    print("\n--- ГЕНЕРАЦИЯ (INFERENCE) ---")
    prompt = "The most important skill for an AI Engineer in 2026 is"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"Промпт: '{prompt}'")
    print("Генерирую ответ (используя встроенный KV-Cache)...")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)


    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n[ ОТВЕТ МОДЕЛИ ]:")
    print(result)

if __name__ ==  '__main__':
    main()
