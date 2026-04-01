import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    messages = [
        {"role": "user", "content": "Сколько тебе лет?"},
    ]
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "../models/qwen-pirate-lora"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("1. Загрузка базовой модели (Base Model)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )

    print("2. Подключение LoRA-адаптера...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    print("\n--- ЧАТ С ПИРАТОМ-ИНЖЕНЕРОМ ---")

    prompt_string = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_string, return_tensors="pt").to(model.device)
    print("Генерирую ответ...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=100, temperature=0.7, do_sample=True
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"\n🏴‍☠️ ИИ-Пират: {response_text}\n")


if __name__ == "__main__":
    main()
