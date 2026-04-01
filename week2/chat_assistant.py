import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Загружаем {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )
    print("Напиши 'exit' для выхода.\n")
    print("\n--- ИНТЕРАКТИВНЫЙ ЧАТ ЗАПУЩЕН ---")

    chat_history = [
        {
            "role": "system",
            "content": "You are a highly skilled AI Engineer from 2026. You are sarcastic but very helpful. You speak English.",
        }
    ]

    while True:
        user_input = input("👤 Ты: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        chat_history.append({"role": "user", "content": user_input})

        prompt_string = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        print(f"СЫРОЙ ПРОМПТ: {prompt_string}")
        inputs = tokenizer(prompt_string, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7, do_sample=True
            )
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"🤖 AI Engineer: {response_text}\n")


if __name__ == "__main__":
    main()
