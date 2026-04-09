import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_json_from_text(text: str) -> str:
    """
    Ищет JSON внутри строки, убирает ```json ``` и лишние переносы.
    Возвращает чистую JSON-строку.
    """
    # Убираем markdown обрамление
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    # Находим первый { ... } блок
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return text


def load_judge_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return (model, tokenizer)


def safe_json_loads(response: str) -> dict:
    """Пытаемся безопасно распарсить JSON из строки. Если не выходит — возвращаем fallback."""
    clean_response = extract_json_from_text(response)
    try:
        return json.loads(clean_response)
    except json.JSONDecodeError:
        print("❌ Invalid JSON! Cannot parse the model output.")
        print("Raw response:", repr(response))
        return {"score": None, "reason": "Model output is not valid JSON"}


def evaluate_faithfulness(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    context: str,
    answer: str,
) -> dict:
    system_prompt = "You are an impartial judge. Evaluate if the Answer contains any facts that are NOT present in the Context.  Output ONLY a JSON with two keys: 'score' (1 if faithful, 0 if hallucinated) and 'reason' (brief explanation)."
    user_prompt = f"""
Question: {question}

Context: {context}

Answer: {answer}
"""
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=100, temperature=0.1, do_sample=True
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print("DEBUG: faithfulness response =", repr(response))
    return safe_json_loads(response)


def evaluate_answer_relevance(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str, answer: str
) -> dict:
    system_prompt = "You are an impartial judge. Evaluate if the Answer directly addresses the Question. Output ONLY a JSON with 'score' (1 if relevant, 0 if irrelevant) and 'reason'."
    user_prompt = f"""
Question: {question}

Answer: {answer}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=100, temperature=0.1, do_sample=True
    )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print("DEBUG: relevance response =", repr(response))
    return safe_json_loads(response)


def main():
    model, tokenizer = load_judge_model()

    # Эмуляция работы нашего RAG (Тестовый кейс 1: Идеальный ответ)
    test_case_1 = {
        "question": "What is the WiFi password for the London office?",
        "context": "The secret WiFi password in the London office is 'Baguette2026'.",
        "answer": "The WiFi password is 'Baguette2026'.",
    }

    # Эмуляция работы нашего RAG (Тестовый кейс 2: Галлюцинация)
    test_case_2 = {
        "question": "Who is the CEO?",
        "context": "The company was founded in 2020 by John Doe.",
        "answer": "The CEO is John Doe and his favorite coffee is espresso.",  # Галлюцинация про кофе!
    }

    print("\n--- ОЦЕНКА КЕЙСА 1 (Идеальный) ---")
    f_score_1 = evaluate_faithfulness(model, tokenizer, **test_case_1)
    r_score_1 = evaluate_answer_relevance(
        model, tokenizer, test_case_1["question"], test_case_1["answer"]
    )
    print(f"Faithfulness: {f_score_1['score']} | Reason: {f_score_1['reason']}")
    print(f"Relevance: {r_score_1['score']} | Reason: {r_score_1['reason']}")

    print("\n--- ОЦЕНКА КЕЙСА 2 (Галлюцинация) ---")
    f_score_2 = evaluate_faithfulness(model, tokenizer, **test_case_2)
    print(f"Faithfulness: {f_score_2['score']} | Reason: {f_score_2['reason']}")
    r_score_2 = evaluate_answer_relevance(
        model, tokenizer, test_case_2["question"], test_case_2["answer"]
    )
    print(f"Relevance: {r_score_2['score']} | Reason: {r_score_2['reason']}")
    test_text = "Here is the result: {'score': 1, 'reason': 'ok'} and here is trash: {'score': 0}"
    print(f"Testing new function: {extract_json_from_text(test_text)}")


if __name__ == "__main__":
    main()
