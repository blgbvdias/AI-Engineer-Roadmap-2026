import json

import outlines
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class FaithfulnessEval(BaseModel):
    is_faithful: bool = Field(
        description="True if the answer contains NO hallucinations, False otherwise."
    )
    reason: str = Field(
        description="A short 1-sentence explanation of why it is faithful or not."
    )


def load_structured_model():
    # ЗАГРУЖАЕМ МОДЕЛЬ ПРАВИЛЬНО
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    hf_model = AutoModelForCausalLM.from_pretrained(model_id)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    model = outlines.from_transformers(
        hf_model, hf_tokenizer
    )  # <-- так подтянет модель + токенизатор
    return model


def evaluate_with_structure(
    model, question: str, context: str, answer: str
) -> FaithfulnessEval:
    prompt = f"""
You are a strict evaluator.
Your task is to determine whether the given Answer is fully supported by the provided Context.

Rules:
- Use ONLY the information from the Context.
- Do NOT use prior knowledge.
- If the Answer contains ANY information not present in the Context, it is NOT faithful.
- Even small extra details count as hallucination.

Question: {question}
Context: {context}
Answer: {answer}

Return ONLY JSON matching this schema exactly.
{{"is_faithful": bool, "reason": str}}
"""

    generator = outlines.Generator(model, FaithfulnessEval)
    json_str = generator(prompt, max_new_tokens=256, temperature=0.01)
    try:
        result = FaithfulnessEval.model_validate_json(json_str)
    except Exception:
        try:
            fixed = json_str + "}"
            parsed = json.loads(fixed)
            result = FaithfulnessEval(**parsed)
        except Exception:
            result = FaithfulnessEval(
                is_faithful=False, reason="Model output invalid JSON"
            )
    return result


def main():
    print("Загрузка модели через Outlines...")
    model = load_structured_model()

    # Тестовый кейс 1: Идеальный ответ
    test_1 = {
        "question": "What is the WiFi password?",
        "context": "The WiFi password is 'Baguette2026'.",
        "answer": "The password is 'Baguette2026'.",
    }

    # Тестовый кейс 2: Галлюцинация
    test_2 = {
        "question": "Who is the CEO?",
        "context": "The company was founded by John Doe.",
        "answer": "The CEO is John Doe and he loves espresso.",
    }

    print("\n--- ОЦЕНКА 1 (Идеальная) ---")
    # ЗАДАЧА 4: Вызови evaluate_with_structure для test_1.
    # Распечатай: result.is_faithful и result.reason

    eval_test_1 = evaluate_with_structure(model, **test_1)
    print(eval_test_1)

    print("\n--- ОЦЕНКА 2 (Галлюцинация) ---")
    # ЗАДАЧА 5: Вызови evaluate_with_structure для test_2.
    # Распечатай: result.is_faithful и result.reason
    eval_test_2 = evaluate_with_structure(model, **test_2)
    print(eval_test_2)


if __name__ == "__main__":
    main()
