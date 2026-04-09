import torch
from celery import Celery
from transformers import AutoModelForCausalLM, AutoTokenizer

celery = Celery(
    "tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0"
)
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
model.to(torch.device("cpu"))
tokenizer = AutoTokenizer.from_pretrained(model_id)


@celery.task
def generate_answer(user_prompt):

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response
