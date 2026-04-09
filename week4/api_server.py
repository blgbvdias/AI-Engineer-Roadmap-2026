import asyncio
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ml_models = {}
model_id = "Qwen/Qwen2.5-0.5B-Instruct"


class Request(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.1


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["tokenizer"] = AutoTokenizer.from_pretrained(model_id)
    ml_models["model"] = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="auto"
    )
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(data: Request):

    result = await asyncio.to_thread(
        answer_generation,
        data.prompt,
        max_new_tokens=data.max_new_tokens,
        temperature=data.temperature,
    )
    return {"result": result}


def answer_generation(
    user_prompt: str, max_new_tokens: int, temperature: float, system_prompt: str = ""
) -> str:
    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response
