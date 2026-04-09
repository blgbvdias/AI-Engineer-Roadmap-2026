from fastapi import FastAPI
from pydantic import BaseModel

from worker import generate_answer


class Request(BaseModel):
    prompt: str

app = FastAPI()

@app.post('/generate')
async def generate(request: Request):
    task = generate_answer.delay(request.prompt)
    return {'task_id': task.id}

@app.get('/result/{task_id}')
async def result(task_id: str):
    result = generate_answer.AsyncResult(task_id)
    if result.ready():
        return result
    return {'status': 'processing'}        return result
    return {'status': 'processing'}
