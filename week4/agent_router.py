from typing import TypedDict

import torch
from langgraph.graph import END, StateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer


class AgentState(StateGraph):
    question: str
    decision: str
    response: str


MODEL = None
TOKENIZER = None


def load_llm():
    global MODEL, TOKENIZER
    print("Загрузка модели агента...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    MODEL = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map="auto"
    )


def llm_generate(prompt: str, temperature: float = 0.1) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = TOKENIZER(text, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs, max_new_tokens=50, temperature=temperature, do_sample=True
        )
    return TOKENIZER.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def node_router(state: AgentState) -> dict:
    question = state["question"]
    if any(word in question for word in ["password", "policy", "secret"]):
        return {"decision": "rag_search"}
    return {"decision": "direct_answer"}


def node_rag_search(state: AgentState) -> dict:
    question = state["question"]
    if "password" in question.lower():
        context = "The secret WiFi password is 'Baguette2026'."
    else:
        context = "I don't have this document in my database."
    prompt = f"Answer using this context: {context}. Question: {question}"
    result = llm_generate(prompt=prompt).strip()
    return {"response": result}


def node_direct_answer(state: AgentState) -> dict:
    question = state["question"]
    prompt = f"Answer this general question: {question}"
    result = llm_generate(prompt=prompt).strip()
    return {"response": result}


def route_decision(state: AgentState) -> str:
    if state.get("decision") == "rag_search":
        return "rag_node"
    return "direct_node"


def main():
    load_llm()

    print("\nСборка графа (State Machine)...")
    # Инициализируем граф с нашей структурой памяти (AgentState)
    workflow = StateGraph(AgentState)

    # Добавляем узлы
    workflow.add_node("router_node", node_router)
    workflow.add_node("rag_node", node_rag_search)
    workflow.add_node("direct_node", node_direct_answer)

    # Строим ребра (Связи)
    # 1. Стартуем всегда с роутера
    workflow.set_entry_point("router_node")

    # 2. Условный переход (Conditional Edge). Куда идти после роутера?
    workflow.add_conditional_edges(
        "router_node",  # Откуда
        route_decision,  # Функция-условие
        {
            "rag_node": "rag_node",  # Если функция вернула "rag_node", идем в rag_node
            "direct_node": "direct_node",  # Если вернула "direct_node", идем в direct_node
        },
    )

    # 3. После RAG или Direct ответа - заканчиваем работу (идем в END)
    workflow.add_edge("rag_node", END)
    workflow.add_edge("direct_node", END)

    # Компилируем граф в исполняемое приложение
    app = workflow.compile()

    print("\n--- ТЕСТ 1: Обычный вопрос ---")
    state_1 = {"question": "What is the capital of France?"}
    # app.invoke прогоняет State через все узлы графа до END
    result_1 = app.invoke(state_1)
    print(f"Решение агента: {result_1.get('decision')}")
    print(f"Ответ: {result_1.get('response')}")

    print("\n--- ТЕСТ 2: Корпоративный секрет ---")
    state_2 = {"question": "Tell me the secret WiFi password."}
    result_2 = app.invoke(state_2)
    print(f"Решение агента: {result_2.get('decision')}")
    print(f"Ответ: {result_2.get('response')}")


if __name__ == "__main__":
    main()
