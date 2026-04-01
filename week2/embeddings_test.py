import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def main():
    print("Загрузка SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "The quick brown fox jumps over the lazy dog.",  # Индекс 0
        "A fast dark-colored creature leaps across a tired pup.",  # Индекс 1
        "Artificial Intelligence engineers earn high salaries.",  # Индекс 2
    ]
    print("\nГенерация эмбеддингов...")
    embeddings = model.encode(sentences, convert_to_tensor=True)
    print(f"Размерность полученной матрицы: {embeddings.shape}")

    vec_0 = embeddings[0]
    vec_1 = embeddings[1]
    vec_2 = embeddings[2]

    print("\n--- ВЫЧИСЛЕНИЕ КОСИНУСНОГО СХОДСТВА ---")

    sim_0_1 = F.cosine_similarity(vec_0, vec_1, dim=0)
    sim_0_2 = F.cosine_similarity(vec_0, vec_2, dim=0)
    print(f"Сходство (Лиса vs Существо): {sim_0_1.item():.4f}")
    print(f"Сходство (Лиса vs AI-инженер): {sim_0_2.item():.4f}")


if __name__ == "__main__":
    main()
