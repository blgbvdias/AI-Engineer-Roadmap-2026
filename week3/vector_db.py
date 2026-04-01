import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


def main():
    print("1. Инициализация SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    EMBEDDING_SIZE = model.get_sentence_embedding_dimension()

    print("2. Инициализация Qdrant (in-memory)...")

    client = QdrantClient(":memory:")

    client.create_collection(
        collection_name="knowledge_base",
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )
    print("\n3. Подготовка Базы Знаний...")

    documents = [
        "The Eiffel Tower is located in Paris, France.",
        "Python was created by Guido van Rossum and released in 1991.",
        "The speed of light is approximately 299,792 kilometers per second.",
        "AI Engineers use Qdrant for fast vector similarity search.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]

    print("Превращаем текст в вектора...")
    doc_embeddings = model.encode(documents)

    print("Загружаем данные в Qdrant...")
    points = [
        PointStruct(id=i, vector=embedding.tolist(), payload={"text": doc})
        for i, (doc, embedding) in enumerate(zip(documents, doc_embeddings))
    ]

    client.upsert(collection_name="knowledge_base", points=points)
    print("База знаний успешно загружена!\n")

    print("--- SEARCH TESTING (RETRIEVAL) ---")
    query = "The distance between Earth and Sun?"
    print(f"USER'S QUESTION: '{query}'")

    query_vector = model.encode(query).tolist()

    search_results = client.query_points(
        collection_name="knowledge_base", query=query_vector, limit=1
    )

    print("\n[ FOUND DOCUMENT ]:")
    best_match = search_results.points[0]
    print(f"Текст: {best_match.payload['text']}")
    print(f"Сходство (Score): {best_match.score:.4f}")


if __name__ == "__main__":
    main()
