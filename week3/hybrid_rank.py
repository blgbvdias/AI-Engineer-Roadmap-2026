import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

MOCK_DOCUMENTS = [
    "Apple is a popular fruit that is red or green and grows on trees.",
    "Apple Inc. is a technology company headquartered in Cupertino, California.",
    "Python is a large, heavy-bodied snake found in Africa and Asia.",
    "Python is a high-level, general-purpose programming language.",
    "The company revenue grew by 20% in Q3 due to strong iPhone sales.",
]


def setup_database(docs: list[str]) -> tuple[QdrantClient, SentenceTransformer]:
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embed_list = encoder.encode(docs).tolist()
    embedding_size = encoder.get_sentence_embedding_dimension()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=i, vector=embedding, payload={"text": doc})
        for i, (doc, embedding) in enumerate(zip(docs, embed_list))
    ]
    client.upsert(collection_name="docs", points=points)
    return client, encoder


def retrieve_candidates(
    client: QdrantClient, encoder: SentenceTransformer, query: str, top_k: int = 5
) -> list[str]:
    query_vector = encoder.encode(query).tolist()
    results = client.query_points(
        collection_name="docs", query=query_vector, limit=top_k
    )
    docs = []
    for i, point in enumerate(results.points):
        text = point.payload.get("text", "")
        score = point.score
        print(f"{i}. {text} | SCORE: {score:.4f}")
        docs.append(text)
    return docs


def rerank_docs(
    query: str,
    docs: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[tuple[str, float]]:
    encoder = CrossEncoder(model_name)
    pairs = [(query, doc) for doc in docs]
    scores = encoder.predict(pairs).tolist()
    paired = list(zip(docs, scores))
    sorted_docs = sorted(paired, key=lambda x: x[1], reverse=True)
    return sorted_docs


def main():
    client, encoder = setup_database(MOCK_DOCUMENTS)
    query = "Where is the headquarters of the tech company that makes iPhones?"
    docs = retrieve_candidates(client=client, encoder=encoder, query=query)
    reranked = rerank_docs(query=query, docs=docs)
    print(query)
    for i, (doc, score) in enumerate(reranked):
        print(f"{i}. {doc} | SCORE: {score:.4f}")


if __name__ == "__main__":
    main()
