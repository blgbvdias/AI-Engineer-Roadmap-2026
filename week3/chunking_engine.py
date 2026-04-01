import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

LONG_ARTICLE = """
Artificial intelligence (AI) is the intelligence of machines or software, as opposed to the intelligence of human beings or animals. AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), and competing at the highest level in strategic games (such as chess and Go).
AI was founded as an academic discipline in 1956. The field went through multiple cycles of optimism, followed by periods of disappointment and loss of funding, known as AI winter. Funding and interest vastly increased after 2012 when deep learning surpassed all previous AI techniques, and after 2017 with the transformer architecture. This led to the AI boom of the early 2020s, with companies, universities, and laboratories overwhelmingly based in the United States pioneering significant advances in artificial intelligence.
The growing use of artificial intelligence in the 21st century is influencing a societal and economic shift towards increased automation, data-driven decision-making, and the integration of AI systems into various economic sectors and areas of life, impacting job markets, healthcare, government, industry, and education. This raises questions about the long-term effects, ethical implications, and risks of AI, prompting discussions about regulatory policies to ensure the safety and benefits of the technology.
"""


def chunk_document(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunk_list = splitter.split_text(text)
    return chunk_list


def init_vector_db(embedding_size: int) -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="wikipedia",
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )
    return client


def ingest_chunks_to_db(
    client: QdrantClient, chunks: list[str], encoder: SentenceTransformer
):
    embedding_matrix = encoder.encode(chunks)
    points = [
        PointStruct(id=i, vector=embedding.tolist(), payload={"text": chunk})
        for i, (chunk, embedding) in enumerate(zip(chunks, embedding_matrix))
    ]
    client.upsert(collection_name="wikipedia", points=points)


def search_database(
    client: QdrantClient, encoder: SentenceTransformer, query: str, top_k: int = 2
):
    vector_query = encoder.encode(query).tolist()
    results = client.query_points(
        collection_name="wikipedia", query=vector_query, limit=top_k
    )
    scores = [point.score for point in results.points]
    contexts = [point.payload.get("text", "") for point in results.points]
    printable = [
        f"{i}. Context {context}, score {score:.4f}"
        for i, (context, score) in enumerate(zip(contexts, scores))
    ]
    for line in printable:
        print(line)


def main():
    print("1. Нарезаем текст...")
    chunks_list = chunk_document(text=LONG_ARTICLE, chunk_size=300, chunk_overlap=50)
    print(len(chunks_list), chunks_list[0])

    print("\n2. Запускаем модели и БД...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    client = init_vector_db(embedding_size=384)

    print("\n3. Загружаем чанки в Qdrant...")
    ingest_chunks_to_db(client=client, chunks=chunks_list, encoder=encoder)

    print("\n4. Ищем ответ...")
    query = "When was AI founded as an academic discipline?"
    search_database(client=client, encoder=encoder, query=query)


if __name__ == "__main__":
    main()
