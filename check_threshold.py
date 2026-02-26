"""
This script calibrates the similarity threshold for the RAG vector database.
It queries the DB with in-domain and out-of-domain queries to find an optimal distance threshold.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "logs"
MODEL_NAME = "BAAI/bge-m3"


def run_experiment():
    print("--- Distance Threshold Calibration Experiment ---")

    # 1. Initialize Database and Model
    client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_PATH)
    )
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)

    # 2. Define Experiment Groups
    # Group A: Relevant queries (Should result in low distance / high similarity)
    related_queries = [
        "sshd failed password",  # SSH login failure
        "disk usage warning",  # Disk space issue
        "connection refused on port 80",  # Network error
    ]

    # Group B: Irrelevant queries (Should result in high distance / low similarity)
    unrelated_queries = [
        "How to cook Kimchi Jjigae?",  # Cooking recipe
        "Who is the president of USA?",  # General knowledge
        "recommend me a good movie",  # Movie recommendation
        "What is the stock price of Apple?",  # Stock market info
    ]

    print(f"\n[Model]: {MODEL_NAME}")
    print("Note: Based on L2 Distance, lower scores indicate higher relevance.\n")

    # 3. Run Experiment
    print("1. Testing Relevant Queries (Target: Low Distance):")
    min_dist = 100
    max_related = 0

    for q in related_queries:
        vec = model.encode([q]).tolist()
        results = collection.query(query_embeddings=vec, n_results=1)
        if results["distances"]:
            dist = results["distances"][0][0]
            # Get a snippet of the matched document
            doc_preview = results["documents"][0][0].replace("\n", " ")[:30]
            print(
                f"  - Query: '{q}' -> Distance: {dist:.4f} (Matched: {doc_preview}...)"
            )
            max_related = max(max_related, dist)
            min_dist = min(min_dist, dist)

    print("\n2. Testing Irrelevant Queries (Target: High Distance):")
    min_unrelated = 100

    for q in unrelated_queries:
        vec = model.encode([q]).tolist()
        results = collection.query(query_embeddings=vec, n_results=1)
        if results["distances"]:
            dist = results["distances"][0][0]
            doc_preview = results["documents"][0][0].replace("\n", " ")[:30]
            print(
                f"  - Query: '{q}' -> Distance: {dist:.4f} (Matched: {doc_preview}...)"
            )
            min_unrelated = min(min_unrelated, dist)

    # 4. Analyze Results
    print("\n" + "=" * 50)
    print("[Analysis Result]")
    print(f"  - Max Distance for Relevant Queries:   {max_related:.4f}")
    print(f"  - Min Distance for Irrelevant Queries: {min_unrelated:.4f}")

    recommended_threshold = (max_related + min_unrelated) / 2

    print("-" * 50)
    if max_related < min_unrelated:
        print(f"Result: Clear separation detected between groups.")
        print(f"Recommended Threshold: {recommended_threshold:.2f}")
        print(
            f"  (Any query with distance > {recommended_threshold:.2f} should be considered 'Unknown')"
        )
    else:
        print(
            "Result: Groups overlap. Data might be insufficient or queries ambiguous."
        )
        print(
            f"  Recommendation: Use a conservative value around {min_unrelated - 0.1:.2f}"
        )
    print("=" * 50)


if __name__ == "__main__":
    run_experiment()
