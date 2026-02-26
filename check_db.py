"""
This utility script inspects the status and contents of the ChromaDB 'logs' collection.
It outputs the total document count and a sample of the most recently added entries.
"""

import chromadb
from chromadb.config import Settings

# Configuration: This path MUST match the one used in the main application
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "logs"


def inspect_database():
    print("--- 🔍 Inspecting ChromaDB Status ---")

    # 1. Initialize the ChromaDB client with persistence settings
    # Ensure 'chroma_db_impl' matches your environment (e.g., duckdb+parquet)
    client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_PATH)
    )

    # 2. Retrieve the collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error: Could not find collection '{COLLECTION_NAME}'.")
        print(f"Details: {e}")
        return

    # 3. Get the total count of documents
    count = collection.count()
    print(f"📊 Total Documents in DB: {count}")

    # 4. Peek at the most recently added data (Limit: 3)
    # Note: 'peek' returns a dictionary with 'ids', 'embeddings', 'documents', 'metadatas'
    if count > 0:
        print("\n--- 📝 Latest 3 Entries ---")
        results = collection.peek(limit=3)

        # Loop through the results to display them cleanly
        for i in range(len(results["ids"])):
            print(f"\n[Item {i + 1}]")
            print(f"ID      : {results['ids'][i]}")
            print(
                f"Content : {results['documents'][i][:100]}..."
            )  # Show only first 100 chars
            print(f"Metadata: {results['metadatas'][i]}")
    else:
        print("\n⚠️ The database is empty.")


if __name__ == "__main__":
    inspect_database()
