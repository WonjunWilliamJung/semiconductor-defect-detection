"""
This utility script checks the status and count of collections in the ChromaDB vector database.
"""

import chromadb
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

print(f"Checking DB at: {CHROMA_DB_PATH}")

try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = client.list_collections()

    print(f"Found {len(collections)} collections.")
    for col in collections:
        print(f"- Name: '{col.name}', Count: {col.count()}")

except Exception as e:
    print(f"Error: {e}")
