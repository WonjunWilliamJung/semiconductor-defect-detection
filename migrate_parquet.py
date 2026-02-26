"""
This script migrates legacy Parquet-based ChromaDB embeddings to the new SQLite format.
It reads existing data from 'chroma-embeddings.parquet' and inserts it into the persistent 'logs' collection.
"""

import pandas as pd
import chromadb
import os
import numpy as np

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
PARQUET_PATH = os.path.join(CHROMA_DB_PATH, "chroma-embeddings.parquet")


def migrate():
    print(f"📂 Reading Parquet: {PARQUET_PATH}")
    if not os.path.exists(PARQUET_PATH):
        print("❌ Parquet file not found!")
        return

    # 1. Read Data
    try:
        df = pd.read_parquet(PARQUET_PATH)
        print(f"✅ Loaded {len(df)} rows from parquet.")
    except Exception as e:
        print(f"❌ Error reading parquet: {e}")
        return

    # 2. Extract Columns
    # We use 'embedding_id' as the ID, 'embedding' as vector, 'document' as text
    ids = df["embedding_id"].tolist()
    documents = df["document"].tolist()
    embeddings = df["embedding"].tolist()

    # Metadatas: If column exists and not all None
    metadatas = None
    if "metadata" in df.columns:
        # Check if it has real data
        # In parquet, it might be stored as json string or just None
        # For this app, we expect None mostly, but let's check
        first_meta = df["metadata"].iloc[0]
        if first_meta is not None and first_meta != "null":
            metadatas = df["metadata"].tolist()
        else:
            print("ℹ️ No metadata found (all None/null).")

    # 3. Connect to ChromaDB (SQLite)
    print(f"🔌 Connecting to ChromaDB at {CHROMA_DB_PATH}...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        # Target collection is 'logs' as per app.py
        collection = client.get_or_create_collection(name="logs")
        print(f"✅ Collection 'logs' ready. Current count: {collection.count()}")
    except Exception as e:
        print(f"❌ Error connecting to DB: {e}")
        return

    # 4. Insert Data
    print("🚀 Migrating data...")
    batch_size = 100
    total = len(ids)

    for i in range(0, total, batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_embs = embeddings[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size] if metadatas else None

        try:
            if batch_metas:
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embs,
                    metadatas=batch_metas,
                )
            else:
                collection.add(
                    ids=batch_ids, documents=batch_docs, embeddings=batch_embs
                )
            print(f"   Processed {min(i + batch_size, total)}/{total}")
        except Exception as e:
            print(f"❌ Error inserting batch {i}: {e}")

    print(f"🎉 Migration Complete. Final Count: {collection.count()}")


if __name__ == "__main__":
    migrate()
