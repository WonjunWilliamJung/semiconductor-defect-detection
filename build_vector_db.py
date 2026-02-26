"""
This script builds the ChromaDB vector database using a dual-strategy approach:
It embeds cleaned log fingerprints for accurate semantic search, but stores raw logs for context display.
"""

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import sys
import re

# Silence ChromaDB Logging
logging.getLogger("chromadb").setLevel(logging.ERROR)

INPUT_FILE = "data/real_dataset_rag.jsonl"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "logs"


def get_fingerprint(text):
    """
    Cleaner for Search Strategy: Removes timestamps & PIDs to focus on semantic pattern.
    """
    text = text.lower().strip()
    # Remove Date/Time: "Jun 14 15:16:01 combo "
    text = re.sub(r"^[a-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+", "", text)
    # Remove PIDs: "[12345]"
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def build_dual_strategy_db():
    print("--- Building Vector DB (Dual-Strategy: Clean Search / Raw Display) ---")

    # 1. Load Data
    display_documents = []  # Raw text for storage
    search_texts = []  # Clean text for embedding
    ids = []

    print(f"Reading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                    # Raw content from file
                    raw_log = entry["messages"][0]["content"]
                    solution = entry["messages"][1]["content"]

                    # A) Search Strategy: Clean Fingerprint
                    clean_pattern = get_fingerprint(raw_log)
                    search_texts.append(clean_pattern)

                    # B) Display Strategy: Full Context
                    doc_text = f"[Log]\n{raw_log}\n\n[Solution]\n{solution}"
                    display_documents.append(doc_text)

                    ids.append(str(i))

                except Exception as e:
                    print(f"Skipping line {i}: {e}")

        print(f"Loaded {len(display_documents)} verified entries.")
        if not display_documents:
            print("No documents found. Exiting.")
            return

    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 2. Compute Embeddings (On ClEAN Text)
    print("Loading Embedding Model (BAAI/bge-m3)...")
    try:
        model = SentenceTransformer("BAAI/bge-m3")
        # CRITICAL: Embed the CLEAN text, not the raw text
        embeddings = model.encode(search_texts)
        embeddings_list = embeddings.tolist()
    except Exception as e:
        print(f"Error loading model or encoding: {e}")
        return

    # 3. Store in ChromaDB (Store RAW Documents)
    print("Initializing ChromaDB...")
    try:
        client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_PATH)
        )

        # Reset Collection
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass

        collection = client.create_collection(name=COLLECTION_NAME)

        print("Inserting documents...")
        collection.add(
            documents=display_documents,  # Store Raw
            embeddings=embeddings_list,  # Search by Clean
            ids=ids,
        )

        client.persist()
        print(
            f"✅ Dual-Strategy Knowledge Base built! ({len(display_documents)} entries)"
        )

    except Exception as e:
        print(f"Error with ChromaDB: {e}")


if __name__ == "__main__":
    build_dual_strategy_db()
