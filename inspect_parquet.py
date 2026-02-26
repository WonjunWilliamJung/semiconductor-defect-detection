"""
This utility script inspects the contents and schema of legacy ChromaDB Parquet files.
It is used for debugging and verifying the structure of embeddings and collections.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

emb_path = os.path.join(CHROMA_DB_PATH, "chroma-embeddings.parquet")
col_path = os.path.join(CHROMA_DB_PATH, "chroma-collections.parquet")

print(f"Reading: {emb_path}")
try:
    df_emb = pd.read_parquet(emb_path)
    print("Embeddings Columns:", df_emb.columns.tolist())
    print("First Row:", df_emb.iloc[0].to_dict())
    print(f"Total Embeddings: {len(df_emb)}")
except Exception as e:
    print(f"Error reading embeddings: {e}")

print("-" * 20)

print(f"Reading: {col_path}")
try:
    df_col = pd.read_parquet(col_path)
    print("Collections Columns:", df_col.columns.tolist())
    print("First Row:", df_col.iloc[0].to_dict())
except Exception as e:
    print(f"Error reading collections: {e}")
