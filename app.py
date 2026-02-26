"""
This is the main Streamlit application script for the Samsung Log Analysis system.
It provides the UI for log analysis, history tracking, and database admin management.
It integrates with ChromaDB for RAG context retrieval and LLMs for generation.
"""

# --- Streamlit Cloud SQLite Fix ---
import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
import json
import uuid
import datetime
import warnings

# Load environment variables
load_dotenv(".env.local")

# Disable Tokenizers Parallelism (Fix for Model Loading Error)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuration (Absolute Path Setup) ---
# [Key Modification] Calculate absolute path based on current file (app.py) location.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")
PENDING_REVIEW_FILE = os.path.join(BASE_DIR, "pending_review.json")

COLLECTION_NAME = "logs"
DISTANCE_THRESHOLD = 0.95

# --- 3. Backend Logic (RAG & Eval) ---


@st.cache_resource
def load_chroma_collection():
    # [Modified] Using ChromaDB PersistentClient (Compatible with v0.4.x+)
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"✅ DB Loaded. Items: {collection.count()}")

        # --- Auto-Migration Logic for Legacy Parquet Data ---
        if collection.count() == 0:
            parquet_path = os.path.join(CHROMA_DB_PATH, "chroma-embeddings.parquet")
            if os.path.exists(parquet_path):
                print(
                    f"🔄 Found legacy parquet file. Attempting migration from: {parquet_path}"
                )
                try:
                    import pandas as pd

                    df = pd.read_parquet(parquet_path)

                    # Prefer 'embedding_id' (UUID), fallback to 'id' (legacy 0,1,2...)
                    id_col = "embedding_id" if "embedding_id" in df.columns else "id"
                    if id_col not in df.columns:
                        print(
                            f"❌ Critical: Neither 'embedding_id' nor 'id' found in columns."
                        )
                        return

                    ids = df[id_col].astype(str).tolist()
                    documents = df["document"].tolist()
                    embeddings = df["embedding"].tolist()

                    # Handle Metadata
                    metadatas = None
                    if "metadata" in df.columns:
                        # Filter out None/null if mostly empty to avoid errors
                        # or simply pass list if Chroma handles paths
                        if df["metadata"].notna().any():
                            metadatas = (
                                df["metadata"]
                                .apply(lambda x: x if isinstance(x, dict) else None)
                                .tolist()
                            )

                    print(f"🚀 Migrating {len(ids)} items...")
                    # Batch Insert
                    batch_size = 50
                    for i in range(0, len(ids), batch_size):
                        b_ids = ids[i : i + batch_size]
                        b_docs = documents[i : i + batch_size]
                        b_embs = embeddings[i : i + batch_size]
                        b_metas = metadatas[i : i + batch_size] if metadatas else None

                        if b_metas:
                            collection.add(
                                ids=b_ids,
                                documents=b_docs,
                                embeddings=b_embs,
                                metadatas=b_metas,
                            )
                        else:
                            collection.add(
                                ids=b_ids, documents=b_docs, embeddings=b_embs
                            )

                    print(f"✅ Migration Complete. New Count: {collection.count()}")
                    # Rename parquet to avoid re-reading?
                    # No, we check collection.count() == 0, so it won't run again if successful.

                except ImportError:
                    print("⚠️ Pandas not found. Skipping migration.")
                except Exception as e:
                    print(f"❌ Migration Failed: {e}")
        return collection
    except Exception as e:
        st.error(f"❌ Critical DB Error: {e}")
        return None


collection = load_chroma_collection()


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


embedding_model = None


def get_fingerprint(text):
    text = text.lower().strip()
    text = re.sub(r"^[a-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def verify_documents_batch(query, documents):
    if not documents:
        return []
    verifier = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    doc_text = ""
    for i, doc in enumerate(documents):
        doc_text += f"{i}. {doc}\n\n"

    prompt = f"""
    Role: RAG Content Verifier.
    Task: Filter out irrelevant logs.
    User Query: "{query}"
    Logs:
    {doc_text}
    Output: JSON list of indices to keep. e.g., [0, 2].
    """
    try:
        response = verifier.invoke(prompt).content
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end != -1:
            return json.loads(response[start:end])
        return []
    except:
        return []


def get_rag_context(query_text):
    if not collection:
        return "Database unavailable."
    try:
        clean_query = get_fingerprint(query_text)
        query_embedding = embedding_model.embed_query(clean_query)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)

        candidates = []
        if results["documents"] and results["distances"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                if dist < DISTANCE_THRESHOLD:
                    candidates.append(doc)

        if not candidates:
            return "No directly related logs found."

        return "\n\n".join(candidates)
    except:
        return "Error retrieving context."


def get_analysis_response(query_text, model_type="gpt-4o"):
    context = get_rag_context(query_text)

    # Reliably control via Python conditionals to completely block LLM conditional errors (hallucinations).
    fallback_instructions = ""
    if context == "No directly related logs found.":
        fallback_instructions = """
    [HALLUCINATION DEFENSE & ZERO-SHOT FALLBACK]
    You have NO historical data or context for this query. You must first evaluate the [User Log Input] before answering.

    Rule 1 (Irrelevant Input): If the [User Log Input] is a casual conversation, general question, or entirely irrelevant to IT/hardware/system logs (e.g., weather, greetings, general knowledge), you MUST refuse to answer and output ONLY the following text exactly:
    "Error: Invalid input. This system only analyzes hardware/system logs."

    Rule 2 (Valid Log, No Context): If the [User Log Input] IS a valid system, hardware, or software log, analyze it using your internal generalized knowledge. However, you MUST start your response exactly with the following warning:
    "[Warning: No historical data found. Analyzing based on generalized knowledge.]"
    """

    template = f"""
    You are a Memory Systems Specialist.
    
    [Verified Knowledge Base]
    {{context}}
    
    [User Log Input]
    {{input}}
    
    [CRITICAL SAFETY WARNING]
    1. DO NOT force a causal link if Context is normal vs User Error.
    2. Extract tools/commands even if context contradicts.
    3. If contradiction, state: "Internal logs show normal state... use suggested commands."
    {fallback_instructions}
    Provide Root Cause and Solution.
    """
    prompt = ChatPromptTemplate.from_template(template)

    if model_type == "gpt-4o":
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
    else:
        llm = ChatOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="wjs-log-expert",
            temperature=0,
        )

    chain = prompt | llm | StrOutputParser()
    try:
        analysis = chain.invoke({"context": context, "input": query_text})
        return {"analysis": analysis, "context": context}
    except Exception as e:
        return {"analysis": f"Error: {e}", "context": context}


# --- 4. Data Flywheel Logic (Auto-Eval & Admin) ---


def evaluate_and_log(query, response):
    evaluator = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
    Rate the quality of this Q&A pair for a Knowledge Base (0-100).
    Topic: Linux Sysadmin / Log Analysis.
    Criteria: Correctness, Clarity, actionable solution.
    
    Query: {query}
    Response: {response}
    
    Output strictly JSON: {{"score": 95, "reason": "..."}}
    """
    try:
        res = evaluator.invoke(prompt).content
        start = res.find("{")
        end = res.rfind("}") + 1
        data = json.loads(res[start:end])
        score = data.get("score", 0)

        if score > 80:
            item = {
                "id": str(uuid.uuid4()),
                "query": query,
                "response": response,
                "score": score,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            save_pending_review(item)
            return True
    except Exception as e:
        print(f"Eval Error: {e}")
    return False


def save_pending_review(item):
    pending = load_pending_reviews()
    pending.append(item)
    with open(PENDING_REVIEW_FILE, "w") as f:
        json.dump(pending, f, indent=2)


def load_pending_reviews():
    if not os.path.exists(PENDING_REVIEW_FILE):
        return []
    try:
        with open(PENDING_REVIEW_FILE, "r") as f:
            return json.load(f)
    except:
        return []


def approve_item(item):
    """Add to Vector DB with Old Version Logic (Manual Persist)"""
    try:
        # A) Embed: BGE-M3 (Dimension 1024)
        clean_key = get_fingerprint(item["query"])
        embedding = embedding_model.embed_query(clean_key)

        # B) Document: Raw text
        doc_text = f"[Log]\n{item['query']}\n\n[Solution]\n{item['response']}"

        # C) Add to collection
        collection.add(embeddings=[embedding], documents=[doc_text], ids=[item["id"]])

        # D) Force Persist (Crucial for 0.3.x)
        try:
            # chroma_client.persist() # REMOVED: Managed by PersistentClient
            print("💾 SUCCESS: Data persisted to disk.")
        except Exception as p_err:
            print(f"⚠️ Persist warning: {p_err}")

        # E) Remove from pending
        reject_item(item["id"])
        return True
    except Exception as e:
        st.error(f"Error saving to DB: {e}")
        return False


def reject_item(item_id):
    pending = load_pending_reviews()
    pending = [i for i in pending if i["id"] != item_id]
    with open(PENDING_REVIEW_FILE, "w") as f:
        json.dump(pending, f, indent=2)


# --- 5. UI Logic ---


def run_analysis(query_text, model_type="gpt-4o", model_label=None):
    if not query_text:
        return
    display_name = model_label if model_label else model_type

    with st.spinner(f"Analyzing Log with {display_name}..."):
        result_data = get_analysis_response(query_text, model_type)
        evaluate_and_log(query_text, result_data["analysis"])

        new_result = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "query": query_text,
            "model": display_name,
            "context": result_data["context"],
            "analysis": result_data["analysis"],
        }
        st.session_state.history.append(new_result)
        st.session_state.current_result = new_result
        st.session_state.show_dashboard = False
        st.session_state.is_admin = False
        st.rerun()


def reset_to_landing():
    st.session_state.current_result = None
    st.session_state.is_admin = False  # Reset Admin Mode
    st.session_state.show_dashboard = False
    st.rerun()


if __name__ == "__main__":
    # --- 1. Page Configuration ---
    st.set_page_config(
        page_title="Memory Quality Analysis Agent",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- 2. Custom CSS ---
    st.markdown(
        """
    <style>
        section[data-testid="stSidebar"] div.stButton > button {
            background-color: transparent !important;
            color: #333333 !important;
            border: none !important;
            text-align: left !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] div.stButton > button:hover {
            background-color: #f0f2f6 !important;
            color: #1428A0 !important;
            font-weight: 600 !important;
        }
        .stApp > header + div div.stButton > button {
            background-color: #1428A0 !important;
            color: white !important;
        }
        [data-testid="stDeployButton"] {display: none !important;}
        footer {visibility: hidden !important;}
        
        /* Admin Button Colors */
        div[data-testid="column"] button[kind="primary"] {
            background-color: #28a745 !important;
            border-color: #28a745 !important;
            color: white !important;
        }
        div[data-testid="column"] button[kind="secondary"] {
            background-color: white !important;
            border-color: #dc3545 !important;
            color: #dc3545 !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 디버깅용 출력 (터미널에서 확인 가능)
    print(f"📍 App uses DB Path: {CHROMA_DB_PATH}")

    # No longer needed to init here, global variable `collection` is loaded via cache

    embedding_model = load_embedding_model()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "show_dashboard" not in st.session_state:
        st.session_state.show_dashboard = False

    # --- Sidebar ---
    with st.sidebar:
        if st.button("➕ New Analysis", key="reset_chat_btn"):
            reset_to_landing()
        st.markdown("---")
        st.markdown("### 🤖 Model Settings")
        model_option = st.radio(
            "Select AI Model:",
            ("Cloud-GPT-4o", "Local-Llama-Secure"),
            key="model_selection",
        )

        st.markdown("---")
        st.caption("Log Analysis History")
        for i, item in enumerate(reversed(st.session_state.history)):
            if st.button(
                f"[{item['timestamp']}] {item['query'][:18]}...", key=f"hist_{i}"
            ):
                st.session_state.current_result = item
                st.session_state.show_dashboard = False  # Force Result View
                st.session_state.is_admin = False  # Admin logout on leaving Dashboard
                st.rerun()

        st.markdown("---")
        # Admin Section
        # Admin Section
        with st.expander("Admin"):
            with st.form(key="admin_login_form"):
                admin_pw = st.text_input("Password", type="password")
                submit_btn = st.form_submit_button("Login")

            if submit_btn:
                if admin_pw == "admin1234":
                    st.session_state.is_admin = True
                elif admin_pw:
                    st.session_state.is_admin = False
                    st.error("Incorrect Password")

            # Persistent Admin State UI
            if st.session_state.get("is_admin"):
                st.success("Admin Access Granted")

                # Persistent Dashboard Button (Styled)
                if st.button(
                    "🚀 Go to Dashboard",
                    key="goto_dash",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.show_dashboard = True
                    st.rerun()

    # --- Main Layout ---

    # Priority 1: Admin Dashboard
    if st.session_state.get("is_admin") and st.session_state.get("show_dashboard"):
        st.title("AI Training Center")
        st.markdown(
            "Review and approve high-quality Q&A pairs for AI's Knowledge Data Base."
        )

        tab1, tab2 = st.tabs(["📥 Review Pending", "🗄️ Manage Knowledge Base"])

        # --- Tab 1: Review Pending ---
        with tab1:
            pending_items = load_pending_reviews()
            if not pending_items:
                st.success("You're all caught up! No new data to review.")
            else:
                for item in pending_items:
                    with st.expander(f"[{item['score']}/100] {item['query'][:60]}..."):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.caption(f"ID: {item['id']}")
                            st.markdown("**Query:**")
                            st.code(item["query"])
                            st.markdown("**Response:**")
                            st.info(item["response"])
                        with c2:
                            st.metric("AI Confidence", f"{item['score']}/100")

                            b_col1, b_col2 = st.columns(2, gap="small")
                            with b_col1:
                                if st.button(
                                    "Approve",
                                    key=f"app_{item['id']}",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    if approve_item(item):
                                        st.toast("Saved to Knowledge Base!")
                                        st.rerun()
                            with b_col2:
                                if st.button(
                                    "Reject",
                                    key=f"rej_{item['id']}",
                                    type="secondary",
                                    use_container_width=True,
                                ):
                                    reject_item(item["id"])
                                    st.toast("Item Rejected.")
                                    st.rerun()

        # --- Tab 2: Manage Knowledge Base (Delete) ---
        with tab2:
            st.markdown("### 🗄️ Database Manager")
            st.caption("View and delete items from the ChromaDB (Showing newest first)")

            if collection:
                try:
                    # 1. Fetch ALL data (Chroma does not support 'last N' natively in 0.3.x)
                    all_data = collection.get()

                    ids = all_data["ids"]
                    docs = all_data["documents"]
                    metas = all_data["metadatas"]

                    if not ids:
                        st.info("The Database is empty.")
                    else:
                        # 2. Reverse lists to show newest first
                        # (Assuming retrieval is in insertion order or acceptable default order)
                        combined = list(zip(ids, docs, metas))
                        combined.reverse()  # Show Newest First

                        total_items = len(combined)
                        ITEMS_PER_PAGE = 20
                        total_pages = max(
                            1, (total_items + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
                        )

                        col1, col2 = st.columns([1, 4])
                        with col1:
                            page_number = st.number_input(
                                f"Page (1 to {total_pages})",
                                min_value=1,
                                max_value=total_pages,
                                step=1,
                                key="db_manager_page",
                            )

                        start_idx = (page_number - 1) * ITEMS_PER_PAGE
                        end_idx = start_idx + ITEMS_PER_PAGE
                        page_items = combined[start_idx:end_idx]

                        st.caption(
                            f"Showing items {start_idx + 1}-{min(end_idx, total_items)} of {total_items} total items"
                        )

                        for db_id, doc_text, meta in page_items:
                            # Preview Title
                            doc_preview = (
                                doc_text[:50].replace("\n", " ") + "..."
                                if doc_text
                                else "No Content"
                            )

                            with st.expander(f"🆔 {doc_preview}"):
                                st.caption(f"ID: {db_id}")

                                st.markdown("**Document Content:**")
                                st.code(doc_text, language="text")

                                st.markdown("**Metadata:**")
                                if meta:
                                    st.json(meta)
                                else:
                                    st.caption("No Metadata available")

                                # Delete Action
                                if st.button(
                                    "🗑️ Delete from DB",
                                    key=f"del_{db_id}",
                                    type="secondary",
                                ):
                                    try:
                                        collection.delete(ids=[db_id])
                                        # chroma_client.persist() # CRITICAL: Persist to disk
                                        st.toast(
                                            "Item permanently deleted from Database."
                                        )
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Delete Failed: {e}")
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
            else:
                st.error("Database connection not active.")

    # Priority 2: Result Page
    elif st.session_state.current_result is not None:
        st.markdown("<h2 style='color: #1428A0;'>Result</h2>", unsafe_allow_html=True)
        result = st.session_state.current_result

        with st.expander("Analyzed Log Data (Click to expand)"):
            st.code(result["query"], language="text")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📄 Retrieved Context")
            st.text_area("Context", value=result["context"], height=500, disabled=True)
        with c2:
            st.subheader("🤖 AI Analysis")
            st.markdown(result["analysis"])

        st.markdown("---")

    # Priority 3: Landing Page (Default)
    else:
        st.markdown(
            """<div style='text-align: center; margin-top: 10vh;'>
            <h2 style='color: #666;'>Welcome to</h2>
            <h1 style='color: #1428A0; font-size: 3.5rem;'>Memory Quality Analysis Agent</h1>
        </div>""",
            unsafe_allow_html=True,
        )

        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            query = st.text_area(
                "Log Input", height=150, placeholder="Paste your log here..."
            )

            model_map = {"Cloud-GPT-4o": "gpt-4o", "Local-Llama-Secure": "local-llama"}
            selected_label = st.session_state.get("model_selection", "Cloud-GPT-4o")
            selected_type = model_map.get(selected_label, "gpt-4o")

            if st.button("Analyze Log", use_container_width=True):
                run_analysis(query, selected_type, selected_label)
