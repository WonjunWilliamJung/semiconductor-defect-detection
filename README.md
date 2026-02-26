# Log Analysis with RAG

A robust, AI-powered system log analysis tool utilizing Retrieval-Augmented Generation (RAG). This project is designed to automatically ingest system logs, retrieve relevant historical context, and provide fast, accurate root-cause analysis and actionable solutions using Large Language Models (LLMs).

## Features

- **Intelligent Log Analysis:** Uses GPT-4o or a local Llama model (via Ollama) to analyze system or hardware logs and offer step-by-step solutions.
- **RAG-Powered Context Retrieval:** Leverages ChromaDB and HuggingFace embeddings (`BAAI/bge-m3`) for accurate semantic search of previous log events and their documented resolutions.
- **Zero-Shot Fallback & Hallucination Defense:** Intelligently handles unknown logs, ensuring the LLM relies on generalized knowledge safely when no direct historical context is found.
- **Automated Data Pipeline:** Includes scripts for parsing raw logs, synthesizing dataset variations, labeling real data, and building the vector database.
- **Interactive UI:** A streamlined Streamlit interface for chat-based log troubleshooting and vector database administration.
- **Performance Benchmarking:** Built-in benchmarking tools to compare the accuracy and performance retention of different LLMs against a ground-truth dataset.

## Project Structure

- `app.py`: The main Streamlit web application serving the UI and RAG pipeline.
- `build_vector_db.py`: Script to construct the ChromaDB vector database using a dual-strategy (clean fingerprint search / raw display).
- `preprocess.py`: Extracts and preprocesses raw system log files into a structured format.
- `generate_data.py` & `generate_real_labels.py`: Utility scripts used to automate the generation of synthetic datasets and label real log data.
- `benchmark_models.py`: Runs a performance benchmark comparing GPT-4o and local models using cosine similarity matching.
- `check_db.py`, `check_threshold.py`: Utility scripts for database inspection and calibrating the optimal RAG distance threshold.

## Technology Stack

- **Frontend:** Streamlit
- **Backend/AI:** LangChain, OpenAI API, Ollama (Local LLMs)
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence Transformers (HuggingFace)
- **Data Processing:** Pandas, Regex

## Getting Started

1. Clone this repository to your local machine.
2. Set up your `.env.local` file with your `OPENAI_API_KEY`.
3. Install the necessary Python dependencies.
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
