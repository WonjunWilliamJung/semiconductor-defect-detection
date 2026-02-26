"""
This script generates a synthetic RAG dataset by taking real log lines,
creating variations using an LLM, and generating root cause analysis labels.
"""

import os
import json
import random
import logging
import re
import main
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

INPUT_LOG_FILE = "data/Linux_2k.log"
OUTPUT_FILE = "data/synthetic_dataset_rag.jsonl"
TARGET_ADDITIONAL_SAMPLES = 500
VARIATIONS_PER_SAMPLE = 3

logging.getLogger("chromadb").setLevel(logging.ERROR)


def get_log_fingerprint(text):

    try:
        if isinstance(text, str) and text.strip().startswith("{"):
            data = json.loads(text)
            text = data["messages"][0]["content"]
    except:
        pass

    text = text.lower()

    text = re.sub(r"\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def load_existing_fingerprints(filepath):
    fingerprints = set()
    if not os.path.exists(filepath):
        return fingerprints

    print("🔍 Scanning existing file for duplicates...")
    try:
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    fp = get_log_fingerprint(line)
                    if fp:
                        fingerprints.add(fp)
    except Exception as e:
        print(f"Warning reading existing file: {e}")

    print(f"   -> Found {len(fingerprints)} unique patterns already processed.")
    return fingerprints


def load_unique_samples(filepath, existing_fingerprints, n_needed):
    candidates = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fp = get_log_fingerprint(line)

            if fp not in existing_fingerprints:
                candidates.append(line)

    print(f"   -> {len(candidates)} unused log lines available in source.")

    if len(candidates) < n_needed:
        print(
            f"⚠️ Warning: Only {len(candidates)} unique lines left. Using all of them."
        )
        return candidates

    random.seed(42)
    return random.sample(candidates, n_needed)


def clean_log_text(line):
    if not line:
        return None
    line = line.strip()
    if line.startswith("```"):
        line = line.replace("```log", "").replace("```text", "").replace("```", "")
    if line.endswith("```"):
        line = line.replace("```", "")
    line = line.strip()
    if len(line) < 10:
        return None
    if all(char in "\"'`.,:;[]{}() " for char in line):
        return None
    return line


def generate_variations(log_entry, llm):
    prompt = f"""You are a Linux Log Generator. 
    Strictly maintain the log structure and error keywords.
    Only vary dynamic parts like Timestamp, IP, PID, and Hostname. 
    Do NOT invent new error codes.

    Original Log: '{log_entry}'
    
    Generate {VARIATIONS_PER_SAMPLE} variations. Return ONLY the log lines, separated by a newline."""

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        raw_lines = response.content.split("\n")
        cleaned = []
        for line in raw_lines:
            cl = clean_log_text(line)
            if cl:
                cleaned.append(cl)
        return list(dict.fromkeys(cleaned))[:VARIATIONS_PER_SAMPLE]
    except Exception as e:
        print(f"Gen Error: {e}")
        return []


def generate_rag_label(log_variation, llm):
    context = main.get_rag_context(log_variation)
    if not context:
        context = "No historical context found."

    prompt = f"""Analyze the following log entry based on the provided context.
    
    Log Entry: {log_variation}
    Context: {context}
    
    Analyze the root cause and suggest a solution."""

    messages = [
        SystemMessage(content="You are a Samsung Memory QA Expert."),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        return response.content
    except:
        return "Analysis failed."


def main_generation():
    print(f"--- Smart Resume: Targeting {TARGET_ADDITIONAL_SAMPLES} NEW samples ---")

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        return

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    existing_fingerprints = load_existing_fingerprints(OUTPUT_FILE)

    samples = load_unique_samples(
        INPUT_LOG_FILE, existing_fingerprints, TARGET_ADDITIONAL_SAMPLES
    )

    print(f"🚀 Starting generation for {len(samples)} unique samples...")
    print("💰 Estimated Cost: ~$10.00")

    with open(OUTPUT_FILE, "a") as f:
        for i, original_log in enumerate(samples):
            variations = generate_variations(original_log, llm)

            saved_count = 0
            for var in variations:
                analysis = generate_rag_label(var, llm)

                entry = {
                    "messages": [
                        {"role": "user", "content": var},
                        {"role": "assistant", "content": analysis},
                    ]
                }

                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
                saved_count += 1

            print(
                f"[{i + 1}/{len(samples)}] Saved {saved_count} variations (Unique Check Passed)..."
            )

    print("\n✅ Final Generation Complete!")
    print(f"Check total lines: wc -l {OUTPUT_FILE}")


if __name__ == "__main__":
    main_generation()
