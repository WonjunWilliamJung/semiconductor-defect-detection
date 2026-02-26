"""
This script automates the process of generating root cause and solution labels for real log files.
It uses GPT-4o to analyze unique log patterns and outputs a JSONL dataset for RAG.
"""

import os
import json
import re
import sys
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load Configuration
load_dotenv(".env.local")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ Error: OPENAI_API_KEY not found in .env.local")
    print("Please ensure your API key is set.")
    sys.exit(1)

INPUT_LOG_FILE = "data/Linux_2k.log"
OUTPUT_FILE = "data/real_dataset_rag.jsonl"
COST_PER_REQ = 0.005  # Approx cost for input + output (gpt-4o)


def get_fingerprint(text):
    """
    Extracts the static pattern of a log message (removes dynamic parts).
    Matches the logic in build_vector_db.py.
    """
    text = text.lower().strip()
    # Remove syslog header (Month Day HH:MM:SS Hostname)
    text = re.sub(r"^[a-z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+\s+", "", text)
    # Remove PIDs: [12345] -> empty
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def main():
    print("--- Automated Real Log Labeling (GPT-4o) ---")

    # 2. Load & Deduplicate Data
    print(f"Reading {INPUT_LOG_FILE}...")
    unique_patterns = set()
    raw_to_pattern = {}  # Keep one raw example for the prompt

    try:
        with open(INPUT_LOG_FILE, "r") as f:
            for line in f:
                if line.strip():
                    fp = get_fingerprint(line)
                    if fp not in unique_patterns:
                        unique_patterns.add(fp)
                        raw_to_pattern[fp] = line.strip()
    except FileNotFoundError:
        print(f"Error: File {INPUT_LOG_FILE} not found.")
        return

    count = len(unique_patterns)
    est_cost = count * COST_PER_REQ

    print(f"✅ Found {count} unique log patterns.")
    print(f"💰 Estimated Cost: ~${est_cost:.3f}")

    # 3. UX: Confirmation
    print("\nPress ENTER to start labeling, or Ctrl+C to cancel...")
    input()

    # 4. Initialize AI
    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []

    print("Starting Labeling Process...")

    # 5. Labeling Loop
    # We use tqdm for progress bar
    with open(OUTPUT_FILE, "w") as f_out:
        for fp in tqdm(unique_patterns, desc="Labeling Logs"):
            raw_log = raw_to_pattern[fp]

            prompt = f"""
Analyzed the following Linux log message.
1. Identify the Root Cause.
2. Provide a specific Solution or Command to fix/investigate.
Keep the response concise and actionable.

[Log Message]
{raw_log}
"""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert Linux System Administrator.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )

                solution = response.choices[0].message.content

                # Format as ChatML JSONL
                entry = {
                    "messages": [
                        {"role": "user", "content": raw_log},
                        {"role": "assistant", "content": solution},
                    ]
                }

                # Write immediately (streaming save)
                f_out.write(json.dumps(entry) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\n❌ Error labeling log: {fp[:30]}... | {e}")
                continue

    print(f"\n✅ Labeling Complete! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
