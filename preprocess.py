"""
This script preprocesses raw system log files into a structured CSV format.
It uses regular expressions to extract the core log content by stripping dates, times, and hostnames.
"""

import pandas as pd
import re


def preprocess_logs(input_file, output_file):
    # Regex Pattern explanation:
    # ^                     Start of line
    # [A-Z][a-z]{2}\s+\d+   Date (e.g., Jun 14)
    # \s+                   Separator
    # \d{2}:\d{2}:\d{2}     Time
    # \s+                   Separator
    # \S+                   Hostname (non-whitespace)
    # \s+                   Separator
    # (.*)$                 Content (capture group)

    log_pattern = re.compile(r"^[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\S+\s+(.*)$")

    data = []

    try:
        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                match = log_pattern.match(line)
                if match:
                    content = match.group(1)
                    data.append(content)
                else:
                    # Skip lines that do not match the expected pattern
                    pass

        df = pd.DataFrame(data, columns=["Content"])
        df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(df)} lines.")

    except Exception as e:
        print(f"Error processing logs: {e}")


if __name__ == "__main__":
    input_path = "data/Linux_2k.log"
    output_path = "data/processed_logs.csv"
    preprocess_logs(input_path, output_path)
