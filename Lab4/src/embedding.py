import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "20_tokens.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Load tokens
# -----------------------------
df = pd.read_csv(INPUT_FILE)

# Change "token" if your CSV uses a different column name
tokens = df["token"].astype(str).tolist()

print("Tokens:")
print(tokens)
print(f"\nTotal tokens: {len(tokens)}")


# ============================================================
# 1. ONE-HOT ENCODING
# ============================================================

vocab = sorted(set(tokens))
vocab_size = len(vocab)

one_hot_vectors = []

for token in tokens:
    vector = np.zeros(vocab_size, dtype=int)

    index = vocab.index(token)
    vector[index] = 1

    one_hot_vectors.append(vector)


one_hot_df = pd.DataFrame(
    one_hot_vectors,
    columns=vocab
)

one_hot_df.insert(0, "token", tokens)

one_hot_file = OUTPUT_DIR / "one_hot_embeddings.csv"
one_hot_df.to_csv(one_hot_file, index=False)

print("\nOne-Hot Embeddings:")
print(one_hot_df)

print(f"\nSaved to: {one_hot_file}")


# ============================================================
# 2. COUNT / FREQUENCY ENCODING
# ============================================================

# Count how many times each token appears
token_counts = Counter(tokens)

total_tokens = len(tokens)

frequency_data = []

for token in tokens:
    count = token_counts[token]

    frequency = count / total_tokens

    frequency_data.append({
        "token": token,
        "count": count,
        "frequency": frequency
    })


frequency_df = pd.DataFrame(frequency_data)

frequency_file = OUTPUT_DIR / "frequency_embeddings.csv"
frequency_df.to_csv(frequency_file, index=False)

print("\nCount / Frequency Embeddings:")
print(frequency_df)

print(f"\nSaved to: {frequency_file}")