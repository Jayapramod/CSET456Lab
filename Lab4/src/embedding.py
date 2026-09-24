import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "20_tokens.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------
# Read CSV
# -----------------------------------

df = pd.read_csv(
    INPUT_FILE,
    header=None,
    names=["rank", "token", "count"]
)

print("Input data:")
print(df)

print("\nColumns:")
print(df.columns.tolist())


# -----------------------------------
# Extract tokens
# -----------------------------------

tokens = df["token"].astype(str).tolist()

print("\nSelected Tokens:")
for i, token in enumerate(tokens, start=1):
    print(f"{i}. {token}")


# ============================================================
# 1. ONE-HOT ENCODING
# ============================================================

# Create vocabulary
vocab = tokens

vocab_size = len(vocab)

one_hot_vectors = []

for token in tokens:

    vector = np.zeros(vocab_size, dtype=int)

    index = vocab.index(token)

    vector[index] = 1

    one_hot_vectors.append(vector)


# Create DataFrame
one_hot_df = pd.DataFrame(
    one_hot_vectors,
    columns=vocab
)

# Add token column
one_hot_df.insert(0, "token", tokens)


# Save
one_hot_file = OUTPUT_DIR / "one_hot_embeddings.csv"

one_hot_df.to_csv(
    one_hot_file,
    index=False
)

print("\n====================================")
print("ONE-HOT EMBEDDINGS")
print("====================================")

print(one_hot_df)

print(f"\nSaved to: {one_hot_file}")


# ============================================================
# 2. COUNT / FREQUENCY EMBEDDING
# ============================================================

frequency_df = df[["token", "count"]].copy()

# Total frequency
total_count = frequency_df["count"].sum()

# Calculate normalized frequency
frequency_df["frequency"] = (
    frequency_df["count"] / total_count
)


# Save
frequency_file = OUTPUT_DIR / "frequency_embeddings.csv"

frequency_df.to_csv(
    frequency_file,
    index=False
)

print("\n====================================")
print("COUNT / FREQUENCY EMBEDDINGS")
print("====================================")

print(frequency_df)

print(f"\nSaved to: {frequency_file}")