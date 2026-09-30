import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

ONE_HOT_FILE = BASE_DIR / "output" / "one_hot_embeddings.csv"
FREQUENCY_FILE = BASE_DIR / "output" / "frequency_embeddings.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# COSINE SIMILARITY FUNCTION
# ============================================================

def cosine_similarity(vector_a, vector_b):

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# 1. ONE-HOT COSINE SIMILARITY
# ============================================================

one_hot_df = pd.read_csv(ONE_HOT_FILE)

tokens = one_hot_df["token"].tolist()

# Only take embedding columns
one_hot_vectors = one_hot_df.drop(columns=["token"]).values


one_hot_results = []

for i in range(len(tokens)):

    for j in range(i + 1, len(tokens)):

        similarity = cosine_similarity(
            one_hot_vectors[i],
            one_hot_vectors[j]
        )

        one_hot_results.append({
            "token_1": tokens[i],
            "token_2": tokens[j],
            "cosine_similarity": similarity
        })


one_hot_similarity_df = pd.DataFrame(one_hot_results)

one_hot_output = OUTPUT_DIR / "one_hot_pairwise_cosine.csv"

one_hot_similarity_df.to_csv(
    one_hot_output,
    index=False
)


print("====================================")
print("ONE-HOT PAIRWISE COSINE SIMILARITY")
print("====================================")

print(one_hot_similarity_df)

print(f"\nSaved to: {one_hot_output}")


# ============================================================
# 2. FREQUENCY COSINE SIMILARITY
# ============================================================

frequency_df = pd.read_csv(FREQUENCY_FILE)

tokens = frequency_df["token"].tolist()

# Use frequency as the embedding
frequency_vectors = frequency_df[["frequency"]].values


frequency_results = []

for i in range(len(tokens)):

    for j in range(i + 1, len(tokens)):

        similarity = cosine_similarity(
            frequency_vectors[i],
            frequency_vectors[j]
        )

        frequency_results.append({
            "token_1": tokens[i],
            "token_2": tokens[j],
            "cosine_similarity": similarity
        })


frequency_similarity_df = pd.DataFrame(frequency_results)

frequency_output = OUTPUT_DIR / "frequency_pairwise_cosine.csv"

frequency_similarity_df.to_csv(
    frequency_output,
    index=False
)


print("\n====================================")
print("FREQUENCY PAIRWISE COSINE SIMILARITY")
print("====================================")

print(frequency_similarity_df)

print(f"\nSaved to: {frequency_output}")


# ============================================================
# 3. SUMMARY
# ============================================================

print("\n====================================")
print("SUMMARY")
print("====================================")

print(f"Number of tokens: {len(tokens)}")

print(
    f"Number of token pairs: "
    f"{len(tokens) * (len(tokens) - 1) // 2}"
)

print("\nOutput files:")
print("1.", one_hot_output)
print("2.", frequency_output)
