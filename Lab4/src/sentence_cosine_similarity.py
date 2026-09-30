import pandas as pd
import numpy as np
from pathlib import Path


# -----------------------------------
# Paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output" / "sentence_embeddings.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------
# Load embeddings
# -----------------------------------

df = pd.read_csv(INPUT_FILE)

tokens = df["token"].tolist()

embeddings = df.drop(
    columns=["token"]
).values


# -----------------------------------
# Cosine similarity function
# -----------------------------------

def cosine_similarity(vector_a, vector_b):

    dot_product = np.dot(vector_a, vector_b)

    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (
        magnitude_a * magnitude_b
    )


# -----------------------------------
# Calculate pairwise similarity
# -----------------------------------

results = []

for i in range(len(tokens)):

    for j in range(i + 1, len(tokens)):

        similarity = cosine_similarity(
            embeddings[i],
            embeddings[j]
        )

        results.append({
            "token_1": tokens[i],
            "token_2": tokens[j],
            "cosine_similarity": similarity
        })


# -----------------------------------
# Save results
# -----------------------------------

result_df = pd.DataFrame(results)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "sentence_pairwise_cosine.csv"
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# -----------------------------------
# Display results
# -----------------------------------

print("\n======================================")
print("SENTENCE EMBEDDING COSINE SIMILARITY")
print("======================================")

print(result_df.to_string(index=False))

print("\nNumber of tokens:", len(tokens))

print(
    "Number of pairs:",
    len(tokens) * (len(tokens) - 1) // 2
)

print(f"\nSaved to: {OUTPUT_FILE}")
