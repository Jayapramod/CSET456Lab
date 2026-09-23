import numpy as np
from tokenizers import Tokenizer


# ============================================================
# FILE PATHS
# ============================================================

TOKENIZER_PATH = "output/tokenizer_analysis/subword_tokenizer.json"

EMBEDDING_PATH = "output/random_embedding_matrix.npy"


# ============================================================
# LOAD TOKENIZER AND EMBEDDING MATRIX
# ============================================================

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

embedding_matrix = np.load(EMBEDDING_PATH)

vocab = tokenizer.get_vocab()


# ============================================================
# TOKEN PAIRS
# ============================================================

pairs = [
    (".", '"""'),
    ('"', '""'),
    ("(", '(""'),
    ("=", "()"),
    ('"""', '""")')
]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector1, vector2):

    dot_product = np.dot(vector1, vector2)

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# ============================================================
# CALCULATE SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("TOKEN EMBEDDING SIMILARITY")
print("=" * 70)

results = []


for pair_number, (token1, token2) in enumerate(pairs, start=1):

    print(f"\nPair {pair_number}")
    print("-" * 50)

    # Check whether tokens exist in vocabulary
    if token1 not in vocab:
        print(f"Token 1 {repr(token1)} NOT FOUND in vocabulary")
        continue

    if token2 not in vocab:
        print(f"Token 2 {repr(token2)} NOT FOUND in vocabulary")
        continue

    # Get token IDs
    token1_id = vocab[token1]
    token2_id = vocab[token2]

    # Get embedding vectors
    vector1 = embedding_matrix[token1_id]
    vector2 = embedding_matrix[token2_id]

    # Calculate cosine similarity
    similarity = cosine_similarity(
        vector1,
        vector2
    )

    # Print information
    print(f"Token 1       : {repr(token1)}")
    print(f"Token 2       : {repr(token2)}")

    print(f"Token 1 ID    : {token1_id}")
    print(f"Token 2 ID    : {token2_id}")

    print(f"Vector 1     : {vector1}")
    print(f"Vector 2     : {vector2}")

    print(
        f"Cosine Similarity : {similarity:.6f}"
    )

    results.append({
        "pair": pair_number,
        "token_1": token1,
        "token_2": token2,
        "token_1_id": token1_id,
        "token_2_id": token2_id,
        "cosine_similarity": similarity
    })


# ============================================================
# SAVE RESULTS
# ============================================================

import pandas as pd

results_df = pd.DataFrame(results)

output_path = (
    "output/tokenizer_analysis/"
    "specified_token_similarities.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "pair",
            "token_1",
            "token_2",
            "cosine_similarity"
        ]
    ].to_string(index=False)
)

print(
    f"\nResults saved to: {output_path}"
)

