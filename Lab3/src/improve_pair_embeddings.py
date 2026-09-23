import os
import numpy as np
import pandas as pd
from tokenizers import Tokenizer


# ============================================================
# FILE PATHS
# ============================================================

TOKENIZER_PATH = "output/tokenizer_analysis/subword_tokenizer.json"

EMBEDDING_PATH = "output/random_embedding_matrix.npy"

OUTPUT_DIR = "output/pair_embedding_training"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

# How strongly the vectors are moved toward each other
LEARNING_RATE = 0.05

# Number of times we update the pairs
EPOCHS = 100

# Random embedding dimension
EMBEDDING_DIM = 128


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

vocab = tokenizer.get_vocab()


# ============================================================
# LOAD ORIGINAL RANDOM EMBEDDING MATRIX
# ============================================================

original_embeddings = np.load(
    EMBEDDING_PATH
)

print(
    "Original embedding matrix shape:",
    original_embeddings.shape
)


# Make a copy so the original matrix is not changed
updated_embeddings = original_embeddings.copy()


# ============================================================
# TOKEN PAIRS
# ============================================================

pairs = [
    (".", '"""'),
    ('"', '""'),
    ("=", "()"),
    ('"""', '""")')
]


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector1, vector2):

    denominator = (
        np.linalg.norm(vector1)
        *
        np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0.0

    return np.dot(
        vector1,
        vector2
    ) / denominator


# ============================================================
# CHECK TOKENS
# ============================================================

print("\nChecking tokens...")

valid_pairs = []

for token1, token2 in pairs:

    if token1 not in vocab:
        print(
            f"WARNING: {repr(token1)} "
            "not found in vocabulary"
        )
        continue

    if token2 not in vocab:
        print(
            f"WARNING: {repr(token2)} "
            "not found in vocabulary"
        )
        continue

    valid_pairs.append(
        (token1, token2)
    )

    print(
        f"OK: {repr(token1)} <-> {repr(token2)}"
    )


# ============================================================
# CALCULATE INITIAL SIMILARITIES
# ============================================================

initial_results = []

for token1, token2 in valid_pairs:

    id1 = vocab[token1]
    id2 = vocab[token2]

    vector1 = original_embeddings[id1]
    vector2 = original_embeddings[id2]

    similarity = cosine_similarity(
        vector1,
        vector2
    )

    initial_results.append({
        "token_1": token1,
        "token_2": token2,
        "token_1_id": id1,
        "token_2_id": id2,
        "initial_similarity": similarity
    })


# ============================================================
# NAIVE EMBEDDING UPDATE
# ============================================================

print("\nStarting embedding updates...")

for epoch in range(EPOCHS):

    for token1, token2 in valid_pairs:

        id1 = vocab[token1]
        id2 = vocab[token2]

        vector1 = updated_embeddings[id1]
        vector2 = updated_embeddings[id2]

        # ----------------------------------------------------
        # Move vector 1 toward vector 2
        # ----------------------------------------------------

        new_vector1 = (
            vector1
            + LEARNING_RATE
            * (vector2 - vector1)
        )

        # ----------------------------------------------------
        # Move vector 2 toward vector 1
        # ----------------------------------------------------

        new_vector2 = (
            vector2
            + LEARNING_RATE
            * (vector1 - vector2)
        )

        # ----------------------------------------------------
        # Normalize vectors
        # ----------------------------------------------------

        norm1 = np.linalg.norm(new_vector1)

        norm2 = np.linalg.norm(new_vector2)

        if norm1 != 0:
            new_vector1 = (
                new_vector1 / norm1
            )

        if norm2 != 0:
            new_vector2 = (
                new_vector2 / norm2
            )

        # ----------------------------------------------------
        # Update embedding matrix
        # ----------------------------------------------------

        updated_embeddings[id1] = new_vector1
        updated_embeddings[id2] = new_vector2


# ============================================================
# CALCULATE FINAL SIMILARITIES
# ============================================================

final_results = []

for token1, token2 in valid_pairs:

    id1 = vocab[token1]
    id2 = vocab[token2]

    original_vector1 = (
        original_embeddings[id1]
    )

    original_vector2 = (
        original_embeddings[id2]
    )

    updated_vector1 = (
        updated_embeddings[id1]
    )

    updated_vector2 = (
        updated_embeddings[id2]
    )

    initial_similarity = cosine_similarity(
        original_vector1,
        original_vector2
    )

    final_similarity = cosine_similarity(
        updated_vector1,
        updated_vector2
    )

    improvement = (
        final_similarity
        - initial_similarity
    )

    final_results.append({
        "token_1": token1,
        "token_2": token2,
        "token_1_id": id1,
        "token_2_id": id2,
        "initial_similarity": initial_similarity,
        "final_similarity": final_similarity,
        "improvement": improvement
    })


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    final_results
)

results_df["initial_similarity"] = (
    results_df["initial_similarity"]
    .round(6)
)

results_df["final_similarity"] = (
    results_df["final_similarity"]
    .round(6)
)

results_df["improvement"] = (
    results_df["improvement"]
    .round(6)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("BEFORE vs AFTER EMBEDDING SIMILARITY")
print("=" * 80)

print(
    results_df[
        [
            "token_1",
            "token_2",
            "initial_similarity",
            "final_similarity",
            "improvement"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "before_after_similarity.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE UPDATED EMBEDDING MATRIX
# ============================================================

updated_embedding_path = os.path.join(
    OUTPUT_DIR,
    "updated_embedding_matrix.npy"
)

np.save(
    updated_embedding_path,
    updated_embeddings
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

parameters = pd.DataFrame({
    "parameter": [
        "embedding_dimension",
        "learning_rate",
        "epochs"
    ],
    "value": [
        EMBEDDING_DIM,
        LEARNING_RATE,
        EPOCHS
    ]
})

parameters.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_parameters.csv"
    ),
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    f"\nResults saved to:\n{results_path}"
)

print(
    f"\nUpdated embeddings saved to:\n"
    f"{updated_embedding_path}"
)
