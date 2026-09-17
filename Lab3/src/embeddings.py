import os
import re
import numpy as np
import pandas as pd

from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/dataset.csv"
OUTPUT_DIR = "output"

VOCAB_SIZE = 10000
EMBEDDING_DIM = 128

TOP_FREQUENT = 50
TOP_SIMILAR = 10
TOP_LEAST_SIMILAR = 10

RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

df = df.dropna(subset=["source_code"])

df["source_code"] = df["source_code"].astype(str)

documents = df["source_code"].tolist()

print(f"Number of source-code records: {len(documents)}")


# ============================================================
# 2. TRAIN SUBWORD TOKENIZER
# ============================================================

print("\nTraining BPE subword tokenizer...")

tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[
        "[UNK]",
        "[PAD]",
        "[CLS]",
        "[SEP]"
    ]
)

tokenizer.train_from_iterator(
    documents,
    trainer=trainer
)

vocab = tokenizer.get_vocab()

actual_vocab_size = len(vocab)

print(f"Actual vocabulary size: {actual_vocab_size}")


# ============================================================
# 3. COUNT TOKEN FREQUENCY
# ============================================================

print("\nCounting token frequencies...")

token_counter = Counter()

for i, text in enumerate(documents):

    encoding = tokenizer.encode(text)

    token_counter.update(encoding.tokens)

    if (i + 1) % 5000 == 0:
        print(f"Processed {i + 1}/{len(documents)} records")


# ============================================================
# 4. TOP 50 MOST FREQUENT TOKENS
# ============================================================

top_50 = token_counter.most_common(TOP_FREQUENT)

print("\n==============================================")
print("TOP 50 MOST FREQUENT TOKENS")
print("==============================================")

for rank, (token, frequency) in enumerate(top_50, start=1):
    print(f"{rank:2d}. {repr(token):30s} {frequency}")


# ============================================================
# SAVE TOP 50
# ============================================================

top50_df = pd.DataFrame(
    top_50,
    columns=["token", "frequency"]
)

top50_df.insert(
    0,
    "rank",
    range(1, len(top50_df) + 1)
)

top50_path = os.path.join(
    OUTPUT_DIR,
    "top_50_tokens.csv"
)

top50_df.to_csv(
    top50_path,
    index=False
)


# ============================================================
# 5. RANDOM EMBEDDING MATRIX
# ============================================================

print("\nInitializing random embedding matrix...")

embedding_matrix = np.random.normal(
    loc=0.0,
    scale=0.1,
    size=(actual_vocab_size, EMBEDDING_DIM)
).astype(np.float32)

print(
    f"Embedding matrix shape: "
    f"{embedding_matrix.shape}"
)


# ============================================================
# 6. NORMALIZE EMBEDDINGS
# ============================================================

norms = np.linalg.norm(
    embedding_matrix,
    axis=1,
    keepdims=True
)

normalized_embeddings = (
    embedding_matrix / (norms + 1e-12)
)


# ============================================================
# 7. CALCULATE SIMILARITY
# ============================================================

print("\nCalculating cosine similarities...")

# Cosine similarity between every pair of embeddings
similarity_matrix = (
    normalized_embeddings
    @ normalized_embeddings.T
)

# Remove self-similarity
np.fill_diagonal(
    similarity_matrix,
    -np.inf
)


# ============================================================
# 8. TOP 10 MOST SIMILAR EMBEDDINGS
# ============================================================

print("\n==============================================")
print("TOP 10 MOST SIMILAR EMBEDDING PAIRS")
print("==============================================")


# Only use upper triangle to avoid duplicates
upper_triangle = np.triu(
    similarity_matrix,
    k=1
)

most_similar_indices = np.argpartition(
    upper_triangle.ravel(),
    -TOP_SIMILAR
)[-TOP_SIMILAR:]

most_similar_indices = most_similar_indices[
    np.argsort(
        upper_triangle.ravel()[most_similar_indices]
    )[::-1]
]


id_to_token = {
    token_id: token
    for token, token_id in vocab.items()
}


similar_pairs = []

for flat_index in most_similar_indices:

    token_id_1, token_id_2 = np.unravel_index(
        flat_index,
        similarity_matrix.shape
    )

    similarity = similarity_matrix[
        token_id_1,
        token_id_2
    ]

    token_1 = id_to_token[token_id_1]
    token_2 = id_to_token[token_id_2]

    similar_pairs.append(
        (
            token_1,
            token_2,
            float(similarity)
        )
    )

    print(
        f"{repr(token_1):25s} "
        f"<-> "
        f"{repr(token_2):25s} "
        f"Similarity: {similarity:.6f}"
    )


# ============================================================
# 9. TOP 10 LEAST SIMILAR EMBEDDINGS
# ============================================================

print("\n==============================================")
print("TOP 10 LEAST SIMILAR EMBEDDING PAIRS")
print("==============================================")


# Replace lower triangle with +inf so we only
# consider unique pairs
lower_triangle = np.tril(
    similarity_matrix,
    k=-1
)

# Zeros outside the triangle must not be selected,
# so create a mask
mask = np.tril(
    np.ones_like(similarity_matrix),
    k=-1
)

least_values = np.where(
    mask == 1,
    similarity_matrix,
    np.inf
)

least_indices = np.argpartition(
    least_values.ravel(),
    TOP_LEAST_SIMILAR
)[:TOP_LEAST_SIMILAR]

least_indices = least_indices[
    np.argsort(
        least_values.ravel()[least_indices]
    )
]


least_similar_pairs = []

for flat_index in least_indices:

    token_id_1, token_id_2 = np.unravel_index(
        flat_index,
        similarity_matrix.shape
    )

    similarity = similarity_matrix[
        token_id_1,
        token_id_2
    ]

    token_1 = id_to_token[token_id_1]
    token_2 = id_to_token[token_id_2]

    least_similar_pairs.append(
        (
            token_1,
            token_2,
            float(similarity)
        )
    )

    print(
        f"{repr(token_1):25s} "
        f"<-> "
        f"{repr(token_2):25s} "
        f"Similarity: {similarity:.6f}"
    )


# ============================================================
# 10. SAVE MOST SIMILAR RESULTS
# ============================================================

similar_df = pd.DataFrame(
    similar_pairs,
    columns=[
        "token_1",
        "token_2",
        "cosine_similarity"
    ]
)

similar_df.insert(
    0,
    "rank",
    range(1, len(similar_df) + 1)
)

similar_path = os.path.join(
    OUTPUT_DIR,
    "top_10_most_similar.csv"
)

similar_df.to_csv(
    similar_path,
    index=False
)


# ============================================================
# 11. SAVE LEAST SIMILAR RESULTS
# ============================================================

least_df = pd.DataFrame(
    least_similar_pairs,
    columns=[
        "token_1",
        "token_2",
        "cosine_similarity"
    ]
)

least_df.insert(
    0,
    "rank",
    range(1, len(least_df) + 1)
)

least_path = os.path.join(
    OUTPUT_DIR,
    "top_10_least_similar.csv"
)

least_df.to_csv(
    least_path,
    index=False
)


# ============================================================
# 12. SAVE EMBEDDING MATRIX
# ============================================================

embedding_path = os.path.join(
    OUTPUT_DIR,
    "random_embedding_matrix.npy"
)

np.save(
    embedding_path,
    embedding_matrix
)


# ============================================================
# DONE
# ============================================================

print("\n==============================================")
print("EXERCISE 2 COMPLETE")
print("==============================================")

print(f"Top 50 tokens: {top50_path}")
print(f"Most similar: {similar_path}")
print(f"Least similar: {least_path}")
print(f"Embedding matrix: {embedding_path}")
