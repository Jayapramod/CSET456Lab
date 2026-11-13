import pandas as pd
import numpy as np
import re
from pathlib import Path
from gensim.models import Word2Vec


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "commit_history.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")

print("\nColumns:")
print(df.columns.tolist())

print("\nNumber of commits:")
print(len(df))


# ============================================================
# CHECK MESSAGE COLUMN
# ============================================================

if "message" not in df.columns:
    raise ValueError(
        "The dataset does not contain a 'message' column."
    )


# ============================================================
# REMOVE EMPTY COMMIT MESSAGES
# ============================================================

messages = df["message"].dropna().astype(str)

messages = messages[
    messages.str.strip() != ""
]

print("\nNumber of valid commit messages:")
print(len(messages))


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(message):

    # Convert to lowercase
    message = message.lower()

    # Extract words and useful programming terms
    tokens = re.findall(
        r"[a-zA-Z0-9_]+",
        message
    )

    return tokens


sentences = []

for message in messages:

    tokens = tokenize(message)

    if tokens:
        sentences.append(tokens)


print("\nExample tokenized commit messages:")

for sentence in sentences[:5]:
    print(sentence)


# ============================================================
# TRAIN WORD2VEC
# ============================================================

print("\nTraining Word2Vec model...")

model = Word2Vec(
    sentences=sentences,

    # Size of embedding vector
    vector_size=100,

    # Number of surrounding words
    window=5,

    # Ignore words appearing fewer than 1 time
    min_count=1,

    # Training algorithm
    # 1 = Skip-Gram
    # 0 = CBOW
    sg=1,

    # Number of training iterations
    epochs=20,

    # Number of worker threads
    workers=4,

    # Random seed for reproducibility
    seed=42
)


# ============================================================
# VOCABULARY
# ============================================================

vocabulary = list(model.wv.index_to_key)

print("\nVocabulary size:")
print(len(vocabulary))


# ============================================================
# SAVE WORD EMBEDDINGS
# ============================================================

embedding_vectors = []

for word in vocabulary:

    vector = model.wv[word]

    row = {
        "word": word
    }

    for i, value in enumerate(vector):

        row[f"dim_{i+1}"] = value

    embedding_vectors.append(row)


embedding_df = pd.DataFrame(
    embedding_vectors
)


OUTPUT_FILE = (
    OUTPUT_DIR /
    "word2vec_embeddings.csv"
)


embedding_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_FILE = (
    OUTPUT_DIR /
    "word2vec_commit_messages.model"
)

model.save(str(MODEL_FILE))


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================================")
print("WORD2VEC RESULTS")
print("==============================================")

print(f"Embedding dimensions: {model.vector_size}")

print(f"Vocabulary size: {len(vocabulary)}")

print("\nFirst 10 words:")

for word in vocabulary[:10]:

    print(
        f"{word} → "
        f"{model.wv[word][:5]} ..."
    )


print("\n==============================================")
print("FILES CREATED")
print("==============================================")

print(f"Embeddings:")
print(OUTPUT_FILE)

print(f"\nWord2Vec model:")
print(MODEL_FILE)
