import os
import json
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, WordLevelTrainer


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/dataset.csv"
OUTPUT_DIR = "output/tokenizer_analysis"

# Same embedding dimension for all tokenizers
EMBEDDING_DIM = 128

# Maximum number of rows to use.
# Set to None to use the complete dataset.
MAX_ROWS = None

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

if "source_code" not in df.columns:
    raise ValueError("Dataset must contain a 'source_code' column.")

df["source_code"] = df["source_code"].fillna("").astype(str)

if MAX_ROWS is not None:
    df = df.head(MAX_ROWS)

print(f"Number of rows: {len(df)}")


# ============================================================
# PREPARE CORPUS
# ============================================================

corpus = df["source_code"].tolist()

print("Preparing source-code corpus...")

# Remove completely empty source-code entries
corpus = [text for text in corpus if text.strip()]

print(f"Non-empty source-code entries: {len(corpus)}")


# ============================================================
# CHARACTER-LEVEL TOKENIZER
# ============================================================

print("\nTraining Character-level tokenizer...")

# Collect every unique character
unique_characters = set()

for text in corpus:
    unique_characters.update(text)

char_tokens = sorted(unique_characters)

# Add special tokens
char_vocab = {
    "[PAD]": 0,
    "[UNK]": 1
}

for character in char_tokens:
    if character not in char_vocab:
        char_vocab[character] = len(char_vocab)

char_tokenizer = Tokenizer(
    WordLevel(
        vocab=char_vocab,
        unk_token="[UNK]"
    )
)

# Character tokenizer:
# split every character
from tokenizers.pre_tokenizers import Split

char_tokenizer.pre_tokenizer = Split(
    pattern="",
    behavior="isolated"
)


# ============================================================
# WORD-LEVEL TOKENIZER
# ============================================================

print("Training Word-level tokenizer...")

word_tokenizer = Tokenizer(
    WordLevel(
        unk_token="[UNK]"
    )
)

word_tokenizer.pre_tokenizer = Whitespace()

word_trainer = WordLevelTrainer(
    special_tokens=["[PAD]", "[UNK]"],
    min_frequency=1
)

word_tokenizer.train_from_iterator(
    corpus,
    trainer=word_trainer
)


# ============================================================
# SUBWORD BPE TOKENIZER
# ============================================================

print("Training Subword BPE tokenizer...")

subword_tokenizer = Tokenizer(
    BPE(
        unk_token="[UNK]"
    )
)

subword_tokenizer.pre_tokenizer = Whitespace()

subword_trainer = BpeTrainer(
    vocab_size=10000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]"]
)

subword_tokenizer.train_from_iterator(
    corpus,
    trainer=subword_trainer
)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_tokenizer(name, tokenizer, texts):
    print(f"\nAnalyzing {name} tokenizer...")

    sequence_lengths = []

    for i, text in enumerate(texts):
        encoding = tokenizer.encode(text)
        sequence_lengths.append(len(encoding.ids))

        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1}/{len(texts)}")

    vocab_size = tokenizer.get_vocab_size()

    average_sequence_length = (
        sum(sequence_lengths) / len(sequence_lengths)
        if sequence_lengths
        else 0
    )

    max_sequence_length = (
        max(sequence_lengths)
        if sequence_lengths
        else 0
    )

    min_sequence_length = (
        min(sequence_lengths)
        if sequence_lengths
        else 0
    )

    total_tokens = sum(sequence_lengths)

    embedding_parameters = vocab_size * EMBEDDING_DIM

    embedding_size_bytes = (
        embedding_parameters * 4
    )  # float32 = 4 bytes

    embedding_size_mb = (
        embedding_size_bytes / (1024 ** 2)
    )

    results = {
        "tokenizer": name,
        "vocab_size": vocab_size,
        "embedding_dimension": EMBEDDING_DIM,
        "embedding_parameters": embedding_parameters,
        "embedding_size_MB_float32": round(
            embedding_size_mb, 2
        ),
        "average_sequence_length": round(
            average_sequence_length, 2
        ),
        "minimum_sequence_length": min_sequence_length,
        "maximum_sequence_length": max_sequence_length,
        "total_tokens": total_tokens
    }

    return results, sequence_lengths


# ============================================================
# RUN ANALYSIS
# ============================================================

all_results = []
sequence_data = {}


# Character
char_results, char_lengths = analyze_tokenizer(
    "Character",
    char_tokenizer,
    corpus
)

all_results.append(char_results)
sequence_data["Character"] = char_lengths


# Word
word_results, word_lengths = analyze_tokenizer(
    "Word",
    word_tokenizer,
    corpus
)

all_results.append(word_results)
sequence_data["Word"] = word_lengths


# Subword
subword_results, subword_lengths = analyze_tokenizer(
    "Subword-BPE",
    subword_tokenizer,
    corpus
)

all_results.append(subword_results)
sequence_data["Subword-BPE"] = subword_lengths


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_path = os.path.join(
    OUTPUT_DIR,
    "tokenizer_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


# ============================================================
# SAVE SEQUENCE LENGTHS
# ============================================================

sequence_df = pd.DataFrame(sequence_data)

sequence_path = os.path.join(
    OUTPUT_DIR,
    "sequence_lengths.csv"
)

sequence_df.to_csv(
    sequence_path,
    index=False
)


# ============================================================
# SAVE TOKENIZERS
# ============================================================

char_tokenizer.save(
    os.path.join(
        OUTPUT_DIR,
        "character_tokenizer.json"
    )
)

word_tokenizer.save(
    os.path.join(
        OUTPUT_DIR,
        "word_tokenizer.json"
    )
)

subword_tokenizer.save(
    os.path.join(
        OUTPUT_DIR,
        "subword_tokenizer.json"
    )
)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "tokenizer_analysis.json"
)

with open(json_path, "w") as f:
    json.dump(
        all_results,
        f,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZER COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "tokenizer",
            "vocab_size",
            "average_sequence_length",
            "maximum_sequence_length",
            "embedding_parameters",
            "embedding_size_MB_float32"
        ]
    ].to_string(index=False)
)

print("\nAnalysis completed!")

print(f"\nResults saved to:")
print(results_path)

print(sequence_path)
print(json_path)
