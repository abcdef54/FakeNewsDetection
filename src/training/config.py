import os
import torch

# --- PATHS ---
DATA_DIR = "../../Organized"
CACHE_PICKLE_PATH = "../cache_dataset.pkl"
RAG_CACHE_PATH = "../../rag_cache"
CHECKPOINT_DIR = "../../checkpoints"

TRAIN_FILES = [
    os.path.join(DATA_DIR, "FAKE_Social.jsonl"),
    os.path.join(DATA_DIR, "REAL_Social.jsonl"),
    os.path.join(DATA_DIR, "FAKE_Article.jsonl"),
    os.path.join(DATA_DIR, "REAL_Article.jsonl"),
]

# --- MODEL NAMES ---
PHOBERT_NAME = "vinai/phobert-base-v2"
VISOBERT_NAME = "uitnlp/visobert"

# --- HYPERPARAMETERS ---
# PhoBERT-base-v2: max_position_embeddings=258 → max seq len = 256
# ViSoBERT: max_position_embeddings=514 → max seq len = 512
MAX_LEN = 256  # Use 256 for PhoBERT, 512 for ViSoBERT
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200

# --- TRAINING ---
TRAIN_SPLIT = 0.8
USE_MIXED_PRECISION = True
NUM_WORKERS = 4
PIN_MEMORY = True

# --- MODEL ARCHITECTURE ---
STYLE_VECTOR_DIM = 10
NUM_LABELS = 2
HIDDEN_DIM = 256
DROPOUT = 0.3

# --- AUGMENTATION DEFAULTS ---
AUGMENTATION_CONFIG = {
    "p_style_drop": 0.1,
    "p_bm25_drop": 0.1,
    "p_white_space": 0.1,
    "p_punctuation_noise": 0.2,
    "p_case_noise": 0.15,
    "p_teencode": 0.3,
    "p_accent_drop": 0.2,
    "p_features_drop": 0.1,
}

# --- DEVICE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
style_vector = [
    # --- EMOTION & FORM ---
    sentiment_score,       # (1) Is it angry/fearful?
    subjectivity_score,    # (2) Is it opinionated?
    
    # --- COMPLEXITY ---
    word_count_normalized, # (3) Is it too short/long?
    lexical_diversity,     # (4) Is the vocabulary repetitive?
    mean_idf,              # (5) Is the vocabulary generic or specific?

    # --- INFORMALITY SIGNALS ---
    caps_ratio,            # (6) Screaming?
    punct_ratio,           # (7) Sensationalist?
    pronoun_density,       # (8) Chatty/Personal?
    typo_density,          # (9) Hasty/Teencode?

    # --- VERIFICATION ---
    max_bm25_score         # (10) Does evidence exist?
]
"""