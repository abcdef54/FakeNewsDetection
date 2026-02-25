
---

# 📘 Fake News Detection Dataset: Usage Guide

This guide explains how to initialize and use the `FakeNewsDetectionDataset` for training and inference. This dataset handles the complex logic of **RAG Retrieval**, **Style Extraction**, and **Robustness Augmentation** automatically.

## 1. The Architecture

The dataset does not just load text; it constructs a hybrid input for the model:

* **Input 1 (Semantic):** `PhoBERT` tokens (Article + RAG Evidence).
* **Input 2 (Stylistic):** A 10-dimensional vector (Emotion, Subjectivity, Teencode density, etc.).

## 2. Quick Start Code

Copy-paste this into your `train.py` or `main.py` to get started immediately.

```python
from torch.utils.data import DataLoader
from dataset import FakeNewsDetectionDataset
from rag_utils import RAGSearch
from features import TextStyleExtractor
from augmentations import TextAugmentations

# --- CONFIGURATION ---
DATA_DIR = "./data/train_jsonl"  # Folder containing .jsonl files
RAG_CACHE = "./cache/rag_index.pkl"
BATCH_SIZE = 16

# 1. Initialize Global Components (Load Once, Reuse)
print("Loading RAG Engine...")
rag = RAGSearch(database_jsonl_paths=DATA_DIR, cache_path=RAG_CACHE)

print("Initializing Feature Extractor...")
style_ext = TextStyleExtractor()

# 2. Define Augmentation Policy (TRAINING ONLY)
# For Validation/Test, set this to None.
augmenter = TextAugmentations(
    p_style_drop=0.15,      # 15% chance to zero out style vector (robustness)
    p_bm25_drop=0.1,        # 10% chance to simulate "No Evidence Found"
    p_teencode=0.3,         # Inject slang to make model robust to social media text
    p_accent_drop=0.3,      # Drop accents (e.g., "duoc" instead of "được")
    p_white_space=0.1       # Add random typos
)

# 3. Create the Dataset
train_dataset = FakeNewsDetectionDataset(
    database_jsonl_paths=DATA_DIR,
    rag_searcher=rag,
    style_extractor=style_ext,
    text_augmentations=augmenter, # Pass None for Val/Test
    tokenizer_name="vinai/phobert-base-v2",
    max_len=512
)

# 4. Create DataLoader
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4, 
    pin_memory=True
)

# 5. Usage in Training Loop
print("Starting Training...")
for batch in train_loader:
    # Move all tensors to GPU
    input_ids = batch['input_ids'].cuda()
    attention_mask = batch['attention_mask'].cuda()
    style_vector = batch['style_vector'].cuda()
    labels = batch['label'].cuda()

    # Forward Pass
    # output = model(input_ids, attention_mask, style_vector)
    # loss = criterion(output, labels)

```

---

## 3. Detailed Component Breakdown

### A. RAG Searcher (`rag_utils.py`)

* **What it does:** Indexes your database using BM25L.
* **First Run:** It will be slow (tokenizing all articles). It saves a `.pkl` cache file.
* **Subsequent Runs:** It loads instantly from the cache.
* **Note:** If you change your source data, **delete the `.pkl` file** to force a rebuild.

### B. Style Extractor (`features.py`)

* **What it does:** Calculates 10 linguistic features (Emotion, Pronouns, etc.).
* **Inputs:** It runs on the *cleaned* text but is robust enough to handle raw signals.
* **Output:** A fixed 10-dimensional float vector, normalized to `[0, 1]`.

### C. Augmentations (`augmentations.py`)

* **Crucial Rule:** Only use this for **Training**.
* **Why?** It injects noise (bad grammar, missing accents) to force the model to learn the *content* and *style signals* rather than overfitting to specific keywords.
* **Features Dropout:** It randomly zeros out parts of the Style Vector. This prevents the model from crashing if the Style Extractor fails or gives weird values during inference.

---

## 4. The Output Batch (Tensor Shapes)

The `DataLoader` returns a dictionary with the following shapes (assuming `batch_size=16`):

| Key | Shape | Type | Description |
| --- | --- | --- | --- |
| `input_ids` | `[16, 512]` | `torch.long` | Token IDs for PhoBERT. Contains Article + Separator + Evidence + Padding. |
| `attention_mask` | `[16, 512]` | `torch.long` | `1` for real tokens, `0` for padding. |
| `style_vector` | `[16, 10]` | `torch.float` | The auxiliary features to be concatenated in the model. |
| `label` | `[16]` | `torch.long` | Binary labels (`0` or `1`). |

## 5. Common Pitfalls Checklist

1. **Missing RAG Cache:** If the `rag_cache.pkl` is missing, the first epoch startup will take 5-10 minutes. This is normal.
2. **Val Set Augmentation:** Ensure you pass `text_augmentations=None` for the Validation/Test dataset. We want to evaluate on real, clean data, not noisy augmented data.
3. **Path Errors:** `database_jsonl_paths` must be a **directory**, not a file. The code scans for all `.jsonl` files inside.