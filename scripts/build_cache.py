"""
Rebuild Dataset Cache (single-process, safe with large KB)
Processes all Organized/ samples → RAG evidence + style + tokenization → pickle
"""
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from features import TextStyleExtractor
from rag_utils import RAGSearch
from preprocessing import clean_text

# ── Config ──────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), '..', 'Organized')
RAG_CACHE     = os.path.join(os.path.dirname(__file__), '..', 'rag_cache')
CACHE_FILE    = os.path.join(os.path.dirname(__file__), '..', 'src', 'cache_dataset.pkl')
MAX_LEN       = 256
TOKENIZER_NAME = "vinai/phobert-base-v2"
BATCH_SIZE     = 64  # tokenizer batch size

# ── Load JSONL ──────────────────────────────────────────
def read_organized(data_dir: str):
    results = []
    for f in Path(data_dir).glob("*.jsonl"):
        with open(f, 'r', encoding='utf-8') as fh:
            for line in fh:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results

# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("REBUILDING DATASET CACHE (single-process)")
    print("=" * 60)

    # 1. Init RAG (loads from KB cache – 35K docs)
    print("\n[1/4] Loading RAG from KB cache...")
    rag = RAGSearch(database_jsonl_paths=DATA_DIR, cache_path=RAG_CACHE)
    print(f"  RAG documents: {len(rag.documents)}")

    # 2. Init Style + Tokenizer
    print("[2/4] Loading Style Extractor + Tokenizer...")
    style = TextStyleExtractor()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    # 3. Load corpus
    corpus = read_organized(DATA_DIR)
    print(f"[3/4] Loaded {len(corpus)} samples from {DATA_DIR}")

    # 4. Process
    print(f"[4/4] Processing samples (batch_size={BATCH_SIZE})...")

    data = []
    errors = 0

    # Process in batches for efficient tokenization
    for batch_start in tqdm(range(0, len(corpus), BATCH_SIZE), desc="Batches"):
        batch = corpus[batch_start : batch_start + BATCH_SIZE]
        
        texts = []
        evidences = []
        metas = []
        
        for news in batch:
            try:
                text_content = news["text"]
                evidence, conf, mean_idf = rag(text_content)
                sv = style.get_style_vector(text_content, mean_idf, conf)
                
                texts.append(clean_text(text_content))
                evidences.append(clean_text(evidence))
                metas.append((sv, conf, news["label"]))
            except Exception as e:
                errors += 1
                continue
        
        if not texts:
            continue
        
        # Pre-truncate evidence tokens to 150 so input text style signals
        # are not overwhelmed by KB real-news patterns.
        evidence_limit = 150
        capped_evidences = []
        for ev in evidences:
            ev_ids = tokenizer.encode(ev, add_special_tokens=False)
            if len(ev_ids) > evidence_limit:
                ev_ids = ev_ids[:evidence_limit]
                ev = tokenizer.decode(ev_ids, skip_special_tokens=True)
            capped_evidences.append(ev)
        evidences = capped_evidences
        
        # Batched tokenization
        encoded = tokenizer(
            texts, evidences,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
        )
        
        for i in range(len(texts)):
            sv, conf, label = metas[i]
            data.append({
                "clean_text": texts[i],
                "clean_evidence": evidences[i],
                "input_ids": encoded["input_ids"][i],
                "attention_mask": encoded["attention_mask"][i],
                "style_vector": sv,
                "bm25_score": conf,
                "label": label,
            })

    # Save
    df = pd.DataFrame(data)
    df.to_pickle(CACHE_FILE)

    print("\n" + "=" * 60)
    print(f"  Saved: {CACHE_FILE}")
    print(f"  Samples: {len(df)} / {len(corpus)} ({errors} errors)")
    print(f"  Labels: {dict(df['label'].value_counts().sort_index())}")
    print(f"  MAX_LEN: {MAX_LEN}")
    print("=" * 60)
