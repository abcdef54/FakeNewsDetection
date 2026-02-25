import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from transformers import AutoTokenizer

from features import TextStyleExtractor
from rag_utils import RAGSearch
from preprocessing import clean_text

# ==================================================
# Globals inside worker
# ==================================================

RAG = None
STYLE = None
TOKENIZER = None
EVIDENCE_LIMIT = None
MAX_LEN = None


# ==================================================
# Utils
# ==================================================

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _read_db(db_dir: str) -> List[Dict]:
    results = []
    base_dir = Path(db_dir)
    files = list(base_dir.glob("*.jsonl")) if base_dir.is_dir() else [base_dir]

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


# ==================================================
# Worker init
# ==================================================

def init_worker(rag_search, text_style_extractor, tokenizer_name, evidence_limit, max_len):
    global RAG, STYLE, TOKENIZER, EVIDENCE_LIMIT, MAX_LEN

    RAG = rag_search
    STYLE = text_style_extractor
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    EVIDENCE_LIMIT = evidence_limit
    MAX_LEN = max_len


# ==================================================
# Batch processing (IMPORTANT)
# ==================================================

def _process_batch(batch: List[Dict]):
    global RAG, STYLE, TOKENIZER, MAX_LEN

    outputs = []

    texts = []
    evidences = []
    metas = []

    # ---------- RAG + Style first ----------
    for news in batch:
        try:
            text_content = news["text"]

            evidence, conf, mean_idf = RAG(text_content)
            style_vector = STYLE.get_style_vector(text_content, mean_idf, conf)

            clean_t = clean_text(text_content)
            clean_e = clean_text(evidence)

            texts.append(clean_t)
            evidences.append(clean_e)

            metas.append((style_vector, conf, news["label"]))

        except Exception:
            continue

    if not texts:
        return []

    # ---------- Pre-truncate evidence to EVIDENCE_LIMIT ----------
    # Evidence from KB is real news — if it dominates the token budget,
    # the model learns "see KB patterns → predict REAL" (spurious shortcut).
    # Cap evidence to ~25 % of max_len so the input text's style signals
    # (caps, emotion words, punctuation) remain the primary signal.
    if EVIDENCE_LIMIT:
        capped_evidences = []
        for ev in evidences:
            ev_ids = TOKENIZER.encode(ev, add_special_tokens=False)
            if len(ev_ids) > EVIDENCE_LIMIT:
                ev_ids = ev_ids[:EVIDENCE_LIMIT]
                ev = TOKENIZER.decode(ev_ids, skip_special_tokens=True)
            capped_evidences.append(ev)
        evidences = capped_evidences

    # ---------- HF tokenizer batched ----------
    encoded = TOKENIZER(
        texts,
        evidences,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )

    for i in range(len(texts)):
        style_vector, conf, label = metas[i]

        outputs.append({
            "clean_text": texts[i],
            "clean_evidence": evidences[i],
            "input_ids": encoded["input_ids"][i],
            "attention_mask": encoded["attention_mask"][i],
            "style_vector": style_vector,
            "bm25_score": conf,
            "label": label,
        })

    return outputs


# ==================================================
# Main
# ==================================================

def cache_dataset(
    database_path: str,
    text_style_extractor: TextStyleExtractor,
    rag_search: RAGSearch,
    tokenizer_name: str = "vinai/phobert-base-v2",
    evidence_limit: int = 150,   # max evidence tokens per sample
    max_len: int = 512,
    cache_path: str = "cache_dataset.pkl",
    max_workers: int = 3,
    batch_size: int = 32,   # HF sweet spot
):
    if max_workers > 3:
        print("-"*50)
        print()
        print("WARNING: EACH THREAD COST ~2GB OF RAM, IF YOU SET WORKER AMOUNT TOO HIGH IT MIGHT CRASH YOUR PC")
        print()
        print("-"*50)
    print(f"Evidence token limit: {evidence_limit} / {max_len}")

    corpus = _read_db(database_path)

    print(f"Loaded {len(corpus)} samples")

    corpus_batches = list(chunks(corpus, batch_size))

    data = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(rag_search, text_style_extractor, tokenizer_name, evidence_limit, max_len),
    ) as executor:

        for batch_result in tqdm(
            executor.map(_process_batch, corpus_batches),
            total=len(corpus_batches)
        ):
            data.extend(batch_result)

    df = pd.DataFrame(data)
    df.to_pickle(cache_path)

    print("Saved:", cache_path)
    return cache_path
