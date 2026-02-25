import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import preprocessing
from features import TextStyleExtractor
from augmentations import TextAugmentations
from rag_utils import RAGSearch
from typing import List, Dict
from pathlib import Path
import json
import os
from underthesea import word_tokenize
import numpy as np
import pandas as pd 

class FakeNewsDetectionDataset(Dataset):
    """
    PyTorch Dataset for Fake News Detection with hybrid features and RAG.

    This dataset manages the end-to-end pipeline:
    1.  **RAG Retrieval**: Fetches evidence from the knowledge base.
    2.  **Style Extraction**: Computes handcrafted linguistic features (10 dims).
    3.  **Preprocessing**: Cleans HTML/URLs but preserves necessary signals.
    4.  **Augmentation**: Injects noise (Teencode, typos) during training for robustness.
    5.  **Tokenization**: Segments words (Vietnamese) and maps to PhoBERT IDs.

    Args:
        database_jsonl_paths (str): Path to directory containing .jsonl data files.
        rag_searcher (RAGSearch): Initialized RAG engine instance.
        style_extractor (TextStyleExtractor): Initialized feature extractor instance.
        text_augmentations (TextAugmentations, optional): Augmentation engine (usually for training split only).
        tokenizer_name (str): HuggingFace model name (default: "vinai/phobert-base-v2").
        max_len (int): Maximum token sequence length (default: 512).
    """
    def __init__(self, database_jsonl_paths: str,
                 rag_searcher: RAGSearch,
                 style_extractor: TextStyleExtractor,
                 text_augmentations: TextAugmentations = None,
                 tokenizer_name: str = "vinai/phobert-base-v2", 
                 max_len=512):
        
        super().__init__()
        self.db_path = database_jsonl_paths
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database path does not exist: {self.db_path}")
        
        self.data = self._read_db(self.db_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        self.rag = rag_searcher
        self.style_extractor =  style_extractor
        self.text_augmentations = text_augmentations

        # Evidence token cap — keeps KB text from overwhelming the input.
        # With max_len=256 this gives text ~102 tokens (40%).
        self.evidence_limit = 150

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves and processes a single sample at the given index.

        Pipeline:
        1. Fetch Raw Text & Label.
        2. RAG: Retrieve evidence & BM25 confidence scores using the injected RAG engine.
        3. Features: Compute 10-dim style vector using the injected Extractor.
        4. NLP: Clean text and segment words (using underthesea).
        5. Tokenization: Convert words to IDs directly (safest method).
        6. Truncation: Apply budget limits (Priority: Evidence > News).
        7. Padding: Pad to max_len manually with Attention Masks.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'input_ids': The token indices for the model.
                - 'attention_mask': Mask to avoid attending to padding.
                - 'style_vector': The auxiliary features.
                - 'label': The ground truth class.
        """
        item = self.data[index] # item is a dict
        text = item['text']
        label = item['label']

        # RAG (Get Evidence)
        evidence = ""
        conf = 0.0
        mean_idf = 0.0
        if self.rag:
            evidence, conf, mean_idf = self.rag(text)

        #  Feature Extraction (Style Vector)
        style_vec = self.style_extractor.get_style_vector(text, mean_idf=mean_idf, bm25_score = conf) # 10-dims vector
        style_vec = np.array(style_vec)

        # Preprocessing
        clean_t = preprocessing.clean_text(text)
        clean_e = preprocessing.clean_text(evidence)

        if self.text_augmentations:
            clean_t, style_vec, conf = self.text_augmentations.apply(clean_t, style_vec, conf)
            # If BM25 was dropped, clear evidence text
            if conf == 0.0:
                clean_e = ""

        segmented_text = word_tokenize(clean_t, format="text")
        segmented_evidence = word_tokenize(clean_e, format="text")

        ids_evidence = self.tokenizer.encode(segmented_evidence, add_special_tokens=False)
        ids_text  =  self.tokenizer.encode(segmented_text, add_special_tokens=False)
        
        if len(ids_evidence) > self.evidence_limit:
            ids_evidence = ids_evidence[:self.evidence_limit]

        num_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=True)
        remaining_space = self.max_len - len(ids_evidence) - num_special_tokens

        if len(ids_text) > remaining_space:
            ids_text = ids_text[:remaining_space]
        
        input_ids = self.tokenizer.build_inputs_with_special_tokens(ids_text, ids_evidence)
        seq_len = len(input_ids)
        pad_len = self.max_len - seq_len
        if pad_len > 0:
            # Pad with the pad_token_id (usually 1 for PhoBERT)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            # Mask: 1 for real tokens, 0 for padding
            attention_mask = [1] * seq_len + [0] * pad_len
        else:
            # Just in case (though math above prevents this)
            input_ids = input_ids[:self.max_len]
            attention_mask = [1] * self.max_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'style_vector': torch.tensor(style_vec, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }
    

    @staticmethod
    def _read_db(db_dir: str) -> List[Dict[str, str]]:
        if not os.path.exists(db_dir):
            raise FileNotFoundError(db_dir)

        results: List[Dict[str, str]] = []

        base_dir = Path(db_dir)
        files = base_dir.iterdir() # Iter the directory where the jsonl files are located
        for file in files:
            if not file.is_file() or not file.suffix == '.jsonl':
                continue
            
            with open(file, 'r', encoding='utf-8') as f: # Open the jsonl file
                for line in f: 
                    results.append(json.loads(line))

        return results
    

class FakeNewsDetectionDatasetCached(Dataset):
    def __init__(
        self,
        pickle_path: str,
        text_augmentations=None,
        tokenizer_name="vinai/phobert-base-v2",
        max_len=512,
    ):
        super().__init__()

        if not os.path.exists(pickle_path):
            raise FileNotFoundError(pickle_path)

        df = pd.read_pickle(pickle_path)


        self.data = df.to_dict("records")

        self.text_augmentations = text_augmentations
        self.max_len = max_len

        # Always init tokenizer so both training (augmented) and
        # validation (clean) paths re-tokenize with the correct model.
        # Without this, ViSoBERT validation would use cached PhoBERT tokens.
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        clean_t = item["clean_text"]
        evidence = item["clean_evidence"]
        style_vec = np.array(item["style_vector"], dtype=np.float32)
        conf = item["bm25_score"]
        label = item["label"]

        # ---------------- AUGMENT (training only) ----------------
        if self.text_augmentations:
            clean_t, style_vec, conf = self.text_augmentations.apply(
                clean_t, style_vec, conf
            )
            if conf == 0:
                evidence = ""

        # Pre-truncate evidence tokens to limit so text dominates
        evidence_limit = 150
        if evidence:
            evid_ids = self.tokenizer.encode(evidence, add_special_tokens=False)
            if len(evid_ids) > evidence_limit:
                evid_ids = evid_ids[:evidence_limit]
                evidence = self.tokenizer.decode(evid_ids, skip_special_tokens=True)

        encoded = self.tokenizer(
            clean_t,
            evidence,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "style_vector": torch.from_numpy(style_vec).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }
