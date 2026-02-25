import json
import os
import re
import pickle
import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss

from rank_bm25 import BM25L
from sentence_transformers import SentenceTransformer
from underthesea import word_tokenize, text_normalize

from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm

class RAGSearch:

    @staticmethod
    def _get_doc_text(doc: Dict) -> str:
        """Return the text content of a document, supporting both 'text' and 'content' fields."""
        return doc.get("text", doc.get("content", ""))

    def __init__(self, database_jsonl_paths: str, cache_path: str = "rag_cache", 
                 workers: int = -1, faiss_k: int = 1000) -> None:
        """
        Initialize the RAGSearch engine.

        This constructor:
        - Loads a pre-built BM25 index, documents, and normalized texts from a cache file if it exists.
        - Otherwise reads JSONL files from the given directory, normalizes and
        tokenizes document texts in parallel, builds a BM25L index, and saves
        all artifacts to a cache file for future reuse.

        Parameters
        ----------
        database_jsonl_paths : str
            Path to a directory containing JSONL files. Each line in each file
            must be a valid JSON object with a "text" or "content" field and
            optionally metadata such as "date".

        cache_path : str, optional
            Path to a directory used to cache the BM25 index, FAISS index,
            and loaded documents. Default is "rag_cache".
        """
        self.faiss_k = faiss_k
        os.makedirs(cache_path, exist_ok=True)

        self.pkl_path = os.path.join(cache_path, "rag.pkl")
        self.faiss_path = os.path.join(cache_path, "index.faiss")

        print("----------- Initializing Hybrid RAG -----------")

        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cuda"
        )
        self.embedder.max_seq_length = 256

        # ================= LOAD CACHE =================

        if os.path.exists(self.pkl_path) and os.path.exists(self.faiss_path):
            print("Loading cached RAG...")

            with open(self.pkl_path, "rb") as f:
                data = pickle.load(f)

            self.documents = data["documents"]
            self.bm25 = data["bm25"]
            self.max_idf = data["max_idf"]

            self.faiss_index = faiss.read_index(self.faiss_path)

            print("Cache loaded.")
            return

        # ================= BUILD =================

        print("Reading database...")
        self.documents = self._read_db(database_jsonl_paths)

        print("Normalizing texts...")
        with tqdm_joblib(tqdm(total=len(self.documents))):
            self.texts = Parallel(n_jobs=workers)(
                delayed(text_normalize)(self._get_doc_text(d))
                for d in self.documents
            )

        print("Tokenizing...")
        with tqdm_joblib(tqdm(total=len(self.texts))):
            self.tokens = Parallel(n_jobs=workers)(
                delayed(word_tokenize)(t)
                for t in self.texts
            )

        print("Building BM25...")
        self.bm25 = BM25L(self.tokens)
        self.max_idf = max(self.bm25.idf.values())

        # ================= FAISS =================

        print("Embedding corpus...")
        embeddings = self.embedder.encode(
            self.texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")
        nlist = int(np.sqrt(len(embeddings)))
        nlist = max(128, min(nlist, 8192))
        dim = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dim)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        self.faiss_index.train(embeddings)
        self.faiss_index.add(embeddings)
        self.faiss_index.nprobe = min(16, nlist)

        del self.texts
        del embeddings
        del self.tokens

        print("Saving cache...")

        with open(self.pkl_path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "bm25": self.bm25,
                    "max_idf": self.max_idf,
                },
                f,
                pickle.HIGHEST_PROTOCOL,
            )

        faiss.write_index(self.faiss_index, self.faiss_path)

        print("----------- RAG Ready -----------")

    
    def _search(self, query: str, top_k=1):

        query_norm = text_normalize(query)
        q_tokens = word_tokenize(query_norm)

        if not q_tokens:
            return (-1, 0.0, "") if top_k == 1 else []

        q_emb = self.embedder.encode(
            [query_norm],
            normalize_embeddings=True
        ).astype("float32")

        _, I = self.faiss_index.search(q_emb, self.faiss_k)
        candidates = I[0]

        scores = np.asarray(self.bm25.get_batch_scores(q_tokens, candidates))

        query_idfs = [self.bm25.idf.get(t, 0) for t in q_tokens]
        total_idf = sum(query_idfs)
        max_possible = total_idf * 2.5 if total_idf > 0 else 1.0

        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        results = []
        for i in idx:
            real_i = int(candidates[i])
            conf = float(np.clip(scores[i] / max_possible, 0, 1))
            results.append((real_i, conf, self._get_doc_text(self.documents[real_i])))

        return results[0] if top_k == 1 else results
        

    def get_evidence(self, query_text: str, max_year: int = 3) -> Tuple[str, float]:
        """
        Return Evidence Text or Empty String ("") if irrelevant.
        Parameters:
            oldest: Any evidence text that are older than n years are discarded
        """
        index, conf, text = self._search(query_text, top_k=1)
        if index < 0:
            return "", 0.0
        from_json = self.documents[index]

        try:
            evidence_date = self._str_to_date(from_json["date"])
        except:
            evidence_date = None
        query_date = self._extract_date(query_text)

        if evidence_date is None or query_date is None:
            valid_date = True  # fallback: allow evidence
        else:
            valid_date = abs(evidence_date - query_date).days <= 365 * max_year

        return (text_normalize(self._get_doc_text(from_json)), conf) if valid_date else ("", 0.0)
        

    def get_mean_idf(self, query_text: str) -> float:
        """
        Calculates the average 'rarity' of words in the query.
        Higher = More specific/technical content.
        """
        if not query_text: return 0.0
        tokenized_query = word_tokenize(text_normalize(query_text))
        idf_scores = [self.bm25.idf[t] for t in tokenized_query if t in self.bm25.idf]
        if not idf_scores: return 0.0

        mean_val = float(np.mean(idf_scores))
        return float(np.clip(mean_val / self.max_idf, 0.0, 1.0))
        


    def __call__(self, query_text: str) -> Tuple[str, float, float]:
        """
        Returns:
            text: retrieved evidence ("" if invalid)
            conf: BM25 score
            mean_idf: query specificity score
        """
        text, conf = self.get_evidence(query_text)
        mean_idf = self.get_mean_idf(query_text)
        return text, conf, mean_idf

    
    @staticmethod
    def _read_db(db_dir: str) -> List[Dict[str, str]]:
        if not os.path.exists(db_dir):
            raise FileNotFoundError(db_dir)

        results: List[Dict[str, str]] = []

        base_dir = Path(db_dir)
        files = base_dir.iterdir() # Iter the directory where the jsonl files are located
        for file in files:
            if not file.is_file():
                continue
            
            with open(file, 'r', encoding='utf-8') as f: # Open the jsonl file
                for line in f: 
                    results.append(json.loads(line))

        return results
    

    @staticmethod
    def _extract_date(text: str) -> datetime.date|None:
        """
        Return the nearest date to today found in the text.
        Supports:
        - dd/mm/yyyy
        - dd-mm-yyyy
        - yyyy
        - Vietnamese formats:
            "ngày 15 tháng 10 năm 2023"
            "15 tháng 10 năm 2023"
            "tháng 10 năm 2023"
            "năm 2023"
        """

        today = datetime.date.today()
        date_objs = []

        numeric_patterns = [
            r"\b\d{2}/\d{2}/\d{4}\b",
            r"\b\d{2}-\d{2}-\d{4}\b",
            r"\b(19\d{2}|20\d{2})\b"
        ]

        for pattern in numeric_patterns:
            for match in re.findall(pattern, text):
                try:
                    if "/" in match:
                        d, m, y = map(int, match.split("/"))
                        date_objs.append(datetime.date(y, m, d))
                    elif "-" in match:
                        d, m, y = map(int, match.split("-"))
                        date_objs.append(datetime.date(y, m, d))
                    else:
                        y = int(match)
                        date_objs.append(datetime.date(y, 12, 31))
                except ValueError:
                    continue

        # Handle Vietnamese text date 
        vn_full_date = re.findall(
            r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
            text,
            flags=re.IGNORECASE
        )

        for d, m, y in vn_full_date:
            try:
                date_objs.append(datetime.date(int(y), int(m), int(d)))
            except ValueError:
                continue

        # month + year text (frequently appear in vietnamese news)
        vn_month_year = re.findall(
            r"tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
            text,
            flags=re.IGNORECASE
        )

        for m, y in vn_month_year:
            try:
                date_objs.append(datetime.date(int(y), int(m), 1))
            except ValueError:
                continue

        # year only text
        vn_year = re.findall(
            r"năm\s*(\d{4})",
            text,
            flags=re.IGNORECASE
        )

        for y in vn_year:
            try:
                date_objs.append(datetime.date(int(y), 12, 31))
            except ValueError:
                continue

        if not date_objs:
            return None


        return min(date_objs, key=lambda d: abs(d - today))
    

    @staticmethod
    def _str_to_date(text: str) -> datetime.date|None:
        """
        Convert a date string to datetime.date.
        Returns None if no valid date is found.
        """

        text = text.lower().strip()

        # text full date
        m = re.search(
            r"ngày\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
            text
        )
        if m:
            try:
                d, mth, y = map(int, m.groups())
                return datetime.date(y, mth, d)
            except ValueError:
                return None

        # text day month year
        m = re.search(
            r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
            text
        )
        if m:
            try:
                d, mth, y = map(int, m.groups())
                return datetime.date(y, mth, d)
            except ValueError:
                return None

        # text month year 
        m = re.search(
            r"tháng\s*(\d{1,2})\s*năm\s*(\d{4})",
            text
        )
        if m:
            try:
                mth, y = map(int, m.groups())
                return datetime.date(y, mth, 1)
            except ValueError:
                return None

        # text year 
        m = re.search(r"năm\s*(\d{4})", text)
        if m:
            try:
                return datetime.date(int(m.group(1)), 12, 31)
            except ValueError:
                return None


        numeric_formats = [
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y/%m/%d",
            "%Y-%m-%d",
            "%Y",
        ]

        for fmt in numeric_formats:
            try:
                dt = datetime.datetime.strptime(text, fmt)
                if fmt == "%Y":
                    return datetime.date(dt.year, 12, 31)
                return dt.date()
            except ValueError:
                continue

        return None
