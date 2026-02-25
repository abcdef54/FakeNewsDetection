import random
import numpy as np
import re
from typing import Tuple,  List

# --- CONSTANTS ---

# 1. Accent Map (Vietnamese vowels -> Latin base)
UNIDECODE_MAP = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'đ': 'd',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y'
}

# 2. Punctuation List (MUST BE A LIST, NOT A SET)
PUNCTUATION_LIST = ["!", "?", ".", "..."]

# 3. Reverse Teencode (Standard -> Slang)
REVERSE_TEENCODE = {
    # không
    "không": ["ko", "k", "kh", "hok", "hk", "hông", "hong"],

    # được
    "được": ["dc", "đc", "ok"],

    # người / mọi người
    "người": ["ng", "nguoi", "ngừi"],
    "mọi người": ["mn"],

    # gì / gì vậy
    "gì": ["j", "ji"],
    "gì vậy": ["jz"],

    # như thế nào
    "như thế nào": ["ntn"],

    # địa danh
    "hà nội": ["hn"],
    "sài gòn": ["sg"],
    "thành phố hồ chí minh": ["tphcm"],
    "việt nam": ["vn"],

    # biết
    "biết": ["bt", "bít", "biet"],

    # thích / yêu
    "thích": ["thik", "thix", "thic", "thík"],
    "yêu": ["iu"],

    # từ đệm / phản hồi
    "ừ": ["uk", "uh"],
    "ừm": ["uhm"],
    "ok": ["oke"],

    # rồi
    "rồi": ["r", "rùi"],

    # với / vậy
    "với": ["vs"],
    "vậy": ["v", "z", "zậy"],

    # quá
    "quá": ["wa", "qá", "qa"],

    # slang mạng xã hội
    "hỏng": ["toang"],
    "ghen tị": ["gato"],
    "chờ xem": ["hóng"],
    "bê bối": ["phốt"],

    # gia đình
    "gia đình": ["gd", "gđ"],

    # trợ từ
    "à": ["ak", "ah", "àk"],
    "á" : ["ák"],

    # pronouns (giữ lại từ bản gốc của bạn)
    "em": ["e"],
    "anh": ["a"],
    "anh em": ["ae"],

    # time (giữ lại)
    "giờ": ["h"],
}

class TextAugmentations:
    """
    A data augmentation pipeline designed to improve the robustness of Vietnamese fake news detection models.

    This class applies stochastic perturbations to both the input text and the auxiliary feature vectors
    (style vector and BM25 score). It is specifically tailored for Vietnamese social media content by 
    simulating common informal writing habits, such as "Teencode" (slang), missing accents (diacritics), 
    typos, and irregular casing.

    Additionally, it performs feature-level dropout (masking style vectors or RAG scores) to prevent 
    the model from over-relying on any single signal during training.

    Args:
        p_style_drop (float): Probability (0.0 to 1.0) of completely zeroing out the 
            style_vector. Simulates cases where style features are unavailable or unreliable.
        p_bm25_drop (float): Probability (0.0 to 1.0) of setting the BM25 score to 0.0. 
            Simulates RAG retrieval failures (i.e., when no evidence is found).
        p_white_space (float): Probability of injecting random extra whitespace between tokens 
            to simulate formatting errors.
        p_punctuation_noise (float): Probability of modifying punctuation (e.g., simplifying 
            "!!!" to "!" or appending random punctuation to words).
        p_case_noise (float): Probability of randomly changing the casing of tokens 
            (lowercase, UPPERCASE, or Title Case).
        p_teencode (float): Probability of replacing canonical Vietnamese words with their 
            "Teencode" (slang) variants (e.g., "không" -> "ko", "được" -> "dc").
        p_accent_drop (float): Probability of removing Vietnamese tone marks/diacritics 
            from words (e.g., "việt nam" -> "viet nam").
        p_features_drop (float): Probability of applying a binary mask to the style_vector, 
            randomly zeroing out ~20% of its dimensions.

    Methods:
        apply(news_text, style_vector, bm25_score):
            Applies the configured augmentations to the inputs and returns the modified versions.
    """
    def __init__(self,
                 p_style_drop: float = 0.0,
                 p_bm25_drop: float = 0.0,
                 p_white_space: float = 0.0,
                 p_punctuation_noise: float = 0.0,
                 p_case_noise: float = 0.0,
                 p_teencode: float = 0.0,
                 p_accent_drop: float = 0.0,
                 p_features_drop: float = 0.0) -> None:
        
        self.p_style_drop = p_style_drop
        self.p_bm25_drop = p_bm25_drop
        self.p_white_space = p_white_space
        self.p_punctuation_noise = p_punctuation_noise
        self.p_case_noise = p_case_noise
        self.p_teencode = p_teencode
        self.p_accent_drop = p_accent_drop
        self.p_features_drop = p_features_drop

    def apply(self, news_text: str, style_vector: np.ndarray, bm25_score: float) -> Tuple[str, np.ndarray, float]:
        tokens = news_text.split()

        # 1. Global Vector Dropouts
        if self.p_style_drop and random.random() < self.p_style_drop:
            style_vector = np.zeros_like(style_vector, dtype=np.float32)

        if self.p_bm25_drop and random.random() < self.p_bm25_drop:
            bm25_score = 0.0

        # 2. Text Noises
        if self.p_punctuation_noise and random.random() < self.p_punctuation_noise:
            tokens = self.add_punctuation_noise(tokens)

        if self.p_case_noise and random.random() < self.p_case_noise:
            tokens = self.add_case_noise(tokens)

        if self.p_teencode and random.random() < self.p_teencode:
            tokens = self.add_random_teencode(tokens)

        if self.p_accent_drop and random.random() < self.p_accent_drop:
            tokens = self.add_random_accent_drop(tokens)
        
        # We handle White Space noise LAST because it affects how we join the tokens.
        if self.p_white_space and random.random() < self.p_white_space:
            tokens = self.add_white_space(tokens)

        # 3. Feature-level Dropout (Partial Vector Zeroing)
        if self.p_features_drop and random.random() < self.p_features_drop:
            style_vector = self.drop_random_features(style_vector)

        final_text  = " ".join(tokens)

        return final_text, style_vector, bm25_score

    def add_white_space(self, tokens: List[str], p_insert_per_gap: float = 0.1) -> List[str]:
        """Randomly adds 1-3 extra spaces between tokens."""
        out = []
        for i, p in enumerate(tokens):
            out.append(p)
            if i < len(tokens) - 1 and random.random() < p_insert_per_gap: 
                out.append(" " * random.randint(1, 3))
        return out

    def add_case_noise(self, tokens: List[str], p: float = 0.15) -> List[str]:
        out = []
        for tok in tokens:
            # Skip numbers/punctuation
            if re.fullmatch(r"\d+|[^\w\s]+", tok):
                out.append(tok)
                continue
            
            if random.random() < p:
                mode = random.choice(["lower", "upper", "title"])
                if mode == "lower": out.append(tok.lower())
                elif mode == "upper": out.append(tok.upper())
                else: out.append(tok.title())
            else:
                out.append(tok)
        return out
    
    def add_punctuation_noise(self, tokens: List[str], p_token_modify: float = 0.15) -> List[str]:
        """Adds random punctuation to end of words or simplifies existing punctuation."""
        outs = []

        for tok in tokens:
            # Case 1: Token is already punctuation (e.g., "!!!")
            if re.fullmatch(r"[^\w\s]+", tok):
                # 50% chance to simplify "!!!" -> "!"
                if len(tok) > 1 and random.random() < 0.5:
                    outs.append(tok[0])
                else:
                    outs.append(tok)
                continue
            
            # Case 2: Token is a word -> Append random punctuation
            if random.random() < p_token_modify and not re.fullmatch(r"\d+", tok):
                # Corrected: Use PUNCTUATION_LIST
                tok += random.choice(PUNCTUATION_LIST)
            outs.append(tok)

        return outs

    def add_random_teencode(self, tokens: List[str], p_teencode_convert: float = 0.2) -> List[str]:
        """
        Replace canonical phrases (including multi-word) with a random teencode variant.
        tokens: token list (punctuation preserved as separate tokens)
        """
        # Precompute phrase lengths (number of tokens) in reverse_map
        phrase_to_variants = {}
        max_phrase_len = 1
        for phrase, variants in REVERSE_TEENCODE.items():
            # ensure variants is a list
            if isinstance(variants, str):
                variants = [variants]
            phrase_to_variants[tuple(phrase.split())] = variants
            max_phrase_len = max(max_phrase_len, len(phrase.split()))

        out = []
        i = 0
        n = len(tokens)

        # normalized token keys for matching (lower, stripped of punctuation)
        norm_tokens = [t.lower() for t in tokens]

        while i < n:
            matched = False
            # Try longest phrase first
            for L in range(min(max_phrase_len, n - i), 0, -1):
                # build candidate phrase skipping punctuation tokens inside phrase
                candidate_tokens = []
                token_indices = []
                j = i
                while len(candidate_tokens) < L and j < n:
                    if not re.fullmatch(r"[^\w\s]+", tokens[j], flags=re.UNICODE):
                        candidate_tokens.append(norm_tokens[j])
                        token_indices.append(j)
                    else:
                        # treat punctuation as boundary; break phrase attempt
                        break
                    j += 1
                if len(candidate_tokens) != L:
                    continue
                key = tuple(candidate_tokens)
                if key in phrase_to_variants and random.random() < p_teencode_convert:
                    variant = random.choice(phrase_to_variants[key])
                    out.append(variant)
                    # advance i by number of tokens consumed (token_indices length)
                    i = token_indices[-1] + 1
                    matched = True
                    break
            if not matched:
                out.append(tokens[i])
                i += 1

        return out
    
    def add_random_accent_drop(self, tokens: List[str], p_drop_accent: float = 0.2) -> List[str]:
        """Removes tone marks from words."""
        out = []
        for tok in tokens:
            if random.random() < p_drop_accent:
                # Map characters to base Latin
                new_tok = "".join(UNIDECODE_MAP.get(c, c) for c in tok)
                out.append(new_tok)
            else:
                out.append(tok)
        return out

    def drop_random_features(self, style_vector: np.ndarray) -> np.ndarray:
        """Randomly masks parts of the 10-dim style vector."""
        new_vec = style_vector.copy()
        dropout_mask = np.random.choice([0, 1], size=new_vec.shape, p=[0.2, 0.8])
        return new_vec * dropout_mask