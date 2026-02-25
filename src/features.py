from typing import List, Dict
import numpy as np
import string

DEFAULT_FACTORS = {
    "emotion" : 0.3,
    "pronouns" : 0.7,
    "slang" : 0.1,
    "punctuation" : 0.7,
    "subjectivity" : 0.3
}


EMOTION_WORDS = {
        # Shock/Fear
        "sốc", "kinh hoàng", "khủng khiếp", "đáng sợ", "rùng mình",
        "thảm họa", "nguy hiểm", "cảnh báo", "khẩn cấp", "chết người",
        "rúng động", "rung động", "hoảng loạn", "biến căng", "đột biến",
        
        # Anger/Outrage
        "phẫn nộ", "bức xúc", "tức giận", "lừa đảo", "trơ trẽn",
        "vô lương tâm", "dã man", "tàn độc",
        
        # Extreme Positivity (Miracle Cures/Scams)
        "thần kỳ", "kỳ diệu", "tuyệt đối", "cam kết", "khỏi hẳn",
        "duy nhất", "vĩnh viễn"
    }

SUBJECTIVE_MARKERS = {
        # First Person / Personal Opinion
        "theo tôi", "tôi nghĩ", "mình thấy", "cảm thấy",
        "tin rằng", "ngờ rằng", "đảm bảo",
        
        # Uncertainty / Rumor
        "hình như", "có vẻ", "dường như", "nghe nói", "nghe đồn", "nghi vấn", 
        
        # Certainty / Exaggeration (without data)
        "chắc chắn", "rõ ràng", "không thể phủ nhận", "bất ngờ",
        "thật sự", "quả thực"
    }

TEENCODE_SET = {
        "ko", "k", "kh", "hok", "hông", "hem", "hơm", # không
        "dc", "đc", "dk", "đk",                       # được
        "ng", "nguoi", "ngừi",                        # người
        "nt", "ntn",                                  # như thế nào
        "ms",                                         # mới
        "hn",                                         # hà nội
        "bt", "bít",                                  # biết
        "tr", "trùi", "troi",                         # trời
        "z", "zậy", "dậy",                            # vậy
        "j", "ji", "gì",                              # gì
        "jz",                                         # jz
        "thik", "thix", "thic", "thík",               # thích
        "uk", "uhm", "uh", "ừm",                      # ừ
        "rùi", "r",                                   # rồi
        "vs",                                         # với
        "wa", "qá",                                   # quá
        "toang", "gato", "hóng", "phốt",              # social slang
        "ak", "ah"                                    # à, ạ
    }


SINGLE_PRONOUNS = {"tôi", "bạn", "mày", "tao", "cậu", "tớ", "mình", "ní", "ta", "nó",
                    "anh", "em", "chú", "bác", "cô", "dì", "mợ", "ông", "bà", "hắn", "ad"}

COLLECTIVE_PRONOUNS = {"chúng tôi", "chúng ta", "chúng em", "các bạn", "các ông", "các bà",
                        "chúng tớ", "các cụ", "các cô", "các cậu", "các anh", "các em",
                        "bọn tôi", "bọn em", "bọn anh"}

class TextStyleExtractor:
    """
    Extracts a fixed-length stylistic feature vector from a Vietnamese text.

    This class is designed to capture *writing style* and *linguistic signals*
    that are commonly associated with misinformation, sensationalism, or
    informal / opinionated content. The extracted features are intended to be
    used alongside semantic embeddings (e.g., PhoBERT) in downstream models
    such as fake news detection or credibility classification.

    The extractor focuses on **how the text is written**, not what it says.
    All features are normalized to the range [0, 1] to allow stable fusion
    with neural embeddings.

    Feature categories
    ------------------
    1. Emotion & Subjectivity
       - Emotional intensity based on predefined emotional keywords.
       - Subjectivity proxy based on opinionated or speculative phrases.

    2. Lexical & Structural Complexity
       - Normalized word count.
       - Lexical diversity (unique word usage).
       - Mean IDF score of query terms (external signal).

    3. Informality & Sensationalism Signals
       - Uppercase letter ratio.
       - Excessive punctuation usage ('!' and '?').
       - Personal / collective pronoun density.
       - Slang / teencode / typo density.

    4. Verification Signal
       - BM25 relevance score from an external retrieval system (RAG).

    Output
    ------
    The `get_style_vector` method returns a list of 10 floating-point values
    in the following order:

        [
            emotion_intensity,
            subjectivity_score,
            word_count_normalized,
            lexical_diversity,
            mean_idf,
            caps_ratio,
            punctuation_ratio,
            pronoun_density,
            typo_slang_density,
            bm25_score
        ]

    Notes
    -----
    - This class is **stateful per call**: internal fields such as tokens and
      word count are updated every time `get_style_vector` is invoked.
    - The class assumes Vietnamese input text.
    - All keyword sets (emotion words, subjective markers, slang, pronouns)
      are rule-based and can be extended or domain-adapted.
    """
    def __init__(self):
        self.word_counts = 0
        self.text = None
        self.lower_text = None
        self.tokens = []

    def get_style_vector(self, text: str, mean_idf: float, bm25_score: float, normalize_factors: Dict[str, float] = DEFAULT_FACTORS) -> List[float]:
        """
        Master function called by Dataset.
        Returns: 10-dims style vector
        """

        self.lower_text = text.lower()
        self.tokens = [
                    w.strip(string.punctuation)
                    for w in self.lower_text.split()
                    if w.strip(string.punctuation)
                ]
        self.word_counts = len(self.tokens)
        self.text = text
        
        return [
            self._get_emotion_intensity(normalize_factors["emotion"]),
            self._get_subjectivity_score(normalize_factors["subjectivity"]),
            self._get_word_count_normalized(max_ref=512),
            self._get_lexical_diversity(),
            mean_idf,
            self._get_caps_ratios(),
            self._get_punctuation_ratio(normalize_factors["punctuation"]),
            self._get_pronouns_density(1.5, normalize_factors["pronouns"]),
            self._get_typo_slang_density(normalize_factors["slang"]),
            bm25_score
        ]

    def _count_phrase_matches(self, phrases: set[str]) -> int:
        count = 0
        tokens = self.tokens # Local reference for speed
        n_tokens = len(tokens)

        for phrase in phrases:
            p_tokens = phrase.split()
            len_p = len(p_tokens)
            
            # Sliding window check
            # This handles "không thể phủ nhận" correctly by matching [token, token, token]
            for i in range(n_tokens - len_p + 1):
                if tokens[i : i + len_p] == p_tokens:
                    count += 1
        return count


    def _get_emotion_intensity(self, normalize_factor: float = 0.3) -> float:
        """Returns emotional intensity score in [0, 1]."""
        if self.word_counts <= 0:
            return 0.0
        count = self._count_phrase_matches(EMOTION_WORDS)

        ratio = count / self.word_counts
        ratio = ratio / (ratio + normalize_factor)
        return ratio


    def _get_caps_ratios(self) -> float:
        """Returns ratio of uppercase letters to total letters."""
        letters = [char for char in self.text if char.isalpha()]
        if not letters:
            return 0.0

        count = sum(char.isupper() for char in letters)
        return count / len(letters)


    def _get_punctuation_ratio(self, normalize_factor: float = 1.0) -> float:
        """Returns a bounded ratio of '!' and '?' divided by word count."""
        if  self.word_counts <= 0:
            return 0.0
        
        count = sum(char in {"!", "?"} for char in self.text)
        ratio = count / self.word_counts
        ratio = ratio / (ratio + normalize_factor)
        return ratio


    def _get_word_count_normalized(self, max_ref: int = 512) -> float:
        norm = np.log1p(self.word_counts) / np.log1p(max_ref)
        return np.clip(norm, 0, 1)


    def _get_lexical_diversity(self) -> float:
        if self.word_counts <= 1:
            return 0.0
        
        return np.log1p(len(set(self.tokens))) / np.log1p(self.word_counts)


    def _get_pronouns_density(self, collective_weight: float = 1.5, normalize_factor: float = 0.7) -> float:
        if self.word_counts <= 0:
            return 0.0
        
        count = 0
        i = 0
        while i < self.word_counts: # loop over all the tokens in the text
            if i < self.word_counts - 1:
                bigram = self.tokens[i] + " " + self.tokens[i+1]
                if bigram in COLLECTIVE_PRONOUNS:
                    count += collective_weight # collective pronouns have more weights (since real/formal news almost never use them)
                    i += 2
                    continue
            
            # fallback to unigram
            if self.tokens[i] in SINGLE_PRONOUNS:
                count += 1
            
            i += 1

        ratio = count / self.word_counts
        ratio = ratio / (ratio + normalize_factor)
        return ratio


    def _get_typo_slang_density(self, normalize_factor: float = 0.1) -> float:
        if self.word_counts == 0:
            return 0.0
        count = sum(word in TEENCODE_SET for word in self.tokens)

        ratio = count / self.word_counts
        ratio = ratio / (ratio + normalize_factor)
        return ratio
        

    def _get_subjectivity_score(self, normalize_factor: float = 0.3) -> float:
        if self.word_counts <= 0:
            return 0.0
        
        count = self._count_phrase_matches(SUBJECTIVE_MARKERS)
        ratio = count / self.word_counts
        ratio = ratio / (ratio + normalize_factor)
        return ratio


"""
style_vector = [
    # --- EMOTION & FORM ---
    emotion_score,         # (1) Is it angry/fearful?
    subjectivity_proxy,    # (2) Is it opinionated?
    
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