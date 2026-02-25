import re
import unicodedata
import html

TEENCODE_DICT = {
    "ko": "không", "k": "không", "kh": "không", "hok": "không",
    "hk": "không",

    "dc": "được", "đc": "được",

    "ng": "người", "nguoi": "người", "ngừi": "người",
    "mn": "mọi người",

    "j": "gì", "ji": "gì", "jz": "gì vậy",

    "ntn": "như thế nào",

    "hn": "hà nội", "sg": "sài gòn", "tphcm": "thành phố hồ chí minh", "vn": "việt nam",

    "bt": "biết", "bít": "biết", "biet": "biết",

    "thik": "thích", "thix": "thích", "thic": "thích", "thík": "thích",
    "iu": "yêu",

    "uk": "ừ", "uhm": "ừm", "uh": "ừ", "oke": "ok",

    "r": "rồi", "rùi": "rồi",

    "vs": "với", "v": "vậy", "z": "vậy", "zậy": "vậy",

    "wa": "quá", "qá": "quá", "qa": "quá",

    "toang": "hỏng",
    "gato": "ghen tị",
    "hóng": "chờ xem",
    "phốt": "bê bối",
    "phot": "bê bối",

    "gd": "gia đình", "gđ": "gia đình",
    "ak": "à", "ah": "à", "ạ": "ạ",
}

# match words (letters/digits/underscore) OR sequences of punctuation
TOKEN_RE = re.compile(r"\w+|[^\w\s]+", re.UNICODE)

# collapse long repeated characters: keep at most two repeats
_REPEAT_RE = re.compile(r"(.)\1{2,}", flags=re.UNICODE)

def _collapse_repeats(token: str) -> str:
    """Collapse very long repeats: 'đẹpppp' -> 'đẹp' (keeps two repeats)."""
    return _REPEAT_RE.sub(r"\1\1", token)

def normalize_teencode(text: str) -> str:
    """
    Replaces slang words with canonical Vietnamese tokens while preserving punctuation.
    - tokenizes by whitespace/punctuation,
    - collapses extreme repeated chars,
    - looks up lowercase token in TEENCODE_DICT,
    - reconstructs text with sensible spacing around punctuation.
    """
    if not text:
        return ""

    # split into tokens (words or punctuation groups)
    tokens = TOKEN_RE.findall(text)
    out_tokens = []
    for tok in tokens:
        # collapse repeated chars (social noise)
        tok2 = _collapse_repeats(tok)
        key = tok2.lower()
        if key in TEENCODE_DICT:
            mapped = TEENCODE_DICT[key]
            out_tokens.append(mapped)
        else:
            out_tokens.append(tok2)

    # reconstruct: put a space between two tokens unless the current token is punctuation
    # or previous token was punctuation (keeps punctuation glued to word).
    reconstructed = []
    for i, t in enumerate(out_tokens):
        is_punct = bool(re.match(r'^[^\w\s]+$', t, flags=re.UNICODE))
        if i == 0:
            reconstructed.append(t)
        else:
            prev_is_punct = bool(re.match(r'^[^\w\s]+$', out_tokens[i-1], flags=re.UNICODE))
            if is_punct or prev_is_punct:
                reconstructed.append(t)
            else:
                reconstructed.append(" " + t)

    return "".join(reconstructed).strip()


def clean_text(text: str) -> str:
    """
    Standardize text for BERT input:
      1. NFC unicode normalization
      2. HTML unescape + remove tags
      3. remove URLs & emails
      4. remove zero-width / NBSP
      5. collapse repeats, normalize teencode
      6. whitespace normalization
    """
    if not text:
        return ""

    # 1. Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # 2. Decode HTML entities (&amp; -> &)
    text = html.unescape(text)

    # 3. Remove HTML Tags
    text = re.sub(r'<[^>]*>', ' ', text)

    # 4. Remove URLs and Emails
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)

    # 5. Remove "Zombie" characters
    text = text.replace('\u200b', '').replace('\xa0', ' ')

    # 6. Normalize whitespace a first pass
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Collapse extreme repeats and normalize teencode
    # (normalize_teencode also collapses repeats per token)
    text = normalize_teencode(text)

    # 8. Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    return text
