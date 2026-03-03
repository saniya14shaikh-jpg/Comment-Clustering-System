"""
Text Preprocessing Engine for Comment Clustering System
Handles: cleaning, tokenization, emoji, sarcasm detection
"""

import re
import string
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

_STOP  = set(stopwords.words("english"))
_LEMMA = WordNetLemmatizer()

# Keep negation words
KEEP_WORDS = {
    "not", "no", "never", "nor", "neither", "none",
    "nothing", "without", "don't", "doesn't", "didn't",
    "won't", "wouldn't", "can't", "couldn't"
}

# Sarcasm indicators
SARCASM_PATTERNS = [
    r"yeah right", r"sure sure", r"oh great",
    r"wow thanks", r"obviously", r"clearly",
    r"totally", r"as if", r"like that",
    r"good luck with that", r"oh really",
]

# Toxic keywords
TOXIC_KEYWORDS = [
    "idiot", "stupid", "moron", "dumb", "loser",
    "worthless", "pathetic", "ugly", "hate", "kill",
    "disgusting", "garbage", "trash", "worst", "terrible",
    "horrible", "awful", "useless", "embarrassment", "quit",
]

def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def remove_html(text):
    return re.sub(r"<[^>]+>", " ", text)

def remove_punctuation(text):
    return text.translate(
        str.maketrans(string.punctuation,
                      " " * len(string.punctuation)))

def remove_numbers(text):
    return re.sub(r"\d+", " ", text)

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def replace_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

def is_emoji_only(text):
    cleaned = text.strip()
    non_emoji = re.sub(r'[^\w\s]', '', cleaned)
    return len(non_emoji.strip()) == 0

def detect_sarcasm(text):
    text_lower = text.lower()
    for pattern in SARCASM_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    if text.isupper() and len(text) > 10:
        return True
    if text.count("!") > 3:
        return True
    return False

def detect_toxic(text):
    text_lower = text.lower()
    for word in TOXIC_KEYWORDS:
        if word in text_lower:
            return True
    return False

def preprocess(text, lemmatize=True):
    if not isinstance(text, str) or not text.strip():
        return ""

    # Extract emojis before removing
    emojis_found = extract_emojis(text)

    # Replace emojis with text
    text = replace_emojis(text)

    # Clean
    text = text.lower()
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize_whitespace(text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords (keep negations)
    tokens = [t for t in tokens
              if t not in _STOP or t in KEEP_WORDS]

    # Lemmatize
    if lemmatize:
        tokens = [_LEMMA.lemmatize(t) for t in tokens]

    return " ".join(tokens)

def preprocess_batch(texts):
    return [preprocess(t) for t in texts]

def get_text_features(text):
    return {
        "original":      text,
        "cleaned":       preprocess(text),
        "tokens":        preprocess(text).split(),
        "emojis":        extract_emojis(text),
        "is_emoji_only": is_emoji_only(text),
        "is_sarcastic":  detect_sarcasm(text),
        "is_toxic":      detect_toxic(text),
        "char_count":    len(text),
        "word_count":    len(text.split()),
        "exclamations":  text.count("!"),
        "questions":     text.count("?"),
        "caps_ratio":    sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }

if __name__ == "__main__":
    samples = [
        "This is AMAZING!!! Love it so much! ❤️",
        "You are such an idiot! Nobody likes you!",
        "Okay I guess. Nothing special.",
        "OMG OMG OMG I cannot believe this!!!",
        "Yeah right! Sure sure! Obviously perfect! 🙄",
    ]
    for s in samples:
        features = get_text_features(s)
        print(f"\nOriginal:  {s}")
        print(f"Cleaned:   {features['cleaned']}")
        print(f"Toxic:     {features['is_toxic']}")
        print(f"Sarcastic: {features['is_sarcastic']}")
        print(f"Emojis:    {features['emojis']}")
