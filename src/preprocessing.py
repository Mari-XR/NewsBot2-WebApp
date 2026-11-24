"""English preprocessing from final notebook cell 75."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_lemmatizer = WordNetLemmatizer()



def clean_text(text):
    """Lowercase, remove HTML, URLs, emails, numbers and symbols."""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text):
    """Clean + remove stop words + lemmatize + drop very short words."""
    tokens = clean_text(text).split()
    stop_words = english_stopwords()
    tokens = [_lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(t for t in tokens if len(t) > 2)


def split_sentences(text):
    """Split an article into sentences (BBC text uses '. ' between sentences)."""
    try:
        sents = nltk.sent_tokenize(str(text))
    except LookupError:
        sents = re.split(r"(?<=[.!?])\s+", str(text))
    return [s.strip() for s in sents if len(s.split()) >= 4]

from functools import lru_cache


@lru_cache(maxsize=1)
def english_stopwords():
    return set(stopwords.words("english"))
