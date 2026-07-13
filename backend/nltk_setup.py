import nltk


def ensure_nltk_data() -> None:
    """Ensure the NLTK 'stopwords' corpus is available (needed for prediction)."""
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
