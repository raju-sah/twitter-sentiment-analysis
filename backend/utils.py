import re
import string
from typing import List

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet: str) -> List[str]:
    """Clean, tokenize, remove stopwords, and stem a tweet."""
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words("english")
    tweet = re.sub(r"\$\w*", "", tweet)
    tweet = re.sub(r"^RT[\s]+", "", tweet)
    tweet = re.sub(r"https?://[^\s\n\r]+", "", tweet)
    tweet = re.sub(r"#", "", tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            tweets_clean.append(stemmer.stem(word))
    return tweets_clean


def build_freqs(tweets, ys):
    """Build a dict mapping (word, label) -> count."""
    yslist = np.squeeze(ys).tolist()
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            freqs[pair] = freqs.get(pair, 0) + 1
    return freqs


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def extract_features(tweet: str, freqs) -> np.ndarray:
    """Return a (1, 3) feature vector: [bias, positive_count, negative_count]."""
    word_l = process_tweet(tweet)
    x = np.zeros((1, 3))
    x[0, 0] = 1
    for word in word_l:
        x[0, 1] += freqs.get((word, 1.0), 0)
        x[0, 2] += freqs.get((word, 0.0), 0)
    return x


def gradient_descent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for _ in range(num_iters):
        z = x.dot(theta)
        h = sigmoid(z)
        theta = theta - (alpha / m) * x.T.dot(h - y)
    return theta


def predict_tweet(tweet: str, freqs, theta) -> float:
    """Return the probability (0-1) that the tweet is positive."""
    x = extract_features(tweet, freqs)
    y_pred = sigmoid(x.dot(theta))
    return float(np.squeeze(y_pred))
