"""Train the logistic-regression sentiment model and persist it as model.pkl.

Run once locally (or in CI) after changing the training procedure:
    python train.py

The resulting model.pkl contains the trained weights (theta) and the word
frequency dictionary (freqs) built from the NLTK twitter_samples corpus.
"""
import pickle

import numpy as np
from nltk.corpus import twitter_samples

from utils import (
    build_freqs,
    extract_features,
    gradient_descent,
    predict_tweet,
)

all_positive = twitter_samples.strings("positive_tweets.json")
all_negative = twitter_samples.strings("negative_tweets.json")

train_pos = all_positive[:4000]
train_neg = all_negative[:4000]
test_pos = all_positive[4000:]
test_neg = all_negative[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)

X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)
Y = train_y

theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 1500)

with open("model.pkl", "wb") as f:
    pickle.dump({"freqs": freqs, "theta": theta}, f)

correct = 0
for tweet, y in zip(test_x, test_y):
    y_hat = 1.0 if predict_tweet(tweet, freqs, theta) > 0.5 else 0.0
    correct += int(y_hat == float(y[0]))
accuracy = correct / len(test_y)
print(f"Trained model. Test accuracy = {accuracy:.4f}")
print("Saved model.pkl")
