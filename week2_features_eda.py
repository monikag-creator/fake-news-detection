"""
Week 2: Feature Engineering (Bag-of-Words, TF-IDF) + Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import time

t0 = time.time()
plt.rcParams.update({"figure.dpi": 130})

data = pd.read_csv("processed_data.csv")
data["clean_text_final"] = data["clean_text_final"].fillna("")

# ---------------------------------------------------------
# EDA 1: Class balance
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
counts = data["label"].map({0: "Fake", 1: "Real"}).value_counts()
ax.bar(counts.index, counts.values, color=["#e74c3c", "#2ecc71"])
ax.set_title("Class Distribution")
ax.set_ylabel("Number of Articles")
for i, v in enumerate(counts.values):
    ax.text(i, v + 200, str(v), ha="center")
plt.tight_layout()
plt.savefig("outputs/eda_class_balance.png")
plt.close()

# ---------------------------------------------------------
# EDA 2: Article length distribution by class
# ---------------------------------------------------------
data["word_count"] = data["clean_text_final"].apply(lambda x: len(x.split()))

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(data.loc[data.label == 0, "word_count"], bins=50, alpha=0.6, label="Fake", color="#e74c3c")
ax.hist(data.loc[data.label == 1, "word_count"], bins=50, alpha=0.6, label="Real", color="#2ecc71")
ax.set_xlim(0, 1000)
ax.set_xlabel("Word Count (after cleaning)")
ax.set_ylabel("Number of Articles")
ax.set_title("Article Length Distribution")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda_word_count.png")
plt.close()

print("Word count stats:")
print(data.groupby("label")["word_count"].describe()[["mean", "50%", "std"]])

# ---------------------------------------------------------
# EDA 3: Top 15 most frequent words per class
# ---------------------------------------------------------
def top_words(subset, n=15):
    c = Counter()
    for row in subset:
        c.update(row.split())
    return c.most_common(n)

fake_top = top_words(data.loc[data.label == 0, "clean_text_final"])
real_top = top_words(data.loc[data.label == 1, "clean_text_final"])

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, top, title, color in [
    (axes[0], fake_top, "Top Words — Fake News", "#e74c3c"),
    (axes[1], real_top, "Top Words — Real News", "#2ecc71"),
]:
    words, freqs = zip(*top)
    ax.barh(words[::-1], freqs[::-1], color=color)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("outputs/eda_top_words.png")
plt.close()

print("\nTop 10 fake-news words:", fake_top[:10])
print("Top 10 real-news words:", real_top[:10])

# ---------------------------------------------------------
# Feature Engineering: Bag-of-Words
# ---------------------------------------------------------
bow_vectorizer = CountVectorizer(max_features=5000)
X_bow = bow_vectorizer.fit_transform(data["clean_text_final"])
print(f"\nBag-of-Words matrix shape: {X_bow.shape}")

# ---------------------------------------------------------
# Feature Engineering: TF-IDF (primary feature set used for modeling)
# ---------------------------------------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X_tfidf = tfidf_vectorizer.fit_transform(data["clean_text_final"])
print(f"TF-IDF matrix shape: {X_tfidf.shape}")

# Save artifacts for Week 3
import scipy.sparse as sp
sp.save_npz("X_tfidf.npz", X_tfidf)
data[["label"]].to_csv("y_labels.csv", index=False)

import pickle
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print(f"\nSaved TF-IDF features + vectorizer | time: {time.time()-t0:.1f}s")
