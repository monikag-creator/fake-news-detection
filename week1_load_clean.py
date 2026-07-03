"""
Week 1: Data Loading & Cleaning
AI-Powered Fake News Detection Using Text Classification
"""
import pandas as pd
import re
import string
import time

t0 = time.time()

# ---------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print(f"Fake articles: {fake.shape[0]}")
print(f"Real articles: {true.shape[0]}")

# Label: 0 = Fake, 1 = Real
fake["label"] = 0
true["label"] = 1

# ---------------------------------------------------------
# 2. Merge title + text into a single text field, combine both classes
# ---------------------------------------------------------
# NOTE: We deliberately drop 'subject' and 'date' columns before modeling.
# In this raw dataset, 'subject' almost perfectly separates real vs fake
# (fake articles use tags like 'News'/'politics'/'left-news', real articles
# use 'politicsNews'/'worldnews'). Keeping it would let the model "cheat"
# using a metadata artifact instead of learning actual textual patterns of
# misinformation, so only title+text are used as model input.
fake["content"] = fake["title"].fillna("") + " " + fake["text"].fillna("")
true["content"] = true["title"].fillna("") + " " + true["text"].fillna("")

data = pd.concat(
    [fake[["content", "label"]], true[["content", "label"]]],
    axis=0,
    ignore_index=True,
)

# Drop empty/near-empty articles
data = data[data["content"].str.strip().str.len() > 20].reset_index(drop=True)

# Shuffle (dataset is currently ordered fake-then-real, which would bias EDA/splits)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTotal combined articles: {data.shape[0]}")
print(data["label"].value_counts().rename({0: "Fake", 1: "Real"}))

# ---------------------------------------------------------
# 3. Manual text cleaning: lowercase, remove punctuation/digits/URLs
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep letters only
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text

data["clean_text"] = data["content"].apply(clean_text)

# ---------------------------------------------------------
# 4. Manual tokenization + manual stopword removal
#    (small hand-built stopword list, no external NLTK download needed)
# ---------------------------------------------------------
STOPWORDS = set("""
a an the and or but if while is are was were be been being to of in on for
with as by at from this that these those it its it's i you he she we they
them his her our your their not no nor so than too very can will just don
should now also said says say reuters get one would could may has have had
who what which about all out after before more most such only into over
than then there here when where why how any other because
""".split())

def tokenize(text: str):
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

data["tokens"] = data["clean_text"].apply(tokenize)
data["clean_text_final"] = data["tokens"].apply(lambda toks: " ".join(toks))

# Drop rows that became empty after cleaning
data = data[data["clean_text_final"].str.len() > 0].reset_index(drop=True)

print(f"\nRows after cleaning/tokenizing: {data.shape[0]}")
print("\nSample cleaned record:")
print(data.loc[0, "clean_text_final"][:300])

# ---------------------------------------------------------
# 5. Save processed data for Week 2 onward
# ---------------------------------------------------------
data[["clean_text_final", "label"]].to_csv("processed_data.csv", index=False)
print(f"\nSaved processed_data.csv  |  time: {time.time()-t0:.1f}s")
