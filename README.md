# AI-Powered Fake News Detection Using Text Classification

A machine learning pipeline that classifies news articles as **real or fake** using TF-IDF text features and four classification algorithms — built as part of the IICT AI/ML Summer Internship Program (2026).

## Problem

Fake news spreads rapidly across digital platforms and is hard to catch manually at scale. This project builds a machine learning pipeline that reads a news article's title and text, and predicts whether it's **real or fake**.

## Approach

Text is cleaned and converted into TF-IDF features, then classified using four different algorithms — K-Nearest Neighbors, Logistic Regression, Random Forest, and a Neural Network — trained on 44,889 labeled news articles from the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

## Result

Best accuracy: **98.88%** (Neural Network), with Random Forest and Logistic Regression close behind.

![Model Comparison](outputs/model_comparison.png)

## Project Structure

```
├── week1_load_clean.py       # Data loading, merging, cleaning & tokenization
├── week2_features_eda.py     # TF-IDF feature extraction + EDA
├── week3_models.py           # Trains KNN, Logistic Regression, Random Forest, Neural Net
├── week4_evaluate.py         # Confusion matrices, ROC curves, model comparison
├── outputs/                  # Generated charts
└── README.md
```

## Tech Stack

Python, scikit-learn, pandas, matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib

python week1_load_clean.py
python week2_features_eda.py
python week3_models.py
python week4_evaluate.py
```

Place `Fake.csv` and `True.csv` from the Kaggle dataset in the same folder before running.

---
**Author**: Monika · B.E.CSE With AI/ML, Sathyabama Institute of Science and Technology
