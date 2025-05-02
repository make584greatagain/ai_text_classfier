import os
import time
import json
import multiprocessing as mp
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths and CPU configuration
MODEL_DIR = "./models/combined_ft"
os.makedirs(MODEL_DIR, exist_ok=True)

num_cores = mp.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

# Load and split data
df = pd.read_csv("./data/train_data.csv")
X_raw, y_raw = df["sentence"].astype(str), df["label"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)

# Fixed vectorizer and classifier parameters
vec_params = {
    "max_features": 40_000,
    "ngram_range": (1, 2),
    "stop_words": None,
    "sublinear_tf": True,
}
clf_params = {
    "C": 10,
    "class_weight": None,
    "penalty": "l2",
}
tag = "ng1-2__mf40k__sw-none__subtf"

print(f"\n▶ Running fixed configuration experiment")
start_time = time.time()

# Vectorization
vectorizer = TfidfVectorizer(**vec_params)
X_tr_vec = vectorizer.fit_transform(X_tr)
X_te_vec = vectorizer.transform(X_te)

# Training Logistic Regression
classifier = LogisticRegression(
    solver="saga", max_iter=2000, n_jobs=-1, random_state=42, **clf_params
)
classifier.fit(X_tr_vec, y_tr)

# Evaluation
y_pred = classifier.predict(X_te_vec)
acc = accuracy_score(y_te, y_pred)
report = classification_report(y_te, y_pred, digits=4)
elapsed = time.time() - start_time

print(f"   ↳ accuracy = {acc:.4%}   elapsed = {elapsed:.1f}s")

# Save model and vectorizer
joblib.dump(vectorizer, os.path.join(MODEL_DIR, f"tfidf.pkl"))
joblib.dump(classifier, os.path.join(MODEL_DIR, f"LogisticRegression.pkl"))

print("\n✓ Experiment completed!")