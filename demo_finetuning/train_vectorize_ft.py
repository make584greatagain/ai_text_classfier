import os
import time
import json
import itertools
import multiprocessing as mp
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths and environment setup
MODEL_DIR = "./models/vectorize_ft"
os.makedirs(MODEL_DIR, exist_ok=True)

num_cores = mp.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)

# Data loading and splitting
df = pd.read_csv("data/train_data.csv")
X_raw, y_raw = df["sentence"].astype(str), df["label"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)

# Hyperparameter combinations
param_grid = {
    "ngram_range": [(1, 1), (1, 2)],
    "max_features": [20_000, 40_000],
    "stop_words": [None, "english"],
    "sublinear_tf": [False, True],
}
grid_keys = sorted(param_grid)
param_sets = [dict(zip(grid_keys, combo)) for combo in itertools.product(*param_grid.values())]

best_accuracy = 0
best_params = None
best_vectorizer = None
best_classifier = None

# Experiment loop
for params in param_sets:
    print(f"\n▶ Testing configuration: {json.dumps(params)}")
    start_time = time.time()

    # Vectorize
    vectorizer = TfidfVectorizer(**params)
    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_te_vec = vectorizer.transform(X_te)

    # Train Logistic Regression
    classifier = LogisticRegression(
        solver="saga", max_iter=2000, n_jobs=-1, random_state=42
    )
    classifier.fit(X_tr_vec, y_tr)

    # Evaluate
    y_pred = classifier.predict(X_te_vec)
    acc = accuracy_score(y_te, y_pred)
    elapsed = time.time() - start_time
    print(f"   ↳ accuracy = {acc:.4%}   elapsed = {elapsed:.1f}s")

    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_params = params
        best_vectorizer = vectorizer
        best_classifier = classifier

# Save best model and vectorizer
joblib.dump(best_vectorizer, os.path.join(MODEL_DIR, "best_vectorizer.pkl"))
joblib.dump(best_classifier, os.path.join(MODEL_DIR, "best_classifier.pkl"))