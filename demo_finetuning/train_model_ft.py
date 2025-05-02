import os
import json
import multiprocessing as mp
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Paths and CPU settings
MODEL_DIR = "./models/model_ft"
os.makedirs(MODEL_DIR, exist_ok=True)
n_cores = mp.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)

# Load and split data
df = pd.read_csv("data/train_data.csv")
X_raw, y_raw = df["sentence"].astype(str), df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
)

# Fixed TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=40_000,
    stop_words=None,
    sublinear_tf=True,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression parameter grid
base_lr = LogisticRegression(solver="saga", max_iter=3000, n_jobs=-1, random_state=42)

param_grid = [
    {"penalty": ["l1", "l2"],
     "C": [0.01, 0.1, 1, 10],
     "class_weight": [None, "balanced"]},

    {"penalty": ["elasticnet"],
     "C": [0.01, 0.1, 1, 10],
     "l1_ratio": [0.25, 0.5, 0.75],
     "class_weight": [None, "balanced"]},

    {"penalty": ["none"],
     "class_weight": [None, "balanced"]},
]

# Grid search with cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=base_lr,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    return_train_score=False,
)

grid.fit(X_train_vec, y_train)

# Best cross-validation result
print("\n======== Best CV Result ========")
print(f"Mean CV accuracy: {grid.best_score_:.4%}")
print("Best params:", grid.best_params_)

# Evaluate on test set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_vec)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {test_acc:.4%}")
print(classification_report(y_test, y_pred, digits=4))

# Save only the best artifacts
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "best_vectorizer.pkl"))
joblib.dump(best_model, os.path.join(MODEL_DIR, "lr_besbest_classifiert.pkl"))

with open(os.path.join(MODEL_DIR, "best_params_.json"), "w", encoding="utf-8") as f:
    json.dump(grid.best_params_, f, indent=2, ensure_ascii=False)