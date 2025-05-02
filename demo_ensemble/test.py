import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = './models'

# Load test dataset
df = pd.read_csv('./data/test_data.csv')

X = df['sentence'].astype(str)
y = df['label']

# Train/test split
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load vectorizer and transform test data
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
X_test_tfidf = tfidf.transform(X_test)

# Load trained models
model_names = ["MultinomialNB", "SGDClassifier", "LightGBM", "CatBoost"]
models = [joblib.load(os.path.join(MODEL_DIR, f"{name}.pkl")) for name in model_names]

# Ensemble predictions
predictions = np.array([model.predict(X_test_tfidf) for model in models])
ensemble_predictions = []
for i in range(predictions.shape[1]):
    preds, counts = np.unique(predictions[:, i], return_counts=True)
    ensemble_predictions.append(preds[np.argmax(counts)])

# Evaluate ensemble performance
acc = accuracy_score(y_test, ensemble_predictions)
report = classification_report(y_test, ensemble_predictions)

print(f"Ensemble Model Accuracy: {acc}")
print(report)