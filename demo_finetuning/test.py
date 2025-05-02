import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Paths
MODEL_DIR = "./models/combined_ft"

# Load test data
df = pd.read_csv("./data/test_data.csv")
X_test, y_test = df["sentence"].astype(str), df["label"]

# Load vectorizer and model
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
classifier = joblib.load(os.path.join(MODEL_DIR, "LogisticRegression.pkl"))

# Vectorize test data
X_test_vec = vectorizer.transform(X_test)

# Predict and evaluate
y_pred = classifier.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

print(f"Test Accuracy: {accuracy:.4%}")
print("Classification Report:")
print(report)