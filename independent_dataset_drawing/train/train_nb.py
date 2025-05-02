import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

MODEL_NAME = './phase-2/results/nb'
os.makedirs(MODEL_NAME, exist_ok=True)

df = pd.read_csv('./phase-2/data/combined_unique.csv')
print(df.head())
print(df['label'].value_counts())

X = df['sentence'].astype(str)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=20000
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

joblib.dump(tfidf, os.path.join(MODEL_NAME, 'tfidf_vectorizer.pkl'))

def train_and_evaluate(model, model_name):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

model_nb = MultinomialNB()
log_path = os.path.join(MODEL_NAME, 'training_log.txt')

for model, name in [
    (model_nb, "MultinomialNB")
]:
    acc, report = train_and_evaluate(model, name)
    model_save_path = os.path.join(MODEL_NAME, f"{name}.pkl")
    joblib.dump(model, model_save_path)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n========== {name} ==========\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(report + "\n")

    print(f"{name} training and saving complete!")
    print(f"Accuracy: {acc}")
    print(report)