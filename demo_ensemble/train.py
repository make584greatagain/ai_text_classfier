import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv('./data/train_data.csv')

X = df['sentence'].astype(str)
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=20000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

joblib.dump(tfidf, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))

# Model training and evaluation
def train_and_evaluate(model, model_name):
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report

# Initialize models
model_mnb = MultinomialNB()
model_sgd = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42)
model_lgb = lgb.LGBMClassifier(random_state=42)
model_cat = CatBoostClassifier(verbose=0, random_state=42)

# Train, evaluate, and save each model
for model, name in [
    (model_mnb, "MultinomialNB"),
    (model_sgd, "SGDClassifier"),
    (model_lgb, "LightGBM"),
    (model_cat, "CatBoost")
]:
    acc, report = train_and_evaluate(model, name)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

    print(f"{name} complete! Accuracy: {acc}")
    print(report)