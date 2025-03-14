import os
import pandas as pd
import joblib  # For saving models with joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Evaluation metrics
from sklearn.metrics import classification_report, accuracy_score

# Name of the folder where we will save models and logs
MODEL_NAME = './results/train-eval-0'

# Create the folder if it doesn't already exist
os.makedirs(MODEL_NAME, exist_ok=True)

# ----- 1. Load CSV -----
df = pd.read_csv('./data/merged_text_label_per_sentence.csv')

# Quick data check
print(df.head())
print(df['label'].value_counts())

# ----- 2. Split into (text, label) -----
# Convert to string to handle any potential NaN values
X = df['text'].astype(str)
y = df['label']

# ----- 3. Split data into Train/Test sets -----
# Use 20% of the data as a test set and stratify if classes are imbalanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----- 4. Tfidf Vectorization -----
# Customize stopwords, n-grams, or other parameters as needed
tfidf = TfidfVectorizer(
    # stop_words='english',   # Uncomment to remove English stopwords
    # ngram_range=(1,2),       # Use 1-gram to 3-gram range
    max_features=20000       # Limit the maximum number of features (for memory considerations)
)

# ----- 5. Fit the vectorizer on the training data, then transform both Train/Test data -----
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Save the trained TfidfVectorizer for future use
joblib.dump(tfidf, os.path.join(MODEL_NAME, 'tfidf_vectorizer.pkl'))

# ----- 6. Training and evaluation function -----
def train_and_evaluate(model, model_name):
    """
    Trains the given model using the prepared TF-IDF vectors,
    makes predictions, and returns accuracy and a classification report.
    """
    # Train the model
    model.fit(X_train_tfidf, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test_tfidf)

    # Calculate accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return acc, report

# ----- 7. Define four models -----
model_mnb = MultinomialNB()
model_sgd = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42)
model_lgb = lgb.LGBMClassifier(random_state=42)
model_cat = CatBoostClassifier(verbose=0, random_state=42)

# Prepare the log file path
log_path = os.path.join(MODEL_NAME, 'training_log.txt')

# ----- 8. Train & evaluate each model, then save models and logs -----
for model, name in [
    (model_mnb, "MultinomialNB"),
    (model_sgd, "SGDClassifier"),
    (model_lgb, "LightGBM"),
    (model_cat, "CatBoost")
]:
    # Train the model and get performance metrics
    acc, report = train_and_evaluate(model, name)
    
    # Save the trained model with joblib
    model_save_path = os.path.join(MODEL_NAME, f"{name}.pkl")
    joblib.dump(model, model_save_path)

    # Append the evaluation log to a file
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n========== {name} ==========\n")
        f.write(f"Accuracy: {acc}\n")
        f.write(report + "\n")

    # Print results
    print(f"{name} training and saving complete!")
    print(f"Accuracy: {acc}")
    print(report)