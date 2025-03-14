import os
import joblib
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt', quiet=True)

# ========= 1) Prepare Test Set =========
df = pd.read_csv('./data/merged_text_label_per_sentence.csv')

# 만약 df['label']이 0/1 형태라면, Human/AI로 매핑
# (이미 Human/AI 문자열이면 이 단계는 필요 없음)
df['label'] = df['label'].map({0: "Human", 1: "AI"})

print(df.head())
print(df['label'].value_counts())

X = df['text'].astype(str)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ========= 2) Load Models and Vectorizer =========
MODEL_NAME = 'train-eval-5'
THRESHOLD = 0.5

def load_models_and_vectorizer(folder_path=MODEL_NAME):
    tfidf_path = os.path.join(folder_path, 'tfidf_vectorizer.pkl')
    tfidf_vectorizer = joblib.load(tfidf_path)
    
    model_paths = {
        'MultinomialNB': 'MultinomialNB.pkl',
        'SGDClassifier': 'SGDClassifier.pkl',
        'LightGBM': 'LightGBM.pkl',
        'CatBoost': 'CatBoost.pkl'
    }
    
    models = {}
    for model_name, model_file in model_paths.items():
        full_path = os.path.join(folder_path, model_file)
        models[model_name] = joblib.load(full_path)
    
    return tfidf_vectorizer, models

def ensemble_predict(user_text, tfidf_vectorizer, models):
    sentences = nltk.tokenize.sent_tokenize(user_text)
    if not sentences:
        return "No valid sentence.", 0.0
    
    X_tfidf = tfidf_vectorizer.transform(sentences)
    
    model_probs = []
    for model_name, model in models.items():
        probs = model.predict_proba(X_tfidf)
        avg_prob_ai = probs[:, 1].mean()
        model_probs.append(avg_prob_ai)
    
    final_prob_ai = sum(model_probs) / len(model_probs)
    final_prob_human = 1 - final_prob_ai
    
    if final_prob_ai >= THRESHOLD:
        return "AI", final_prob_ai
    else:
        return "Human", final_prob_human

# ========= 3) Evaluate on Test Set =========
if __name__ == "__main__":
    tfidf_vectorizer, models = load_models_and_vectorizer(MODEL_NAME)
    
    # We'll gather predictions for every sample in X_test
    y_preds = []
    
    for text in X_test:
        label, conf = ensemble_predict(text, tfidf_vectorizer, models)
        # label: "AI" or "Human"
        y_preds.append(label)
    
    y_test_list = list(y_test)   # already "Human"/"AI" after map
    y_preds_list = list(y_preds) # "Human"/"AI"
    
    acc = accuracy_score(y_test_list, y_preds_list)
    print("\n===== Ensemble Model Evaluation on Test Set =====")
    print("Accuracy:", acc)
    
    print("\nClassification Report:")
    # 이제 y_test_list와 y_preds_list 둘 다 문자열이므로 OK
    print(classification_report(y_test_list, y_preds_list,
                                target_names=["Human", "AI"]))

# ===== Ensemble Model Evaluation on Test Set =====
# Accuracy: 0.8822439468338543

# Classification Report:
#               precision    recall  f1-score   support

#        Human       0.89      0.74      0.81    112909
#           AI       0.88      0.95      0.91    223320

#     accuracy                           0.88    336229
#    macro avg       0.88      0.85      0.86    336229
# weighted avg       0.88      0.88      0.88    336229