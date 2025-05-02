import os
import joblib
import nltk
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ======== Hyperparameters ========
# MODEL_NAME = './results/lr/'
# MODEL_PATH = os.path.join(MODEL_NAME, 'LogisticRegression.pkl')
DATA_PATH = './data/independent_dataset.csv'

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

# Download NLTK tokenizer
nltk.download('punkt_tab', quiet=True)

# ======== Global Configuration ========
THRESHOLD = 0.5  # Threshold to classify text as "AI"

# ======== Load Model and Vectorizer ========
def load_models_and_vectorizer(model_dir, model_path):
    """
    Loads the TF-IDF vectorizer and the trained CatBoost model from the specified folder.
    """
    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    tfidf_vectorizer = joblib.load(tfidf_path)
    
    # Load CatBoost model
    model = joblib.load(model_path)
    
    return tfidf_vectorizer, model

# ======== Prediction Function ========
def ensemble_predict(user_text, tfidf_vectorizer, model, threshold=THRESHOLD):
    """
    Splits the input text (user_text) into sentences, transforms each sentence into TF-IDF vectors,
    and predicts the probability of AI for each sentence. The mean probability across all sentences 
    is compared with 'threshold' to decide AI vs. Human.
    
    Returns:
        (predicted_label, prob):
            predicted_label: "AI" or "Human"
            prob: The probability (confidence) for the predicted label
    """
    # Sentence tokenization
    sentences = nltk.tokenize.sent_tokenize(user_text)
    
    # If there are no valid sentences, return default values
    if not sentences:
        return "No valid sentence.", 0.0
    
    # Transform sentences into TF-IDF vectors
    X_tfidf = tfidf_vectorizer.transform(sentences)
    
    # Predict probabilities (shape: [n_samples, 2], where columns are [prob_Human, prob_AI])
    probs = model.predict_proba(X_tfidf)
    
    # Average AI probability across all sentences
    avg_prob_ai = probs[:, 1].mean()
    
    # Determine final label based on threshold
    if avg_prob_ai >= threshold:
        return 1, avg_prob_ai
    else:
        return 0, 1 - avg_prob_ai

# ======== Main Execution Block ========
if __name__ == "__main__":

    model_names = [
        './results/lr/',
        './results/cat/',
        './results/gbm/',
        './results/nb/',
        # './results/sgd/'
    ]
    model_paths = [ 
        './results/lr/LogisticRegression.pkl',
        './results/cat/CatBoost.pkl',
        './results/gbm/LightGBM.pkl',
        './results/nb/MultinomialNB.pkl',
        # './results/sgd/SGDClassifier.pkl'
    ]

    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    accuracies = {}
    for index, model_name in enumerate(model_names):
        model_path = model_paths[index]

        # Optional: Map numeric labels to string labels if needed (uncomment if applicable)
        # df['label'] = df['label'].map({0: "Human", 1: "AI"})
        
        # 2) Load model and TF-IDF vectorizer
        tfidf_vectorizer, model = load_models_and_vectorizer(model_name, model_path)
        
        # 3) Ensure the 'sentence' column is string type
        df['sentence'] = df['sentence'].astype(str)
        
        # 4) Perform predictions
        predicted_labels = []
        predicted_probs = []
        
        for _, row in df.iterrows():
            text = row['sentence']
            pred_label, prob = ensemble_predict(text, tfidf_vectorizer, model, THRESHOLD)
            predicted_labels.append(pred_label)
            predicted_probs.append(prob)
        
        # Add the prediction results to the DataFrame
        df['predicted_label'] = predicted_labels
        df['predicted_prob'] = predicted_probs
        
        # 5) Save the prediction results as a CSV file (with index labeled as 'index')
        # output_path = './predictions_output.csv'
        # df.to_csv(output_path, index_label='index')
        # print(f"Prediction results have been saved to: {output_path}")

        test_accuracy = accuracy_score(df['label'], df['predicted_label'])
        print(model_path, " model against independent set accuracy: ", test_accuracy)

        accuracies[model_path] = test_accuracy

accuracies['NavieBayes'] = 0.54
# leave a blank to avoid model name conflict
accuracies[' LogisticRegression'] = 0.54
print(accuracies)

model_names = list(os.path.splitext(os.path.basename(path))[0] for path in accuracies.keys())
accuracy_values = list(accuracies.values())



colors = ['royalblue'] * 4 + ['darkorange'] * 2
labels = ['Dataset A'] * 4 + ['Dataset B'] * 2 

plt.figure(figsize=(10, 9))
bars = plt.bar(model_names, accuracy_values, color=colors, edgecolor="black", linewidth=1.2)

plt.xlabel("Model", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
plt.title("Model Accuracy for Independent Test Set", fontsize=16, fontweight='bold', pad=15)
plt.ylim(0.4, 0.7)
plt.xticks(rotation=30, fontsize=12)
plt.yticks(fontsize=12)

for bar, acc in zip(bars, accuracy_values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{acc:.2%}", 
             ha='center', fontsize=12, fontweight='bold', color="black")

plt.axvline(x=3.5, color='red', linestyle='--', linewidth=2, label='Dataset Boundary')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='royalblue', edgecolor='black', label='Trained with Smaller Dataset in Phase I'),
                   Patch(facecolor='darkorange', edgecolor='black', label='Trained with Larger Dataset in Phase II')]
plt.legend(handles=legend_elements, fontsize=12, loc='upper right')

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.show()