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

def plot_accuracy_bar_chart(model_names, accuracy_values, title, filename):

    colors = ['royalblue'] * 3 + ['darkorange'] * 3

    plt.figure(figsize=(10, 9))
    bars = plt.bar(model_names, accuracy_values, color=colors, edgecolor="black", linewidth=1.2)

    plt.xlabel("Model", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.ylim(0.3, 0.9)
    plt.xticks(rotation=30, fontsize=12)
    plt.yticks(fontsize=12)

    for bar, acc in zip(bars, accuracy_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{acc:.2%}", 
                ha='center', fontsize=12, fontweight='bold', color="black")

    plt.savefig(filename, format="jpg", dpi=300, bbox_inches='tight')
    plt.close()

# ======== Main Execution Block ========
if __name__ == "__main__":

    model_names = [
        './results/cat/',
        './results/gbm/',
        './results/nb/',
        './results/lr/',
        './results/lr_vectorize_tuned/',
        './results/lr_vectorize_model_tuned/',
        # './results/sgd/'
    ]
    model_paths = [ 
        './results/cat/CatBoost.pkl',
        './results/gbm/LightGBM.pkl',
        './results/nb/MultinomialNB.pkl',
        './results/lr/LogisticRegression.pkl',
        './results/lr_vectorize_tuned/lr_vectorize_tuned.pkl',
        './results/lr_vectorize_model_tuned/lr_vectorize_model_tuned.pkl',
        # './results/sgd/SGDClassifier.pkl'
    ]

    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    accuracies = {}
    accuracies_0 = {}
    accuracies_1 = {}
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
        label_0_accuracy = accuracy_score(df[df['label'] == 0]['label'], df[df['label'] == 0]['predicted_label'])
        label_1_accuracy = accuracy_score(df[df['label'] == 1]['label'], df[df['label'] == 1]['predicted_label'])

        print(model_path, " model against independent set accuracy: ", test_accuracy)

        accuracies[model_path] = test_accuracy
        accuracies_0[model_path] = label_0_accuracy
        accuracies_1[model_path] = label_1_accuracy
        

    print(accuracies)
    print(label_0_accuracy)
    print(label_1_accuracy)

    model_names = list(os.path.splitext(os.path.basename(path))[0] for path in accuracies.keys())
    accuracy_values = list(accuracies.values())
    accuracy_0_values = list(accuracies_0.values())
    accuracy_1_values = list(accuracies_1.values())

    plot_accuracy_bar_chart(model_names, accuracy_values, "Model Accuracy for Independent Test Set", "accuracy.jpg")
    plot_accuracy_bar_chart(model_names, accuracy_0_values, "Model Accuracy for Independent Test Set (Humain Written)", "accuracy_0.jpg")
    plot_accuracy_bar_chart(model_names, accuracy_1_values, "Model Accuracy for Independent Test Set (AI generated)", "accuracy_1.jpg")
