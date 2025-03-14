import os
import joblib
import nltk

# Download the 'punkt' tokenizer if it isn't already installed
nltk.download('punkt_tab', quiet=True)

# Default folder name where vectorizer and models are stored
MODEL_NAME = './results/train-eval-5'

# Set the threshold for deciding AI vs Human
THRESHOLD = 0.5

USER_INPUT = "Machine learning is a method of data analysis that allows computers to learn patterns and make predictions or decisions without being explicitly programmed."

def load_models_and_vectorizer(folder_path=MODEL_NAME):
    """
    Loads the TfidfVectorizer and four predefined models (MultinomialNB, SGDClassifier,
    LightGBM, CatBoost) from the specified folder path.
    
    :param folder_path: The directory containing 'tfidf_vectorizer.pkl' and the model .pkl files
    :return: (tfidf_vectorizer, models)
    """
    # Load the saved TfidfVectorizer
    tfidf_path = os.path.join(folder_path, 'tfidf_vectorizer.pkl')
    tfidf_vectorizer = joblib.load(tfidf_path)
    
    # Dictionary of model file names
    model_paths = {
        'MultinomialNB': 'MultinomialNB.pkl',
        'SGDClassifier': 'SGDClassifier.pkl',
        'LightGBM': 'LightGBM.pkl',
        'CatBoost': 'CatBoost.pkl'
    }
    
    # Load each model from its corresponding file
    models = {}
    for model_name, model_file in model_paths.items():
        model_full_path = os.path.join(folder_path, model_file)
        models[model_name] = joblib.load(model_full_path)
    
    return tfidf_vectorizer, models


def ensemble_predict(user_text, tfidf_vectorizer, models):
    """
    Splits user_text into sentences, uses the provided tfidf_vectorizer to transform them,
    then obtains prediction probabilities from all models to produce an ensemble result.
    
    Process:
      1) Split text into sentences
      2) Vectorize each sentence using tfidf_vectorizer
      3) For each model, compute predict_proba and average the probability of the 'AI' class
      4) Ensemble by taking the mean of these probabilities (soft voting)
      5) Decide 'AI' if the average probability >= THRESHOLD, else 'Human'
      
    Assumptions:
      - Binary classification: 0 -> Human, 1 -> AI
      - Each model implements predict_proba
      - The user text can be any arbitrary string
    
    :param user_text: The text to be analyzed
    :param tfidf_vectorizer: A trained TfidfVectorizer object
    :param models: A dictionary of loaded models
    :return: (final_label, confidence)
    """
    # 1) Split the text into sentences
    sentences = nltk.tokenize.sent_tokenize(user_text)
    
    # If no valid sentences, return early
    if not sentences:
        return "No valid sentence.", 0.0
    
    # 2) Transform the sentences using the loaded TF-IDF vectorizer
    X_tfidf = tfidf_vectorizer.transform(sentences)
    
    # 3) Collect probabilities of the 'AI' class from each model, averaged over sentences
    model_probs = []
    for model_name, model in models.items():
        probs = model.predict_proba(X_tfidf)  # shape: (n_sentences, 2)
        avg_prob_ai = probs[:, 1].mean()      # Probability of the AI class
        model_probs.append(avg_prob_ai)
    
    # 4) Ensemble by averaging each model's mean probability
    final_prob_ai = sum(model_probs) / len(model_probs)
    final_prob_human = 1 - final_prob_ai
    
    # 5) Determine the final label and confidence
    if final_prob_ai >= THRESHOLD:
        final_label = "AI"
        confidence = final_prob_ai
    else:
        final_label = "Human"
        confidence = final_prob_human

    return final_label, confidence


if __name__ == "__main__":
    # 1) Load the TF-IDF vectorizer and all models
    tfidf_vectorizer, models = load_models_and_vectorizer(folder_path=MODEL_NAME)

    # 2) Generate the ensemble prediction using the loaded models
    predicted_label, conf = ensemble_predict(USER_INPUT, tfidf_vectorizer, models)
    
    # 3) Print the prediction results
    print("===== Prediction Result =====")
    print(f"Input Text:\n{USER_INPUT.strip()}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence (Probability): {conf:.4f}")