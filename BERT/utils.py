from transformers import BertTokenizerFast
from config import MODEL_NAME, MAX_LENGTH

def tokenize_function(examples):
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }
