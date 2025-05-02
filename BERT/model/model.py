from transformers import BertForSequenceClassification
from config import MODEL_NAME, NUM_LABELS

def get_model():
    return BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
