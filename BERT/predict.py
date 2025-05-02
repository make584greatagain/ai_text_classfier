import torch
from config import MODEL_SAVE_PATH
from transformers import BertTokenizerFast, BertForSequenceClassification
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(device)
model.eval()

def predict(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    return preds.cpu().numpy(), probs.cpu().numpy()
