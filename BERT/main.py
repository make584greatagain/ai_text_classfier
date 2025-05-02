from config import MODEL_SAVE_PATH
from data import load_and_prepare_data
from utils import tokenize_function
from transformers import BertTokenizerFast
from train import train_model
from evaluate import evaluate_model
from predict import predict

def main():
    dataset = load_and_prepare_data("data.csv")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenized_dataset = dataset.map(tokenize_function, batched=True).remove_columns(["sentence"])
    tokenized_dataset.set_format("torch")

    trainer = train_model(tokenized_dataset, tokenizer)
    preds = trainer.predict(tokenized_dataset["test"])
    evaluate_model(preds)

    # Save
    trainer.model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # Predict
    texts = ["This is a sample text"]
    predictions, probabilities = predict(texts)
    for text, pred, prob in zip(texts, predictions, probabilities):
        label = "Human" if pred == 0 else "AI"
        print(f"\nText: {text}\nLabel: {label}\nConfidence: Human={prob[0]:.2%}, AI={prob[1]:.2%}")

if __name__ == "__main__":
    main()
