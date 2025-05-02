# run_predict.py

import argparse
from predict import predict

def main():
    parser = argparse.ArgumentParser(description="Predict whether text is Human or AI-generated.")
    parser.add_argument("--text", type=str, required=True, help="Input sentence to classify")

    args = parser.parse_args()
    input_text = [args.text]

    predictions, probabilities = predict(input_text)

    for text, pred, prob in zip(input_text, predictions, probabilities):
        label = "Human" if pred == 0 else "AI"
        print("\n==============================")
        print(f"Text: {text}")
        print(f"Predicted Label: {label}")
        print(f"Confidence: Human={prob[0]:.2%}, AI={prob[1]:.2%}")
        print("==============================\n")

if __name__ == "__main__":
    main()
