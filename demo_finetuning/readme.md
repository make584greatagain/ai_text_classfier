# AI Text Classification with TF-IDF and Logistic Regression

This repository provides a complete workflow for text classification using TF-IDF vectorization and Logistic Regression, including hyperparameter tuning and evaluation scripts for improved accuracy.

## Project Structure

```
.
├── data
│   ├── train_data.csv       # Training data (sentence, label)
│   └── test_data.csv        # Testing data (sentence, label)
├── models                   # Directory to save/load trained models
├── vectorize_ft.py          # Script for TF-IDF vectorizer tuning
├── model_ft.py              # Script for Logistic Regression hyperparameter tuning
├── combined_ft.py           # Script for training with fixed optimal parameters
├── test.py                  # Script for evaluating trained models
├── requirements.txt         # Dependencies for the project
└── README.md                # Project documentation
```

## Usage

### Model Training

Train models with hyperparameter tuning or fixed parameters:

```bash
python vectorize_ft.py
python model_ft.py
python combined_ft.py
```

These scripts perform:

* Hyperparameter tuning for TF-IDF vectorizer.
* Hyperparameter tuning for Logistic Regression.
* Training with fixed, optimal parameters.

Trained models and TF-IDF vectorizers are saved in the `./models` directory.

### Model Evaluation

Evaluate trained models on test data:

```bash
python test.py
```

This script:

* Loads trained models and TF-IDF vectorizer.
* Outputs accuracy and detailed classification report.

## Requirements

Python 3.8 or newer is recommended.

Refer to `requirements.txt` for exact package versions.