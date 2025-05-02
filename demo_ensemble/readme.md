# AI Text Classification with Ensemble

This repository provides a complete workflow for text classification using multiple machine learning models and ensemble methods for improved accuracy.

## Project Structure

```
.
├── data
│   ├── train_data.csv       # Training data (sentence, label)
│   └── test_data.csv        # Testing data (sentence, label)
├── models                   # Directory to save/load trained models
├── train.py                 # Script for training and saving models
├── test.py                  # Script for evaluating ensemble models
├── requirements.txt         # Dependencies for the project
└── README.md                # Project documentation
```

## Usage

### Model Training

Train and save models using the provided script:

```bash
python train.py
```

This script trains the following models:

* Multinomial Naive Bayes
* SGD Classifier
* LightGBM Classifier
* CatBoost Classifier

Models and the TF-IDF vectorizer are saved in the `./models` directory.

### Ensemble Testing

Evaluate trained models using an ensemble approach:

```bash
python test.py
```

This script:

* Loads trained models and TF-IDF vectorizer.
* Performs ensemble predictions using majority voting.
* Outputs accuracy and detailed classification report.

## Requirements

Python 3.8 or newer is recommended.

Refer to `requirements.txt` for exact package versions.