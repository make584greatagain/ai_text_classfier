import torch
from transformers import Trainer, TrainingArguments
from model.bert_model import get_model
from utils import compute_metrics
from config import OUTPUT_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, SEED

def train_model(tokenized_dataset, tokenizer):
    model = get_model().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        report_to="none",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # Save model and tokenizer after training
    model.save_pretrained("bert_detector/model")
    tokenizer.save_pretrained("bert_detector/model")
    return trainer




