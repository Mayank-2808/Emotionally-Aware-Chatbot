"""
train_emotion_classifier.py
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from scripts.utils import load_dataset, setup_logging

from sklearn.metrics import f1_score, precision_score, recall_score

# 1) Subclass the Trainer to override compute_loss

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom multi-label loss using BCEWithLogitsLoss.
        """
        labels = inputs.pop("labels")          # pop "labels" so model doesn't receive it as input
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.BCEWithLogitsLoss()
        # Ensure labels are float
        labels = labels.float()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    logger = setup_logging("TrainEmotionClassifier")

    # 2. Load config

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    clf_config = config["emotion_classifier"]

    model_name = clf_config["model_name"]
    num_labels = clf_config["num_labels"]
    train_data_path = clf_config["train_data"]
    val_split = clf_config["val_split"]
    batch_size = clf_config["batch_size"]
    epochs = clf_config["epochs"]
    max_length = clf_config["max_length"]
    lr = float(clf_config["learning_rate"])

    # 3. Load data

    logger.info(f"Loading data from {train_data_path}...")
    train_df, val_df = load_dataset(train_data_path, val_split=val_split)
    logger.info(f"Training samples: {len(train_df)}")
    if val_df is not None:
        logger.info(f"Validation samples: {len(val_df)}")

    # 4. Define the list of emotions in the same order as the CSV columns
 
    all_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]

    assert len(all_labels) == num_labels, (
        f"Config says num_labels={num_labels} but all_labels has {len(all_labels)} items."
    )

   
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}
    id2label = {i: lbl for i, lbl in enumerate(all_labels)}
    logger.info(f"Label mapping: {label2id}")


    # 5. Initialize model & tokenizer

    if "roberta" in model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

 
    # 6. Preprocess function for multi-label

    def preprocess(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

        # Build a multi-hot label vector for each row
        batch_size = len(examples["text"])
        label_matrix = []
        for i in range(batch_size):
            row_labels = []
            for lbl in all_labels:
                # each emotion column is 0/1 in the CSV
                row_labels.append(examples[lbl][i])
            label_matrix.append(row_labels)

        tokenized["labels"] = label_matrix
        return tokenized

 
    # 7. Convert train/val DataFrame to Dataset & map

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = None
    if val_df is not None and len(val_df) > 0:
        val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(preprocess, batched=True)
    if val_dataset is not None:
        val_dataset = val_dataset.map(preprocess, batched=True)

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    if val_dataset is not None:
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 8. TrainingArguments
  
    training_args = TrainingArguments(
        output_dir="models/emotion_classifier",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",  # or "epoch"
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        learning_rate=lr,
        logging_dir="logs/emotion_classifier",  # optional
    )

    # 9. Multi-label metrics function

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-logits))
        # Threshold at 0.5
        y_pred = (probs >= 0.5).astype(int)
        y_true = labels

        # micro-averaged F1, P, R
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
        recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro
        }


    # 10. Create our custom multi-label Trainer

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )


    # 11. Train

    logger.info("Starting training ...")
    trainer.train()

    # 12. Save final model and tokenizer

    logger.info("Saving model and tokenizer ...")
    trainer.save_model("models/emotion_classifier")
    tokenizer.save_pretrained("models/emotion_classifier")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
