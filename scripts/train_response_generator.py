"""
train_response_generator.py
"""
import os
import yaml
import ast
import torch
import numpy as np
import pandas as pd
import nltk 
from nltk.translate.bleu_score import corpus_bleu
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
from scripts.utils import load_dataset, setup_logging

def main():
    logger = setup_logging("TrainResponseGenerator")

    # 1. Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    gen_config = config["response_generator"]

    model_name = gen_config["model_name"]
    train_data_path = gen_config["train_data"]
    val_split = gen_config["val_split"]
    batch_size = gen_config["batch_size"]
    epochs = gen_config["epochs"]
    max_length = gen_config["max_length"]
    lr = float(gen_config["learning_rate"])

    # 2. Load data (DailyDialog CSV)
    logger.info(f"Loading data from {train_data_path} ...")
    train_df, val_df = load_dataset(train_data_path, val_split=val_split)

    logger.info(f"Train size: {len(train_df)}")
    if val_df is not None:
        logger.info(f"Val size: {len(val_df)}")

 
    # 3. Initialize GPT-2 tokenizer & model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT-2 doesn't have a pad token by default; we set eos as pad
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    # 4. Preprocess function for multi-turn daily dialog
    def preprocess(examples):
        inputs = []
        raw_dialogs = examples["dialog"]
        raw_emotions = examples["emotion"]
        
        for dialog_str, emo_arr in zip(raw_dialogs, raw_emotions):
            try:
                utterances = ast.literal_eval(dialog_str)
            except:
                utterances = [dialog_str]

            if isinstance(emo_arr, str):
                try:
                    emo_arr = emo_arr.strip("[]").split()
                    emo_arr = [int(e) for e in emo_arr]
                except:
                    emo_arr = []
            if len(emo_arr) > 0:
                last_emotion_id = emo_arr[-1]
            else:
                last_emotion_id = -1

            emotion_label = f"EmotionID_{last_emotion_id}"

            if len(utterances) <= 1:
                context = utterances[0]
                response = ""
            else:
                context = " ".join(utterances[:-1])
                response = utterances[-1]

            prompt = (
                f"Emotion: {emotion_label}\n"
                f"User: {context}\n"
                "Chatbot:"
            )
            full_input = prompt + " " + response
            inputs.append(full_input)

        tokenized = tokenizer(
            inputs,
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    # 5. Convert to HF Dataset & map
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = None
    if val_df is not None and len(val_df) > 0:
        val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(preprocess, batched=True)
    if val_dataset:
        val_dataset = val_dataset.map(preprocess, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=cols)
    if val_dataset:
        val_dataset.set_format(type="torch", columns=cols)


    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir="models/response_generator",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        learning_rate=lr,
    )

    # 7. Compute Metrics (using BLEU)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_tokens = [pred.split() for pred in decoded_preds]
        ref_tokens = [[ref.split()] for ref in decoded_labels]
        bleu_score = corpus_bleu(ref_tokens, pred_tokens)
        return {"bleu": bleu_score}


    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        compute_metrics=compute_metrics
    )

    logger.info("Starting generator training...")
    trainer.train()

    logger.info("Saving generator model and tokenizer...")
    trainer.save_model("models/response_generator")
    tokenizer.save_pretrained("models/response_generator")

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
