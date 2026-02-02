"""
inference.py
"""

import argparse
import torch
import yaml
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel
)
from scripts.utils import setup_logging

def load_classifier(model_path):

    # Attempts to load a multi-label emotion classifier.
    # Tries RoBERTa first and falls back to BERT if necessary.

    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        from transformers import BertTokenizer, BertForSequenceClassification
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def load_generator(model_path):

    # Loads the GPT-2 based response generator
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model

def main():
    logger = setup_logging("EnhancedInference")

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="User input text")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    inf_config = config["inference"]
    classifier_path = inf_config["classifier_path"]
    generator_path = inf_config["generator_path"]
    max_new_tokens = inf_config["max_new_tokens"]
    temperature = inf_config["temperature"]
    top_p = inf_config["top_p"]

    # 1) Load the multi-label emotion classifier and detect emotion probabilities
    clf_tokenizer, clf_model = load_classifier(classifier_path)
    clf_model.eval()

    inputs = clf_tokenizer(args.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        output = clf_model(**inputs)
        logits = output.logits 

    # Compute probabilities using the sigmoid function
    probs = torch.sigmoid(logits).squeeze()  # Shape: [num_labels]

    all_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval",
        "caring", "confusion", "curiosity", "desire", "disappointment",
        "disapproval", "disgust", "embarrassment", "excitement", "fear",
        "gratitude", "grief", "joy", "love", "nervousness",
        "optimism", "pride", "realization", "relief", "remorse",
        "sadness", "surprise", "neutral"
    ]

    # Pair each emotion with its probability and sort them in descending order
    emotion_probs = list(zip(all_labels, probs.cpu().numpy()))
    emotion_probs_sorted = sorted(emotion_probs, key=lambda x: x[1], reverse=True)

    # Select the top three emotions
    top_emotions = emotion_probs_sorted[:3]
    formatted_emotions = ", ".join([f"{emotion} ({prob:.2f})" for emotion, prob in top_emotions])
    logger.info(f"Enhanced Detected Emotions: {formatted_emotions}")

    # 2) Generate response with GPT-2 conditioned on the enriched emotion context
    gen_tokenizer, gen_model = load_generator(generator_path)
    gen_model.eval()

    # Build an enhanced prompt that includes the top emotion probabilities
    prompt = (
        f"Detected Emotions: {formatted_emotions}\n"
        f"User: {args.text}\n"
        "Chatbot:"
    )
    
    input_ids = gen_tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        generated_ids = gen_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=gen_tokenizer.eos_token_id
        )
    
    # Decode the generated sequence and extract the chatbot response
    response = gen_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "Chatbot:" in response:
        response = response.split("Chatbot:")[-1].strip()

    print(f"\nUser: {args.text}")
    print(f"Enhanced Detected Emotions: {formatted_emotions}")
    print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    main()