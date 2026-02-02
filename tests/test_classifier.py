"""
test_classifier.py
"""

import pytest
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# This label list must match the one used during training
ALL_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]

@pytest.mark.parametrize("test_input, expected_emotion", [
    ("I am so happy to see you again!", "joy"),
    ("This is really terrible.", "sadness"),
    ("Wow, that's hilarious!", "amusement"),
    ("I'm a bit afraid of the dark.", "fear"),
])
def test_emotion_classifier(test_input, expected_emotion):
    """
    This test tries to see if the multi-label classifier can detect
    a known emotion in the input text.
    """
    # Load classifier
    tokenizer = AutoTokenizer.from_pretrained("models/emotion_classifier")
    model = AutoModelForSequenceClassification.from_pretrained("models/emotion_classifier")
    model.eval()

    # Tokenize
    inputs = tokenizer(test_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: [1, num_labels]

    # Convert logits -> probabilities via sigmoid
    probs = torch.sigmoid(logits).squeeze().numpy()  # shape: [num_labels]

    # Decide which labels are predicted (prob >= 0.5)
    pred_bools = probs >= 0.5
    predicted_labels = [
        label for label, is_pred in zip(ALL_LABELS, pred_bools) if is_pred
    ]

    # For testing, just check if the expected_emotion is among the predicted
    assert expected_emotion in ALL_LABELS, f"'{expected_emotion}' not in label list!"
    assert (expected_emotion in predicted_labels), (
        f"Expected '{expected_emotion}', got predictions: {predicted_labels}"
    )
