"""
app.py
"""
import gradio as gr
import torch
import yaml
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel
)

# 1. Load config, get inference settings

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
inf_config = config["inference"]
classifier_path = inf_config["classifier_path"]
generator_path = inf_config["generator_path"]


# 2. Load the multi-label emotion classifier

try:
    classifier_tokenizer = RobertaTokenizer.from_pretrained(classifier_path)
    classifier_model = RobertaForSequenceClassification.from_pretrained(classifier_path)
except:
    classifier_tokenizer = BertTokenizer.from_pretrained(classifier_path)
    classifier_model = BertForSequenceClassification.from_pretrained(classifier_path)

classifier_model.eval()

# 3. Load GPT-2 response generator

gen_tokenizer = GPT2Tokenizer.from_pretrained(generator_path)
gen_tokenizer.pad_token = gen_tokenizer.eos_token
gen_model = GPT2LMHeadModel.from_pretrained(generator_path)
gen_model.eval()

# 4. Define the label list (same as used during training)
all_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral"
]


# 5. Inference function for Gradio

def chatbot_response(user_text):

    # 5a) Multi-label emotion detection
    inputs = classifier_tokenizer(user_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = classifier_model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze()  # shape: [num_labels]
    # Threshold at 0.5
    pred_bools = (probs >= 0.5).cpu().numpy().astype(bool)
    predicted_emotions = [label for label, is_present in zip(all_labels, pred_bools) if is_present]

    # Fallback to "neutral" if no emotion above threshold
    if not predicted_emotions:
        predicted_emotions = ["neutral"]

    # For display
    emotion_str = ", ".join(predicted_emotions)

    # 5b) Generate response with GPT-2
    prompt = (
        f"Emotion(s): {emotion_str}\n"
        f"User: {user_text}\n"
        "Chatbot:"
    )
    input_ids = gen_tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = gen_model.generate(
        input_ids,
        max_new_tokens=inf_config["max_new_tokens"],
        temperature=inf_config["temperature"],
        top_p=inf_config["top_p"],
        do_sample=True,
        pad_token_id=gen_tokenizer.eos_token_id
    )
    output_text = gen_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if "Chatbot:" in output_text:
        output_text = output_text.split("Chatbot:")[-1].strip()

    return f"Emotion Detected: {emotion_str}\n\n{output_text}"

# 6. Create Gradio Interface
demo = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="text",
    title="Emotionally Aware Chatbot",
    description=(
        "A multi-label emotion classifier detects your emotion(s) "
        "and GPT-2 generates a response based on that emotion context."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)