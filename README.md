# Emotionally Aware Chatbot

This repository contains a two-stage chatbot system:

1. **Emotion Detection**: Classifies user messages into emotion categories using RoBERTa.
2. **Emotion-Sensitive Response Generation**: Generates responses conditioned on the detected emotion using GPT-2.

## Project Structure

- `data/`: Contains sample CSV files for emotion classification and dialogue generation.
- `models/`: Contains folders where trained models (emotion classifier and response generator) will be saved.
- `scripts/`: Training and inference scripts.
- `app.py`: Gradio-based UI for live interaction.
- `tests/`: Basic tests for classifier and generator.

## Installation & Setup

1. **Clone this repo**:
   ```bash
   git clone <REPO_URL>
   cd Final_Project

2. **Install Python Dependencies**:
    pip install -r requirements.txt

3. **Configure environment**:
	•	GPU support is recommended for training with large datasets.
	•	Update any hyperparameters or file paths in config.yaml.

4. **Data**:
	•	We provide dailydialog.csv.zip and goemotions_1.csv.zip for demonstration.

## Usage

1. **Train Emotion Classifier**:
    python -m scripts.train_emotion_classifier

    The trained classifier will be saved under models/emotion_classifier.

2. **Train Response Generator**:
    python -m scripts.train_response_generator

    The model will be saved under models/response_generator.

3. **Run Inference**:
    python -m scripts.inference --text "I am feeling fantastic\!"

    This will:
	1.	Detect emotion of the input text.
	2.	Generate a response conditioned on the detected emotion.

4. **Launch Gradio Demo**:
    python app.py

    Open the displayed local URL in your browser to interact with the chatbot.

5. **Testing**:
    pytest tests

    Runs the basic test suite to check if the classification and generation pipelines work on the sample data.

## Acknowledgements

	•	Hugging Face Transformers
	•	GoEmotions Dataset
	•	RoBERTa
	•	GPT-2