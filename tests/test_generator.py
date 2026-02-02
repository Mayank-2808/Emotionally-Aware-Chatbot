"""
test_generator.py
"""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@pytest.mark.parametrize("emotion", ["happy", "sad"])
def test_response_generator(emotion):

    # Load tokenizer and model from your fine-tuned 'models/response_generator'
    tokenizer = AutoTokenizer.from_pretrained("models/response_generator")
    model = AutoModelForCausalLM.from_pretrained("models/response_generator")
    model.eval()

    # Create a sample prompt following the same format you used during training
    prompt = f"Emotion: {emotion}\nUser: This is a test\nChatbot:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate a short response for testing
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    if "Chatbot:" in response:
        response = response.split("Chatbot:")[-1].strip()

    # Assert that the response is a valid string and not empty
    assert isinstance(response, str), "Response must be a string."
    assert len(response) > 0, "Response should not be empty."
