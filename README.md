# Simple Chatbot Using GPT-2

This is a simple example of a chatbot using the GPT-2 model from the Hugging Face Transformers library. The bot takes user input and generates a response using GPT-2.

## Installation

Before running the script, make sure you have Python installed. Then, install the required packages:

```bash
pip install torch
pip install transformers
```
## Usage
To run the chatbot, simply execute the Python script:

```
python chatbot.py
```
## Code Overview
- Import required libraries: Torch and Transformers are imported at the top of the script.
- Initialize GPT-2 Model and Tokenizer: The GPT2LMHeadModel and GPT2Tokenizer are initialized from pre-trained "gpt2" versions.
- Chat Loop: A while loop facilitates the chat with the user. The loop does the following in each iteration:
- Takes the user input.
- Tokenizes the input and appends it to the chat history.
- Generates a response using the GPT-2 model.
- Decodes and prints the model's response.
- Appends the model's response to the chat history.
## Parameters
- max_length=1000: Maximum length of the sequence to be generated.
- pad_token_id=tokenizer.eos_token_id: Padding token for short sequences.
- no_repeat_ngram_size=3: To ensure that no 3-grams appear twice.
- top_k=50: The number of highest probability vocabulary tokens to keep for top-k filtering.
- top_p=0.7: Nucleus sampling: the cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.
- temperature=0.8: Controls the randomness of the output.
