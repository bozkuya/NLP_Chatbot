import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chat():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_text = "Hello, how can I assist you today?"
    # Initialize chat history
    chat_history_ids = tokenizer.encode(input_text, return_tensors="pt")

    print("ChatGPT: Hello, how can I assist you today?")

    while True:
        # Get user input
        user_input = input("You: ")

        # Encode user input and append to chat history
        new_input_ids = tokenizer.encode(user_input, return_tensors="pt")
        chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

        # Generate a response from the model
        with torch.no_grad():
            outputs = model.generate(
                chat_history_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                top_k=50,
                top_p=0.7,
                temperature=0.8,
                do_sample=True
            )

        # Get the predicted text
        chat_output = tokenizer.decode(outputs[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

        # Print the chat history
        print(f"ChatGPT: {chat_output}")

        # Append the model's response to chat history
        chat_history_ids = torch.cat([chat_history_ids, outputs[:, chat_history_ids.shape[-1]:]], dim=-1)

        # Limit the chat history to a reasonable length to prevent memory issues
        if chat_history_ids.shape[-1] > 1000:
            chat_history_ids = chat_history_ids[:, -1000:]

if __name__ == "__main__":
    chat()