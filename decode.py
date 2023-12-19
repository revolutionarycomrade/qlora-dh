from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/c/models/daryl149_llama-2-7b-chat-hf")

# Example tokenized input (replace this with your actual tokenized data)
tokenized_input = {
    "input_ids": [50256, 262, 517, 50256],  # This is just an example
    "attention_mask": [1, 1, 1, 1]  # This is just an example
}

# Convert the tokenized input back to text
decoded_text = tokenizer.decode(tokenized_input["input_ids"], skip_special_tokens=True)

print(f"Decoded text: {decoded_text}")
