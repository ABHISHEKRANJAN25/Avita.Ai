from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer from Hugging Face
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_response(prompt, max_length=100):
    """
    Generate a response based on the given prompt using a pre-trained GPT model.
    
    :param prompt: str, the input text provided by the user.
    :param max_length: int, the maximum length of the response.
    :return: str, the generated response.
    """
    # Encode the prompt into tokens
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a response using the model
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens back into a string
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

if __name__ == "__main__":
    print("Hello! I'm a simple GPT-based chatbot. Type 'quit' to end the chat.")

    while True:
        # Get user input
        user_input = input("You: ")
        
        # Exit the loop if the user types 'quit'
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        # Generate a response
        response = generate_response(user_input)
        
        # Display the response
        print(f"Chatbot: {response}")
