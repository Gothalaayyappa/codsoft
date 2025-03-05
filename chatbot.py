def simple_chatbot(user_input):
    responses = {
        "hello": "Hi there!",
        "how are you": "I'm doing well, thank you!",
        "what is your name": "I'm a simple chatbot.",
        "bye": "Goodbye!",
    }

    user_input = user_input.lower()

    if user_input in responses:
        return responses[user_input]
    else:
        return "I'm sorry, I don't understand."

# Example usage
user_input = input()
while(user_input != "bye"):
    print(simple_chatbot(user_input))
    user_input = input()
print(simple_chatbot(user_input))