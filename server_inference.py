import openai
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1/"

model = "gemma-3-1b-it"
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "give me code to print 'hello world' in python"},
]

prompt = tokenizer.apply_chat_template(
    chat, 
    tokenize=False, 
    add_generation_prompt=True, # Recommended to signal the model to start generating
)


# create a completion
completion = openai.completions.create(model=model, prompt=prompt, max_tokens=200, temperature=1)
# print the completion
print(prompt + completion.choices[0].text)

