import ollama

stream = ollama.chat(
    model="llama3.1:70b",
    messages=[{"role": "user", "content": "What is Ollama?"}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
