import ollama
from langchain_ollama import OllamaEmbeddings

response = ollama.generate(
    model="phi4:latest",
    prompt="how are you?"
)

print(response['response'])

embedding = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

vector = embedding.embed_query("Hello, how are you?")
print(vector)