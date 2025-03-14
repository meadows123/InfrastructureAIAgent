from llama_index.llms.ollama import Ollama
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen2.5")

result = model.invoke(input="hello world")
print(result)