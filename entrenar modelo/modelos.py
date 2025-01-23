import google.generativeai as genai

genai.configure(api_key="AIzaSyCW-_8k6OacvjTUSzDiKtijy6W2y1ZhaYk")

models = genai.list_models()
for m in models:
    print(m.name)
