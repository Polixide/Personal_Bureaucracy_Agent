"""Manual connection test for the local Ollama model."""

import ollama

MODEL_NAME = "qwen3:4b"  # Change this to another locally installed model.

if __name__ == "__main__":
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Reply only with: connection works"}],
    )
    print(response.message.content)
