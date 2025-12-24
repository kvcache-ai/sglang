#!/usr/bin/env python3
"""
Simple script to chat with SGLang server
"""
import requests
import json

# Server configuration
SERVER_URL = "http://localhost:8173/v1/chat/completions"
MODEL_NAME = "DeepSeekV2-Lite-west"
LORA_PATH = "/mnt/data/lpl/test_adapter/Kllama_deepseekV2_WEST/checkpoint-1321_converted"

def chat(message, use_lora=False):
    """Send a chat message to the server"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": message}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    # Add LoRA name if requested
    # Use the lora name defined in --lora-paths (e.g., "lora0")
    if use_lora:
        payload["lora_path"] = "lora0"  # Use the lora name, not the full path

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        assistant_message = result["choices"][0]["message"]["content"]
        return assistant_message

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

def interactive_chat():
    """Interactive chat loop"""
    print("=== SGLang Server Chat ===")
    print(f"Server: {SERVER_URL}")
    print(f"Model: {MODEL_NAME}")
    print("\nType 'quit' or 'exit' to stop")
    print("Type 'lora' to toggle LoRA adapter\n")

    use_lora = False

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'lora':
            use_lora = not use_lora
            print(f"LoRA adapter: {'enabled' if use_lora else 'disabled'}")
            continue

        if not user_input:
            continue

        print("\nAssistant: ", end="", flush=True)
        response = chat(user_input, use_lora=use_lora)
        print(response)

if __name__ == "__main__":
    # Simple test
    print("Testing server connection...")
    response = chat("Hello")
    print(f"Response: {response}\n")

    # Start interactive chat
    interactive_chat()
