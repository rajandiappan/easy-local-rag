import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as infile:
            return infile.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Function to get relevant context from the vault based on user input
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    # Input validation
    if not isinstance(rewritten_input, str) or not rewritten_input.strip():
        print("Invalid input")
        return []
    
    if vault_embeddings.nelement() == 0:
        print("Empty vault embeddings")
        return []
        
    try:
        # Get embeddings with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                input_embedding = ollama.embeddings(
                    model='mxbai-embed-large', 
                    prompt=rewritten_input
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to get embeddings after {max_retries} attempts")
                    return []
                time.sleep(1)
        
        # Convert to tensor and normalize dimensions
        input_tensor = torch.tensor(input_embedding["embedding"], dtype=torch.float32)
        input_tensor = input_tensor.unsqueeze(0) if len(input_tensor.shape) == 1 else input_tensor
        
        # Ensure same device
        input_tensor = input_tensor.to(vault_embeddings.device)
        
        # Calculate similarities
        cos_scores = torch.cosine_similarity(input_tensor, vault_embeddings)
        top_k = min(top_k, len(cos_scores))
        top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
        
        return [vault_content[idx].strip() for idx in top_indices]
        
    except Exception as e:
        print(f"Error in context retrieval: {e}")
        return []

def rewrite_query(user_input_json, conversation_history, ollama_model):
    try:
        if not isinstance(user_input_json, str):
            print("Invalid input: expected JSON string")
            return json.dumps({"Query": "", "Rewritten Query": ""})
            
        parsed = json.loads(user_input_json)
        query = parsed.get("Query", "")
        
        return json.dumps({
            "Query": query,
            "Rewritten Query": query  # For now, return original query
        })
    except json.JSONDecodeError:
        print("Invalid JSON input")
        return json.dumps({"Query": "", "Rewritten Query": ""})
    except Exception as e:
        print(f"Error in rewrite_query: {e}")
        return json.dumps({"Query": "", "Rewritten Query": ""})

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, ollama_model, conversation_history):
    try:
        conversation_history.append({"role": "user", "content": user_input})
        
        if len(conversation_history) > 1:
            query_json = {
                "Query": user_input,
                "Rewritten Query": ""
            }
            
            rewritten_query_json = rewrite_query(json.dumps(query_json), conversation_history, ollama_model)
            
            try:
                rewritten_query_data = json.loads(rewritten_query_json)
                rewritten_query = rewritten_query_data.get("Rewritten Query", user_input)
            except json.JSONDecodeError:
                print("Failed to parse rewritten query")
                rewritten_query = user_input
                
            print(PINK + "Original Query: " + user_input + RESET_COLOR)
            print(PINK + "Rewritten Query: " + rewritten_query + RESET_COLOR)
        else:
            rewritten_query = user_input
        
        relevant_context = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
        if relevant_context:
            context_str = "\n".join(relevant_context)
            print("Context Pulled from Documents: \n\n" + CYAN + context_str + RESET_COLOR)
        else:
            print(CYAN + "No relevant context found." + RESET_COLOR)
        
        user_input_with_context = user_input
        if relevant_context:
            user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
        
        conversation_history[-1]["content"] = user_input_with_context
        
        messages = [
            {"role": "system", "content": system_message},
            *conversation_history
        ]
        
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages,
            max_tokens=2000,
        )
        
        conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in ollama_chat: {e}")
        return {"role": "assistant", "content": "Sorry, I encountered an error processing your request."}

# Parse command-line arguments
print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Ollama Chat")
parser.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
args = parser.parse_args()

# Configuration for the Ollama API client
print(NEON_GREEN + "Initializing Ollama API client..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

# Load the vault content
print(NEON_GREEN + "Loading vault content..." + RESET_COLOR)
vault_content = []
if os.path.exists("vault.txt"):
    with open("vault.txt", "r", encoding='utf-8') as vault_file:
        vault_content = vault_file.readlines()

# Generate embeddings for the vault content using Ollama
print(NEON_GREEN + "Generating embeddings for the vault content..." + RESET_COLOR)
vault_embeddings = []
for content in vault_content:
    response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
    vault_embeddings.append(response["embedding"])

# Convert to tensor and print embeddings
print("Converting embeddings to tensor...")
vault_embeddings_tensor = torch.tensor(vault_embeddings) 
print("Embeddings for each line in the vault:")
print(vault_embeddings_tensor)

# Conversation loop
print("Starting conversation loop...")
conversation_history = []
system_message = "You are a cybersecurtiy expert extracting the most useful requirements from a given text. Also bring in extra relevant infromation to the user query from outside the given context."

while True:
    user_input = input(YELLOW + "Ask a query about your documents (or type 'quit' to exit): " + RESET_COLOR)
    if user_input.lower() == 'quit':
        break
    
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, args.model, conversation_history)
    print(NEON_GREEN + "Response: \n\n" + response + RESET_COLOR)
