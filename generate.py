import torch
from model import GPTLanguageModel
from tokenizer import load_tokenizer

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer
stoi, itos = load_tokenizer()

def encode(text):
    return [stoi[c] for c in text]

def decode(tokens):
    return ''.join([itos[i] for i in tokens])

# Load checkpoint
parameters = torch.load(
    'saved_parameters/gpt_v2_10M.pth',
    map_location=device,
    weights_only=True
)

config = parameters['config']

# Initialize architecture
model = GPTLanguageModel(
    vocab_size=config['vocab_size'],
    embedding=config['embedding'],
    block_size=config['block_size'],
    n_layers=config['n_layers'],
    n_heads=config['n_heads']
).to(device)

# Load learned parameters
model.load_state_dict(parameters['model_state_dict'])

model.eval()

print("GPT model loaded successfully!")

while True:

    prompt = input("\nEnter prompt (or 'exit'): ")

    if prompt.lower() == "exit":
        break

    try:
        context = torch.tensor(
            [encode(prompt)],
            dtype=torch.long,
            device=device
        )

        generated = model.generate(
            context,
            max_new_tokens=500,
            temperature=0.8
        )

        print("\nGenerated Text:\n")
        print(decode(generated[0].tolist()))

    except KeyError as e:
        print(f"Character not found in vocabulary: {e}")