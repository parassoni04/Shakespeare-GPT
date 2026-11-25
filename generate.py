import torch
from model import GPTLanguageModel
from tokenizers import Tokenizer
from bpe_tokenizer import load_tokenizer
from config import *

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer
tokenizer = Tokenizer.from_file("Tokenizer/bpetokenizer.json")
# vocab_size = tokenizer.get_vocab_size()

print("Tokenizer Loaded")
print("Vocabulary Size : ", tokenizer.get_vocab_size())


def encode(text):

    return tokenizer.encode(text).ids


def decode(token_ids):

    return tokenizer.decode(token_ids)

# Load checkpoint
parameters = torch.load(
    'saved_parameters/checkpoint2500.pth',
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
    n_heads=config['n_heads'],
    p_dropout=p_dropout
).to(device)

print("If vocab size same : ",tokenizer.get_vocab_size() == config['vocab_size'])

# Load learned parameters
model.load_state_dict(parameters['model_state_dict'])

model.eval()

print("GPT model loaded successfully!")

while True:

    # prompt = input("\nEnter prompt (or 'exit'): ")
    prompt = "Fang Yuan understood his situation clearly"

    if prompt.lower() == "exit":
        break

    try:

        print("\nGenerated Text:\n")
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        generated_tokens = []
        for i in range(500):
            next_token = model.generate(context, temperature = temperature)
            context = torch.cat((context, next_token), dim = 1)
            generated_tokens.append(next_token.item())
            print(decode([next_token.item()]), end = "", flush = True)


    except KeyError as e:
        print(f"Character not found in vocabulary: {e}")

    break