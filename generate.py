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
    'saved_parameters/checkpoint_final.pth',
    map_location=device,
    weights_only=True
)

config = parameters['config']

# Initialize architecture
model = GPTLanguageModel(
    vocab_size=vocab_size,
    embedding=embedding,
    block_size=block_size,
    n_layers=n_layers,
    n_heads=n_heads,
    p_dropout=p_dropout
).to(device)

# Load learned parameters
model.load_state_dict(parameters['model_state_dict'])

model.eval()

print("GPT model loaded successfully!")

while True:

    # prompt = input("\nEnter prompt (or 'exit'): ")
    prompt = "Fang Yuan understood his situation clearly, but even in the face of death his expression did not change"

    if prompt.lower() == "exit":
        break

    try:

        print("\nGenerated Text:\n")
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        for i in range(500):
            generated = model.generate(context, temperature=0.8)
            context = torch.tensor([encode(prompt + decode(generated[0].tolist()))], dtype=torch.long, device=device)
            print(decode(generated[0].tolist()), end = "")


    except KeyError as e:
        print(f"Character not found in vocabulary: {e}")

    break