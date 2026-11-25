from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from config import vocab_size

special_token = ['<|unk|>']

# Create Tokenizer
tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

# Pre-tokenization
tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)

# Decoder
tokenizer.decoder = ByteLevelDecoder()

# Trainer
trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special_token, show_progress=True)

# Train Tokenizer
def train_tokenizer(file_path):

    tokenizer.train(
        files=[file_path],
        trainer=trainer
    )

    tokenizer.save("Tokenizer/bpetokenizer.json")

    print("Tokenizer training complete!")
    print("Vocabulary size:", tokenizer.get_vocab_size())


# Encoder
def encode(text):
    output = tokenizer.encode(text)
    return output.ids

# Decoder
def decode(token_ids):
    return tokenizer.decode(token_ids)

# Load Tokenizer
def load_tokenizer(file_path="Tokenizer/bpetokenizer.json"):

    global tokenizer

    tokenizer = Tokenizer.from_file(file_path)

    print("Tokenizer loaded successfully!")
    print("Vocabulary size:", tokenizer.get_vocab_size())

# Testing
if __name__ == "__main__":

    # dataset_path = "input/ri_data.txt"
    dataset_path = "input/claude_utf8.txt"

    train_tokenizer(dataset_path)

    test_text = "Fang Yuan obtained the immortal gu"

    encoded = encode(test_text)

    decoded = decode(encoded)

    print("\nOriginal text:")
    print(test_text)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(decoded)

    assert test_text == decoded
    
    print("\nEncoding-decoding test passed!")