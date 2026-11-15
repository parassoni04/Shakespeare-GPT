import json


def build_tokenizer(text):

    chars = sorted(list(set(text)))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return stoi, itos


def encode(text, stoi):

    return [stoi[c] for c in text]


def decode(tokens, itos):

    return ''.join([itos[i] for i in tokens])


def save_tokenizer(stoi, path="tokenizer.json"):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stoi, f)


def load_tokenizer(path="tokenizer.json"):

    with open(path, "r", encoding="utf-8") as f:
        stoi = json.load(f)

    itos = {int(v): k for k, v in stoi.items()}

    return stoi, itos