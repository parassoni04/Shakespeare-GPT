import torch

batch_size = 32
block_size = 512
embedding = 256
n_heads = 8
n_layers = 8
# hidden_dim = 512
learning_rate = 0.0003
min_lr = 0.00003
epochs = 10000
warmup_steps = 200
eval_iters = 100

weight_decay = 0.05
p_dropout = 0.3
split_ratio = 0.9

vocab_size = 256

device = "cuda" if torch.cuda.is_available() else "cpu"