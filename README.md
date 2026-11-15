# Shakespeare GPT

A character-level GPT-style language model implemented from scratch
using **PyTorch**, trained on Shakespeare's works.

The project explores the core components behind decoder-only Transformer
language models: tokenization, causal self-attention, positional
embeddings, feed-forward networks, training, and autoregressive text
generation.

> **Project status:** Experimental. The model can generate
> Shakespeare-like text, but its output is not expected to be coherent
> or historically accurate.

## Features

-   Character-level tokenizer built from the training corpus
-   Decoder-only Transformer architecture
-   Causal (masked) self-attention
-   Multi-head attention
-   Positional embeddings
-   Feed-forward networks and residual connections
-   Training and validation loss evaluation
-   Autoregressive text generation with temperature sampling
-   Separate training and inference scripts

## Project Structure

``` text
shakespeare-gpt/
├── config.py             # Model and training configuration
├── tokenizer.py          # Character-level tokenizer
├── model.py              # GPT and Transformer components
├── train.py              # Dataset preparation and training loop
├── generate.py           # Load a checkpoint and generate text
├── input.txt             # Training corpus (if included)
├── tokenizer.json        # Saved character-to-index mapping
├── saved_parameters/     # Model checkpoints (not committed to Git)
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Overview

This project uses a **character-level** language model. Each unique
character in the training corpus is assigned an integer ID. The model
learns to predict the next character based on the preceding context.

The architecture follows the decoder-only Transformer design:

1.  Token embeddings convert character IDs into vectors.
2.  Positional embeddings provide information about token positions.
3.  Transformer blocks process the sequence using causal self-attention
    and feed-forward layers.
4.  A language-model head produces logits for the next-character
    prediction task.
5.  During generation, the model repeatedly samples the next character
    and appends it to the context.

### Experiment Configuration

The following are the configuration values used in the reported
experiment. Confirm them against `config.py` before publishing changes
to the experiment.

  Setting                                          Value
  ------------------------------- ----------------------
  Tokenization                           Character-level
  Vocabulary size                                     65
  Training corpus size              1,115,394 characters
  Train/validation split                       90% / 10%
  Batch size                                          64
  Context length (`block_size`)                      256
  Embedding dimension                                384
  Attention heads                                      6
  Transformer layers                                   6
  Learning rate                                   0.0003
  Minimum learning rate                          0.00003
  Warmup steps                                       200
  Parameter Count                         10788929 ~ 10M


## Results

In one reported training run, the losses were approximately:

  Metric              Loss
  ----------------- ------
  Training loss       1.29
  Validation loss     1.50

The validation loss began increasing while training loss continued to
improve, indicating overfitting. These figures are from an experimental
run and may change as the architecture and training process are
improved.

## Installation

Clone the repository:

``` bash
git clone https://github.com/parassoni04/Shakespeare-GPT.git
cd Shakespeare-GPT
```

Create and activate a virtual environment (recommended):

``` bash
python -m venv venv
```

On Windows:

``` bash
venv\Scripts\activate
```

On macOS/Linux:

``` bash
source venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Training

Make sure the training corpus is available at the path expected by
`train.py`. Then run:

``` bash
python train.py
```

The training script should save the tokenizer and model checkpoint.
Check the paths configured in `config.py` or `train.py`.

## Text Generation

After training, run:

``` bash
python generate.py
```

Enter a starting prompt when requested. The generation script loads the
saved tokenizer and model checkpoint, then generates text from the
prompt.

If you download or use a checkpoint separately, make sure it was created
with the same model configuration and tokenizer mapping used by the
generation script.

## Example Output

prpmpt : The sun is bright

The sun is bright lips of his life,
But shall said like a country in the least,
Ere a king the embroke of their season's sea,
To day the claughter for their offices of death.
I say the dereath of did black and loves to hear
Against the charged from that for where short:
My brother dishonour hearts from that woold of him;
Doth be, am bagking back, when some for hither,
Someting that the malicy, look that men,
What labide we wark the man, how my bloods
Where to straight these is the easters arted of?

## Limitations and Future Improvements

-   Character-level tokenization is simple, but it is less efficient
    than subword tokenization.
-   Generated text can be repetitive, inconsistent, or grammatically
    incorrect.
-   The model may overfit the training corpus.
-   Training quality depends on the dataset, model configuration, and
    training duration.

Potential improvements:

-   Experiment with dropout and weight decay.
-   Save the checkpoint with the best validation loss.
-   Add top-k or top-p sampling.
-   Compare different context lengths and model sizes.
-   Track training and validation loss over time.
-   Explore subword tokenization.

## Technologies

-   Python
-   PyTorch

## Acknowledgements

This project was developed as a learning exercise to understand and
implement the fundamentals of GPT-style language models and Transformer
architectures. If parts of the implementation were adapted from
tutorials or other learning resources, credit those resources here.