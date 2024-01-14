# %%
"""
# W2D2 Part 3 - WikiText Data Prep

Now we'll prepare text data to train a BERT from scratch! The largest BERT would require days of training and a large training set, so we're going to train a tiny BERT on a small training set: [WikiText](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/). This comes in a small version WikiText-2 (the 2 means 2 million tokens in the train set) and a medium version WikiText-103 with 103 million tokens. For the sake of fast feedback loops, we'll be using the small version but in the bonus you'll be able to use the medium version with the same code by changing the `DATASET` variable below. Both versions consist of text taken from Good and Featured articles on Wikipedia.

<!-- toc -->
"""
# %%

import hashlib
import os
import sys
import zipfile

import torch as t
import transformers
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

import w2d2_test
from w2d2_part1_data_prep_solution import maybe_download

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
DATA_FOLDER = "./data/w2d2"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")
IS_CI = os.getenv("IS_CI")


# %%
"""
## Data Preparation

Since we aren't using pretrained weights, we don't have to match the tokenizer like we did when fine-tuning. We're free to use any tokenization strategy we want.

### Vocab Size

For example, we could use a smaller vocabulary in order to save memory on the embedding weights. It's straightforward to train a new tokenizer, but for the sake of time we'll continue using the existing tokenizer and its vocabulary size.

### Context Length

We're also free to use a shorter or longer context length, and this doesn't require training a new tokenizer. The only thing that this really affects is the positional embeddings. For a fixed compute budget, it's not obvious whether we should decrease the context length or increase it.

The computational cost of attention is quadratic in the context length, so decreasing it would allow us to use more compute elsewhere, or just finish training earlier. Increasing it would allow for longer range dependencies; for example, our model could learn that if a proper noun appears early in a Wikipedia article, it's likely to appear again.

The authors pretrain using a length of 128 for 90% of the steps, then use 512 for the rest of the steps. The idea is that early in training, the model is mostly just learning what tokens are more or less frequent, and isn't able to really take advantage of the longer context length until it has the basics down. Since our model is small, we'll do the simple thing to start: a constant context length of 128.

### Data Inspection

Run the below cell and inspect the text. It is one long string, so don't try to print the whole thing. What are some things you notice?

<details>

<summary>Spoiler - Things to Notice</summary>

There is still some preprocessing done even though this is allegedly "raw" text. For example, there are spaces before and after every comma.

There are Japanese characters immediately at the start of the training set, which in a real application we might want to do something with depending on our downstream use case.

There is some markup at least for section headings. Again, this might be something we'd want to manually handle.

</details>

### Use of zipfile library

It's important to know that the `zipfile` standard library module is written in pure Python, and while this makes it portable it is extremely slow as a result. It's fine here, but for larger datasets, definitely don't use it - it's better to launch a subprocess and use an appropriate decompression program for your system like `unzip` or `7-zip`.

"""
# %%
if MAIN and not IS_CI:
    path = os.path.join(DATA_FOLDER, DATASETS[DATASET])
    maybe_download(BASE_URL + DATASETS[DATASET], path)
    expected_hexdigest = {
        "103": "0ca3512bd7a238be4a63ce7b434f8935",
        "2": "f407a2d53283fc4a49bcff21bc5f3770",
    }
    with open(path, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest[DATASET]

# %%
if MAIN:
    print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

if MAIN and not IS_CI:
    z = zipfile.ZipFile(path)

    def decompress(split: str) -> str:
        return z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8")

    train_text = decompress("train").splitlines()
    val_text = decompress("valid").splitlines()
    test_text = decompress("test").splitlines()

# %%
"""
### Preprocessing

To prepare data for the next sentence prediction task, we would want to use a library like [spaCy](https://spacy.io/) to break the text into sentences - it's tricky to do this yourself in a robust way. We'll ignore this task and just do masked language modelling today.

Right now we have a list of lines, but we need (batch, seq) of tokens. We could use padding and truncation as before, but let's play with a different strategy:

- [Call the tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__) on the list of lines with `truncation=False` to obtain lists of tokens. These will be of varying length, and you'll notice some are empty due to blank lines.
- Build one large 1D tensor containing all the tokens in sequence
- Reshape the 1D tensor into (batch, sequence).

Instead of padding, we'll just discard tokens at the very end that would form an incomplete sequence. This will only discard up to (max_seq - 1) tokens, so it's negligible.

This is nice because we won't waste any space or compute on padding tokens, and we don't have to truncate long lines. Some fraction of sequences will contain both the end of one article and the start of another, but this won't happen too often and there will be clues the model can use, like the markup for a heading appearing.

Note we won't need the attention mask, because we're not using padding. We'll also not need the `token_type_ids` or the special tokens CLS or SEP. You can pass arguments into the tokenizer to prevent it from returning these.

Don't shuffle the tokens here. This allows us to change the context length at load time, without having to re-run this preprocessing step.

You can ignore a warning about 'Token indices sequence length is longer than the specified maximum sequence length' - this is expected.

<details>

<summary>Why pass lists of lines instead of one big string</summary>

The "fast" version of the tokenizer is written in Rust and will spawn threads to process the lines in parallel. If you only pass one big string, it doesn't know where to split the string and will only use one thread.

</details>

<details>

<summary>I'm doing WikiText-103; how long should tokenization take?</summary>

It takes about 2 minutes on my machine to tokenize the full WikiText-103.

</details>
"""
# %%
def tokenize_1d(tokenizer, lines: list[str], max_seq: int) -> t.Tensor:
    """Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype
    """
    "SOLUTION"
    tokens = tokenizer(
        lines,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=False,
    )["input_ids"]

    out = t.cat([t.tensor(line_tok, dtype=t.int64) for line_tok in tokens])
    n = len(out) // max_seq * max_seq
    return rearrange(out[:n], "(b seq) -> b seq", seq=max_seq)


if MAIN and not IS_CI:
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq)
    print("Training data shape is: ", train_data.shape)

    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)

# %%
"""
### Masking

Implement `random_mask`, which we'll call during training on each batch.

Hint: ensure any tensors that you create are on the same device as `input_ids`. When sampling a random token, sample uniformly at random from [0..vocabulary size).

<details>

<summary>Help, I'm confused about the indexing!</summary>

I found it easier to flatten batch and seq together into one dimension, so that indexes can be a simple integer instead of having to index into multiple dimensions. You can use `randperm` to select random indexes, and then partition the indexes among the possible modifications.

Unflatten at the end.

</details>

<details>

<summary>Is there anything special or optimal about the numbers 15%, 80%, and 10%?</summary>

No, these are just some ad-hoc numbers that the BERT authors chose. The paper [Should You Mask 15% in Masked Language Modelling](https://arxiv.org/pdf/2202.08005.pdf) suggests that you can do better.

</details>

"""
# %%
def flat(x: t.Tensor) -> t.Tensor:
    """Helper function for combining batch and sequence dimensions."""
    return rearrange(x, "b s ... -> (b s) ...")


def unflat(x: t.Tensor, max_seq: int) -> t.Tensor:
    """Helper function for separating batch and sequence dimensions."""
    return rearrange(x, "(b s) ... -> b s ...", s=max_seq)


def random_mask(
    input_ids: t.Tensor,
    mask_token_id: int,
    vocab_size: int,
    select_frac=0.15,
    mask_frac=0.80,
    random_frac=0.10,
) -> tuple[t.Tensor, t.Tensor]:
    """Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.

    input_ids: (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    """
    "SOLUTION"
    B, S = input_ids.shape
    out_flat = flat(input_ids.clone())
    n = len(out_flat)
    n_select = int(n * select_frac)
    idx = t.randperm(n, device=input_ids.device)
    start = 0

    # Masking
    n_mask = int(n_select * mask_frac)
    mask_idx = idx[start : start + n_mask]
    out_flat[mask_idx] = mask_token_id
    start += n_mask

    # Random token
    n_rand = int(n_select * random_frac)
    rand_idx = idx[start : start + n_rand]
    out_flat[rand_idx] = t.randint(0, vocab_size, rand_idx.shape, device=input_ids.device)
    # TBD lowpri: is this actually the right thing to do? Should we ever sample CLS/SEP/UNK/MASK?
    start += n_rand

    was_selected_flat = t.zeros_like(out_flat)
    was_selected_flat[idx[:n_select]] = 1
    return unflat(out_flat, S), unflat(was_selected_flat, S)


if MAIN:
    w2d2_test.test_random_mask(random_mask, input_size=10000, max_seq=max_seq)


# %%

"""
### Loss Function

Exercise: what should the loss be if the model is predicting tokens uniformly at random? Use the [formula for discrete cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy).

<details>

<summary>Solution - Random Cross-Entropy Loss</summary>

Let $k$ be the vocabulary size. For each token to be predicted, the expected probability assigned by the model to the true token is $1/k$. Plugging this into the cross entropy formula gives an expected loss of $log(k)$, which for $k=28996$ is about 10.2.

Importantly, this is the loss per predicted token, and we have to decide how to aggregate these over the batch and sequence dimensions.

For a batch, we can aggregate the loss per token in any way we want, as long as we're clear and consistent about what we're doing. Taking the mean loss per predicted token has the nice property that we can compare models with a different number of predicted tokens per batch.

</details>

Now, find the cross-entropy loss of the distribution of unigram frequencies. This is the loss you'd see when predicting words purely based on word frequency without the context of other words. During pretraining, your model should reach this loss very quickly, as it only needs to learn the final unembedding bias to predict this unigram frequency.
"""
# %%
if MAIN:
    if "SOLUTION":
        sample = train_data[:50]
        B, S = sample.shape
        counts = t.bincount(train_data.flatten(), minlength=tokenizer.vocab_size)
        freqs = counts.float() / t.tensor(train_data.shape).prod()
        top_tokens = t.argsort(counts, descending=True)[:10]
        print("Top tokens: ")
        for tok in top_tokens:
            print(tokenizer.decode(tok), f"{freqs[tok]:.04f}")
        logprobs = repeat(t.log(freqs), "v -> bs v", bs=B * S)
        freq_loss = F.cross_entropy(logprobs, flat(sample))
        print(f"Sample cross-entropy loss of unigram frequencies: {freq_loss:.2f}")
        ce = freqs * freqs.log()
        print(
            f"Dataset computed cross-entropy loss of unigram frequencies: ",
            -ce[freqs > 0].sum().item(),
        )

# %%
"""
### Cross Entropy of MLM

For our loss function, we only want to sum up the loss at the tokens that were chosen with probability `select_frac`. As a reminder, when a token is selected, that input token could be replaced with either [MASK], a random token, or left as-is and the target is the original, unmodified input token.

Write a wrapper around [torch.nn.functional.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy) that only looks at the selected positions. It should output the total loss divided by the number of selected tokens.

`torch.nn.functional.cross_entropy` divides by the batch size by default, which means that the magnitude of the loss will be larger if there are more predictions made per batch element. We will want to divide by the number of tokens predicted: this ensures that we can interpret the resulting value and we can compare models with different sequence lengths.

<details>

<summary>I'm confused about how to do this!</summary>

Again, it's simpler to flatten batch and seq together so that you only have to think about one spatial dimension.

You can either slice both input and target arguments to `cross_entropy` so that you're only passing the contributing parts, or you can make use of the `ignore_index` keyword argument of `cross_entropy` and set the target to -100 when it shouldn't contribute.

</details>
"""
# %%
def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    """
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    """
    "SOLUTION"
    idx = flat(was_selected).nonzero().flatten()
    inp = flat(pred)[idx]
    target = flat(target)[idx]
    loss = F.cross_entropy(inp, target)
    return loss


# %%
if MAIN:
    w2d2_test.test_cross_entropy_selected(cross_entropy_selected)

if MAIN and not IS_CI:
    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    masked, was_selected = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")


# %%
"""
## Bonus

Go on to step 4!

### Context Length Experimentation

Play with a shorter context length and observe the difference in training speed. Does it decrease performance, or was the small model unable to make much use of the longer context length anyway?

### Whole Word Masking

The [official BERT repo](https://github.com/google-research/bert) has a README section on a different way of computing the mask. Try implementing it and see if you get any benefit.
"""
