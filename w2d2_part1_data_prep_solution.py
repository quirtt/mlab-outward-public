# %%
"""
# W2D2 - BERT Inference and Training

Yesterday, we wrote the code for BERT's architecture and were able to do inference using an already trained model. Today's material has four parts:

- Part 1: prepare a dataset for fine-tuning
- Part 2: fine-tune a pretrained BERT to do sentiment classification
- Part 3: prepare a dataset for pretraining
- Part 4: Train from scratch a tiny BERT on the masked language modeling task.

<!-- toc -->

## Readings

- [BERT Paper](https://arxiv.org/pdf/1810.04805.pdf) - focus on the details of pretraining, found primarily in Section 3.1 and Appendix A.

## HuggingFace Tokenizer Warning

You might see the warning "The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks" when re-running cells. Nothing bad will happen if you ignore this warning, other than that your tokenizer may run more slowly without the parallelism.

"""
# %%

import hashlib
import os
import re
import sys
import tarfile
from dataclasses import dataclass

import requests
import torch as t
import transformers
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm

MAIN = __name__ == "__main__"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/w2d2/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")
device = t.device("cuda" if t.cuda.is_available() else "cpu")
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)
# %%
"""
## Fine-Tuning BERT

Fine-tuning a pretrained model is awesome - it typically is much faster and more accurate than directly training on the task you care about, especially if your task has relatively few labels available. In our case we will be using the [IMDB Sentiment Classification Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

It's traditional to treat this as a binary classification task where each review is positive or negative. Today we're also going to predict the star rating review from 1 to 10, inclusive.

It's a bit redundant to train with both the star rating and the positive/negative labels as targets (you could just use the star rating), but we'll do it anyway to practice having multiple terms in the loss function.

There are a few ways to treat the star rating. One way is to have each rating be a category and use the regular cross entropy loss.

Exercise: what are the disadvantages of doing this?

<details>

<summary>Solution - disadvantages of cross-entropy for star rating</summary>

Cross entropy doesn't capture the intuition that the classes are ordered. Intuitively, we should penalize our model more for predicting 10 stars for a 1-star movie than predicting 2 stars.

</details>

Another way would be to treat the star rating as a continuous value, and use mean squared or mean absolute error. We'll do this today because it's simple and works well, but note that a more sophisticated approach like [ordinal regression](https://en.wikipedia.org/wiki/Ordinal_regression) could also be used.

### IMDB Dataset

Previously, we've used the `torchvision` package to download CIFAR10 for us. Today we'll load and process the training data ourselves to get an idea of what's involved.

Use [requests.get](https://requests.readthedocs.io/en/latest/user/quickstart/) to fetch the data and then write the `content` field of the response to disk. It's 82MB, so may take a few seconds depending on your connection. On future calls to the function, if the file already exists, your function should just read the local file instead of downloading the data again.

We've provided code that hashes the data using `hashlib.md5` and verifies that it matches a known good reference. Why is this a good practice?

<details>

<summary>Solution - Why Hash the Data?</summary>

We don't expect the file pointed to by the URL to change, and if it does we would like things to break loudly. At the very least, our training pipeline would no longer be reproducible, and it's possible that an adversary compromised the website and is supplying malicious data of some form. It's also possible that hardware errors could corrupt the data either on our local disk or the remote machine.

</details>
"""
# %%
def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    """SOLUTION"""
    if os.path.exists(path):
        return
    print("Downloading:", url)
    response = requests.get(url)
    with open(path, "wb") as f:
        f.write(response.content)


if MAIN:
    os.makedirs(DATA_FOLDER, exist_ok=True)
    expected_hexdigest = "7c2ac02c03563afcf9b574c7e56c153a"
    maybe_download(IMDB_URL, IMDB_PATH)
    with open(IMDB_PATH, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest


# %%
"""
Now we have a tar archive, which we can read using the standard library module [tarfile](https://docs.python.org/3/library/tarfile.html). Note the warning about extracting archives from untrusted sources.

Open the archive with `tarfile.open`, loop over the entries returned by `getmembers()` and use the `extractfile` method as appropriate to create a list of `Review`. A filename like `aclImdb/test/neg/127_3.txt` means it belongs to the test set, has a negative sentiment, has an id of 127 (we will ignore this), and was rated 3/10 stars.

You should have 25000 train and 25000 test entries - ignore the unlabeled folder.

This should take less than 10 seconds, but it's good practice to use tqdm to monitor your progress as most datasets will be much larger than this one.
"""
# %%
@dataclass(frozen=True)
class Review:
    split: str
    is_positive: bool
    stars: int
    text: str


def load_reviews(path: str) -> list[Review]:
    "SOLUTION"
    reviews = []
    tar = tarfile.open(path, "r:gz")
    for info in tqdm(tar.getmembers()):
        m = re.match(r"aclImdb/(train|test)/(pos|neg)/\d+_(\d+)\.txt", info.name)
        if m is not None:
            split, posneg, stars = m.groups()
            buf = tar.extractfile(info)
            assert buf is not None
            text = buf.read().decode("utf-8")
            reviews.append(Review(split, posneg == "pos", int(stars), text))
    return reviews


reviews = []
if MAIN:
    reviews = load_reviews(IMDB_PATH)
    assert sum(r.split == "train" for r in reviews) == 25000
    assert sum(r.split == "test" for r in reviews) == 25000

# %%
"""
## Data Visualization

Charles Babbage, the inventor of the first mechanical computer, was famously asked "Pray, Mr. Babbage, if you put into the machine wrong figures, will the right answers come out?"

200 years later, if you put wrong figures into the machine, the right answers still do not come out.

Inspecting the data before training can be tedious, but will catch many errors, either in your code or upstream, and allow you to validate or refute assumptions you've made about the data. Remember: "Garbage In, Garbage Out".

### Basic Inspection

Take some time now to do a basic inspection of your data. This should at minimum include:

- Plot the distribution of review lengths in characters.
    - Our BERT was only trained to handle 512 tokens maximum, so if we assume that a token is roughly 4 characters, we will have to truncate reviews that are longer than around 2048 characters.
    - Are positive and negative reviews different in length on average? If so, truncating would differentially affect the longer reviews.
- Plot the distribution of star ratings. Is it what you expected?

<details>
<summary>Star ratings distribution</summary>
That's right, there are no five or six star reviews in the dataset.
</details>

### Detailed Inspection

Either now, or later while your model is training, it's a worthwhile and underrated activity to do a more in-depth inspection. For a language dataset, some things I would want to know are:

- What is the distribution over languages? Many purportedly English datasets in fact have some of the data in other natural languages like Spanish, and computer languages like HTML tags.
    - This can cause bias in the results. Suppose that a small fraction of reviews were in Spanish, and purely by chance they have more positive/negative sentiment than the base rate. Our classifier would then incorrectly learn that Spanish words inherently have positive/negative sentiment.
    - Libraries like [Lingua](https://github.com/pemistahl/lingua-py) can (imperfectly) check for this.
- How are non-ASCII characters handled?
    - The answer is often "poorly". A large number of things can go wrong around quoting, escaping, and various text encodings. Spending a lot of time trying to figure out why there are way too many backslashes in front of your quotation marks is an Authentic ML Experience. Libraries like [`ftfy`](https://pypi.org/project/ftfy/) can be useful here.
- What data can you imagine existing that is NOT part of the dataset? Your neural network is not likely to generalize outside the specific distribution it was trained on. You need to understand the limitations of your trained classifier, and notice if you in fact need to collect different data to do the job properly:
    - What specific geographical area, time period, and demographic was the data sampled from? How does this compare to the deployment use case?
    - What filters were applied upstream that could leave "holes" in the distribution?
- What fraction of labels are objectively wrong?
    - Creating accurate labels is a laborious process and humans inevitably make mistakes. It's expensive to check and re-check labels, so most published datasets do contain incorrect labels.
    - Errors in training set labels can be mitigated through stronger regularization to prevent the model from memorizing the errors, or other techniques.
    - Most pernicious are errors in **test set** labels. Even a small percentage of these can cause us to select a model that outputs the (objectively mistaken) label over one that does the objectively right thing. The paper [Pervasive Label Errors in Test Sets
Destabilize Machine Learning Benchmarks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf) shows that these are more common than you might expect, and describes implications of this in more detail.

"""
# %%
if MAIN:
    if "SOLUTION":
        pos_lengths = [len(r.text) for r in reviews if r.is_positive]
        neg_lengths = [len(r.text) for r in reviews if not r.is_positive]
        fig, ax = plt.subplots()
        ax.hist(pos_lengths, bins=30, color="green", label="positive")
        ax.hist(neg_lengths, bins=30, color="red", label="negative")
        ax.set(xlabel="Characters in Review", ylabel="Count")
        ax.axvline(2048, color="red", label="Truncation point")
        fig.legend()

        pos_stars = [r.stars for r in reviews if r.is_positive]
        neg_stars = [r.stars for r in reviews if not r.is_positive]
        fig, ax = plt.subplots()
        ax.hist(pos_stars, bins=range(10), color="green", label="positive")
        ax.hist(neg_stars, bins=range(10), color="red", label="negative")
        ax.set(xlabel="Star rating", ylabel="Count")
        fig.legend()
# %%
"""
## Tokenization

When fine-tuning, you need to use the same tokenizer as was used for pre-training. The tokenizer already knows about the 512 token maximum, so it will take care of the truncation if we pass `truncation=True`. We also need to pad out the short reviews with a special padding token. In this case, `tokenizer.pad_token_id` is 0, but it's good practice to not assume this in our code.

It's most convenient to tokenize the dataset once and store the preprocessed data. Roughly, how large will our preprocessed dataset be?

<details>

<summary>Solution - Train Data Size</summary>

Naively, using int64 the tokens are 25,000 reviews * 512 tokens/review * 8 bytes/token or approximately 104MB. The labels are negligible.

We know the maximum token value is `tokenizer.vocab_size` or 28996, so it's safe to use int16 instead, decreasing our storage requirements by a factor of 4 to approximately 25MB. We would also expect this data to be very compressible, since there are long strings of padding tokens.

</details>

Implement `to_dataset`. Calling this function could take a minute, as tokenization requires a lot of CPU even with the efficient Rust implementation provided by HuggingFace. We aren't writing our own tokenizer because it would be extremely slow to do it in pure Python.

Note that you really don't want to have to do long-running tasks like this repeatedly. It's always a good idea to store the preprocessed data on disk and load that on future runs. Then you only have to re-run it if you want to preprocess the data in some different way.
"""
# %%
def to_dataset(tokenizer, reviews: list[Review]) -> TensorDataset:
    """Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    """
    "SOLUTION"
    out = tokenizer(
        [r.text for r in reviews],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
    )
    return TensorDataset(
        out["input_ids"],
        out["attention_mask"],
        t.tensor([r.is_positive for r in reviews]),
        t.tensor([r.stars for r in reviews]),
    )


if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"])
    test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"])
    t.save((train_data, test_data), SAVED_TOKENS_PATH)


"""
## Bonus

Go on to Step 2, but if you have time at the end, you can come back and try the bonus exercise for this part.

### Better Truncation

We arbitrarily kept the first `max_length` tokens and truncated the rest. Is this strategy optimal? If you read some of the reviews, a common pattern is to sum up and conclude the review in the final 1-2 sentences.

This suggests that we might do better by keeping some number of tokens at the end and truncating the middle instead. Implement this strategy and see if you can measure a difference in accuracy.

### Better Data Cleaning

You may have noticed that paragraph breaks are denoted by the string "< br / > < br / >". We might suppose that these are not very informative, and it would be better to strip them out. Particularly, if we are truncating our reviews then we would rather have 10 tokens of text than this in the truncated sequence. Replace these with the empty string (in all splits of the data) and see if you can measure a difference in accuracy.
"""

# %%
