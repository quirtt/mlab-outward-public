# %%
"""
# W2D2 Part 2 - BERT Fine-Tuning

<!-- toc -->
"""


# %%
import os
import sys
import time
from dataclasses import dataclass

import torch as t
import transformers
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

import wandb
from w2d1_solution import BertCommon, BertConfig, load_pretrained_weights
from w2d2_part1_data_prep_solution import DATA_FOLDER, SAVED_TOKENS_PATH

MAIN = __name__ == "__main__"
device = t.device("cuda" if t.cuda.is_available() else "cpu")
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)

if MAIN:
    train_data, test_data = t.load(SAVED_TOKENS_PATH)
    bert_config = BertConfig()

# %%
"""
### BertClassifier

Now we'll set up our BERT to do classification, starting with the `BertCommon` from before and adding a few layers on the end:

- Use only the output logits at the first sequence position (index 0).
- Add a dropout layer with the same dropout probability as before.
- Add a `Linear` layer from `hidden_size` to `2` for the classification as positive/negative.
- Add a `Linear` layer from `hidden_size` to `1` for the star rating.
- By default, our star rating Linear layer is initialized to give inputs of roughly mean 0 and std 1. Multiply the output of this layer by 5 and add 5 to bring these closer to the 1-10 output we want; this isn't strictly necessary but helps speed training.
"""
# %%
@dataclass(frozen=True)
class BertClassifierOutput:
    """The output of BertClassifier."""

    is_positive: t.Tensor
    star_rating: t.Tensor


class BertClassifier(nn.Module):
    def __init__(self, config: BertConfig):
        "SOLUTION"
        super().__init__()
        self.common = BertCommon(config)
        self.dropout = nn.Dropout(config.dropout)
        self.pos_linear = nn.Linear(config.hidden_size, 2)
        self.star_linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: t.Tensor, one_zero_attention_mask: t.Tensor) -> BertClassifierOutput:
        "SOLUTION"
        input_ids = self.common(input_ids, one_zero_attention_mask=one_zero_attention_mask)
        input_ids = input_ids[:, 0]
        input_ids = self.dropout(input_ids)
        is_positive = self.pos_linear(input_ids)
        star_rating = 5 * rearrange(self.star_linear(input_ids), "b 1 -> b") + 5
        return BertClassifierOutput(is_positive, star_rating)


# %%
"""
## Training Loop

Load your `BertLanguageModel` from yesterday and use the pretrained `BertCommon` inside it for the initial parameters of your `BertClassifier`.

Copy over a training loop from before and modify it.

### Training All Parameters

When fine-tuning a language model, ensure all the parameters have `requires_grad=True`. This is different from fine-tuning an image model, where you typically "freeze" (`requires_grad=False`) the existing layers and just train your new layers.

### Learning Rate

The learning rate for fine-tuning should be much lower than when training from scratch. In Appendix A.3 of the [BERT Paper](https://arxiv.org/pdf/1810.04805.pdf), they suggest a learning rate for Adam between 2e-5 and 5e-5. I found that even 2e-5 was too high for this problem, and that 1e-5 worked well.

### Loss Functions

Use `torch.nn.CrossEntropyLoss` for the classification loss. For the star loss, empirically `F.l1_loss` works well. When you have multiple loss terms, you usually need to weight them by importance so their scales aren't too different. A default parameter for this is part of the training config.

### Gradient Clipping

Especially early in training, some batches can have very large gradients, like more than 1.0. The resulting large parameter updates can break training. To work around this, you can manually limit the size of gradients using `t.nn.utils.clip_grad_norm_`. Generally, a limit of 1.0 works decently.

### Batch Size

For a model the size of BERT, you typically want the largest batch size that fits in GPU memory. I found that a batch size of 8 used around 8GB of GPU memory, and a batch size of 16 used about 12GB of GPU memory. The BERT paper suggests a batch size of 16, so if your GPU doesn't have enough memory you could use a smaller size, accumulate your gradients, and call `optimizer.step` every second batch. In a later day, we'll learn how to use multiple GPUs instead.

### Optimizer

I found that `t.optim.AdamW` worked well.

### Logging

Send detailed information to Weights and Biases. Sending too much information too often can slow down training, but something I would recommend is every few batches, periodically decoding some training data back to text and logging the text and model's predictions using `wandb.Table`.

<details>

<summary>Training isn't converging and I don't know why!</summary>

- Double check that your BertClassifier is actually using the pretrained weights and not random ones.
- The classification loss for positive/negative should be around `log(2)` before any optimizer steps are taken, because the model is predicting randomly. If this isn't the case, there might be a bug in your loss calculation.
- Try decoding a batch from your DataLoader and verify that the labels match up and the tokens and padding are right. It should be [CLS], the review, [SEP], and then [PAD] up to the end of the sequence.
- Try using an even smaller learning rate to see if this affects the loss curve. It's usually better to have a learning rate that is too low and spend more iterations reaching a good solution than to use one that is too high, which can cause training to not converge at all.
- If your model is predicting all 1 or all 0, this can be a helpful thing to investigate.
- Check your model output for `NaN` and if you find any, use hooks to track down where it comes from.
- It may just be a bad seed. The paper [On the Stability of Fine-Tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/pdf/2006.04884.pdf) notes that random seed can make a large difference to the results.
</details>

<details>

<summary>My loss seems too good to be true!</summary>

This can happen if your training set isn't shuffled, and the model learns it can always predict a constant label. This can also happen if you've mixed up some tensors and the model is getting the labels as input.

</details>

On a V100, my model was able to reach 0.20 classification loss after 2000 training examples, with slight improvement after that, corresponding to 92% accuracy.

State of the art for this problem is around [96% accuracy as of 2022](https://paperswithcode.com/sota/sentiment-analysis-on-imdb).

"""
# %%
def train(tokenizer, config_dict: dict) -> BertClassifier:
    "SOLUTION"
    DEBUG_CHECK_NANS = False
    wandb.init(project="w2d2_imdb", config=config_dict)
    config = wandb.config
    n = len(train_data)
    perm = t.randperm(n)[: config.num_steps]
    small_train_data = TensorDataset(*train_data[perm])

    train_loader = DataLoader(small_train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)  # type: ignore
    bert_lm = load_pretrained_weights(bert_config)
    model = BertClassifier(bert_config)
    model.common = bert_lm.common
    del bert_lm
    t.manual_seed(1)

    for p in model.parameters():
        p.requires_grad_(True)
    model.train().to(device)
    optimizer = t.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    optimizer.zero_grad()
    classification_loss_fn = t.nn.CrossEntropyLoss()
    wandb.watch(model, criterion=classification_loss_fn, log="all", log_freq=100)
    examples_seen = 0
    start_time = time.time()

    # We would want learning rate warmup on a harder fine-tuning, but this is short enough that it's fine
    for epoch in range(config.epochs):
        for i, (input_ids, one_zero_attention_mask, y_positive, y_stars) in enumerate(tqdm(train_loader)):
            input_ids = input_ids.to(device)
            one_zero_attention_mask = one_zero_attention_mask.to(device)
            y_positive = y_positive.long().to(device)
            y_stars = y_stars.long().to(device)
            # TBD lowpri: could use with t.autocast(device.type) on supported GPUs
            out = model(input_ids, one_zero_attention_mask)

            if DEBUG_CHECK_NANS and t.isnan(out.is_positive).any():
                raise ValueError("NaN detected!")

            cls_loss = classification_loss_fn(out.is_positive, y_positive)
            star_loss = config.star_loss_weight * F.l1_loss(out.star_rating, y_stars)
            (cls_loss + star_loss).backward()
            # See Table 3 of https://arxiv.org/pdf/2006.04884.pdf
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (i + 1) % config.step_every == 0:
                optimizer.step()
                optimizer.zero_grad()

                wandb.log(
                    dict(
                        cls_loss=cls_loss,
                        star_loss=star_loss,
                        elapsed=time.time() - start_time,
                        examples_seen=examples_seen,
                    )
                )
            examples_seen += len(input_ids)

            if i % 100 == 0:
                table = wandb.Table(
                    columns=[
                        "text",
                        "predicted sentiment",
                        "true sentiment",
                        "predicted stars",
                        "true stars",
                    ]
                )
                n = 5
                texts = tokenizer.batch_decode(input_ids[:n], skip_special_tokens=True)
                for i, text in enumerate(texts):
                    table.add_data(
                        text,
                        out.is_positive[i].argmax(-1).item(),
                        y_positive[i].item(),
                        out.star_rating[i].item(),
                        y_stars[i].item(),
                    )
                wandb.log(dict(examples=table))

        t.save(model.state_dict(), config.filename)
    return model


if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    config_dict = dict(
        lr=1e-5,
        batch_size=16,
        step_every=1,
        epochs=1,
        weight_decay=0.01,
        num_steps=9600,
        star_loss_weight=0.02,
        filename="./data/w2d2/bert_classifier.pt",
    )

    if "SKIP":
        # Automatically adjust for smaller GPU
        if t.cuda.get_device_properties(0).total_memory < 12 * 2**30:
            config_dict["batch_size"] = 8
            config_dict["step_every"] = 2

    if "SKIP":
        if os.path.exists(config_dict["filename"]):  # type: ignore
            print("Loading previously trained model: ", config_dict["filename"])
            model = BertClassifier(bert_config).to(device)
            state_dict = t.load(config_dict["filename"])
            model.load_state_dict(state_dict)
        else:
            print("Training model from scratch.")
            model = train(tokenizer, config_dict)

# %%
"""
## Evaluation

Produce predictions for a random subset of the test set, then use the provided code to measure your model's performance. On a V100, it would take around 4 minutes for all 25,000 test examples, so test your code on a tiny subset and then try a medium sized one.

- Use `torch.inference_mode` to ensure gradients aren't calculated.
- Unlike in training, for evaluation there's no such thing as a batch size that is too big. Since we don't need to store optimizer state or gradients, we can fit a larger batch onto the GPU. Depending on the specifics of the computation, it will either be equally as fast or faster if you use the largest batch that fits in the GPU.
"""
# %%
def test_set_predictions(model: BertClassifier, test_data: TensorDataset, batch_size=256) -> tuple[t.Tensor, t.Tensor]:
    """
    Return (predicted sentiment, predicted star rating) for each test set example.

    predicted sentiment: shape (len(test_data),) - 0 or 1 for positive/negative
    star: shape (len(test_data), ) - star rating
    """
    "SOLUTION"
    model.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)  # type: ignore
    pred_sentiments = []
    pred_stars = []
    with t.inference_mode():
        for i, (input_ids, one_zero_attention_mask, y_positive, y_stars) in enumerate(tqdm(test_loader)):
            input_ids = input_ids.to(device)
            one_zero_attention_mask = one_zero_attention_mask.to(device)
            out = model(input_ids, one_zero_attention_mask)
            pred_sentiments.append(out.is_positive.argmax(dim=-1))
            pred_stars.append(out.star_rating)

    return t.cat(pred_sentiments, dim=0), t.cat(pred_stars, dim=0)


if MAIN:
    n = len(test_data)
    perm = t.randperm(n)[:1000]
    test_subset = TensorDataset(*test_data[perm])
    pred_sentiments, pred_stars = test_set_predictions(model, test_subset)
    correct = pred_sentiments.cpu() == test_subset.tensors[2]
    sentiment_acc = correct.float().mean()
    star_diff = pred_stars.cpu() - test_subset.tensors[3]
    star_error = star_diff.abs().mean()
    print(f"Test accuracy: {sentiment_acc:.2f}")
    print(f"Star MAE: {star_error:.2f}")

# %%
"""
### Inspecting the Errors

Print out an example that your model got egregiously wrong - for example, the predicted star rating is very different from the actual, or the model placed a very high probability on the incorrect class. Ideally, do this without looking at the true label.

Decode the text and make your own prediction, then check the true label. How good was your own prediction? Do you agree more with the "true" label or with your model?

If the model was in fact wrong, speculate on why it got that example wrong.
"""
# %%
if MAIN:
    if "SOLUTION":
        worst_preds = star_diff.abs().argsort(descending=True)
        wrong_tokens = test_subset.tensors[0][worst_preds]
        wrong_text = tokenizer.batch_decode(wrong_tokens, skip_special_tokens=True)
        print(wrong_text[0])


"""
## Bonus

Go on to Part 3, but if you have time at the end you can come back and try the bonus exercise for this part.

### Advanced Fine-Tuning

Read the paper [Revisiting Few-Sample BERT Fine-Tuning](https://openreview.net/pdf?id=cO1IH43yUF). Summarize their claims and see if you can improve your accuracy by implementing one.

### More on Metrics

Our evaluation metric is not fully aligned with the loss function:

    - The cross entropy loss rewards the model more for getting the probability of the correct class closer and closer to 1.
    - In evaluation, the accuracy metric only cares if the correct class prediction is the highest, and make no distinction between predicting the correct class with 51% probability and with 99% probability.

This suggests that we could potentially obtain a higher accuracy by using a different loss function that would encourage the model to focus more on getting past the 50% mark. Propose an alternative loss function, or research existing alternatives, and try implementing one to see if it improves accuracy.
"""
# %%
