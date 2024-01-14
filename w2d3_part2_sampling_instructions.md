
# W2D3 Part 2 - Sampling

The sampling procedure can make a huge difference to the perceived quality of the model output, but as of 2022 it's unclear what the best strategy is, or even how to operationalize what "best" means other than the expensive process of asking humans to rate samples.

The speed of sampling is also critical for commercial use of these language models. As expensive as they are to train, the cost of inference also adds up quickly for these large models.

Today we're going to implement common algorithms in their basic form to build conceptual understanding, particularly around their weaknesses. Don't worry about writing the most efficient code possible; you can go back and speed it up in the bonus section.

## Table of Contents

- [Sampling Boilerplate](#sampling-boilerplate)
- [Greedy Search](#greedy-search)
    - [Temperature](#temperature)
    - [Frequency Penalty](#frequency-penalty)
- [Sampling with `Categorical`](#sampling-with-categorical)
- [Sampling - Manual Testing](#sampling---manual-testing)
- [Top-K Sampling](#top-k-sampling)
    - [Top-K Sampling - Example](#top-k-sampling---example)
- [Top-p aka Nucleus Sampling](#top-p-aka-nucleus-sampling)
    - [Top-P Sampling Example](#top-p-sampling-example)
- [Efficient Text Generation Via Caching](#efficient-text-generation-via-caching)
- [Beam Search](#beam-search)
- [Bonus](#bonus)
    - [Larger GPT-2 Models](#larger-gpt--models)
    - [Batched Sampling with Temperature](#batched-sampling-with-temperature)
    - [Cached Beam Search](#cached-beam-search)
    - [PyTorch JIT](#pytorch-jit)
    - [Top-K and Top-P Together](#top-k-and-top-p-together)



```python
import os
import sys
from dataclasses import dataclass
import torch as t
import transformers
from einops import rearrange, repeat
from tqdm.auto import tqdm
import utils
import w2d3_test
from w2d3_part1_loading_solution import GPT2, GPT2Block, load_pretrained_weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)
if MAIN:
    my_gpt = load_pretrained_weights().eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

```

## Sampling Boilerplate

The provided functions `sample_tokens` and `sample_next_token` include the boilerplate for sampling from the model. Note that there is a special token `tokenizer.eos_token`, which during training was added to the end of a each article. GPT-2 will generate this token when it feels like the continuation is at a reasonable stopping point, which is our cue to stop generation.

The functions called in `sample_next_token` are not defined yet - you are going to implement them below.


```python
def sample_next_token(
    model: GPT2, input_ids: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0, cache=None
) -> int:
    """Return the next token, sampled from the model's probability distribution with modifiers.

    input_ids: shape (seq,)
    """
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"
    model.eval()
    with t.inference_mode():
        all_logits = model(input_ids.unsqueeze(0), cache=cache)
    (B, S, V) = all_logits.shape
    assert B == 1
    assert S == len(input_ids)
    logits = all_logits[0, -1]
    if temperature == 0:
        return greedy_search(logits)
    logits = apply_temperature(logits, temperature)
    logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def sample_tokens(
    model: GPT2,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=0.0,
    stop_at_eos=True,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> str:
    """Sample tokens using sample_next_token until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    """
    model.eval()
    input_ids: list = tokenizer(initial_text).input_ids
    generated = []
    device = next(model.parameters()).device
    for _ in tqdm(range(max_tokens_generated)):
        new_token = sample_next_token(
            model,
            t.tensor(input_ids + generated, dtype=t.int64, device=device),
            temperature=temperature,
            freq_penalty=freq_penalty,
            top_k=top_k,
            top_p=top_p,
            cache=cache,
        )
        generated.append(new_token)
        if stop_at_eos and new_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids + generated)

```

## Greedy Search

Implement `greedy_search`, which just returns the most likely next token. If multiple tokens are equally likely, break the tie by returning the smallest token.

Why not break ties randomly? It's nice that greedy search is deterministic, and also nice to not have special code for a case that rarely occurs (floats are rarely exactly equal).

Tip: the type checker doesn't know the return type of `item()` is int, but you can assert that it really is an int and this will make the type checker happy.


```python
def greedy_search(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, )

    Return: the most likely token
    """
    pass


if MAIN:
    logits = t.ones(100)
    logits[5] = 10
    logits[8] = 10
    assert greedy_search(logits) == 5
    w2d3_test.test_sample_zero_temperature(my_gpt, tokenizer, sample_tokens)

```

### Temperature

Temperature sounds fancy, but it's literally just dividing the logits by the temperature. As temperature goes to zero, this becomes the same as greedy sampling, and as the temperature goes to infinity this becomes the same as sampling from a uniform distribution.



```python
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    """
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    assert temperature > 0
    pass


if MAIN:
    logits = t.tensor([1, 2]).log()
    cold_logits = apply_temperature(logits, 0.001)
    print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
    utils.allclose(cold_logits, 1000.0 * logits)
    hot_logits = apply_temperature(logits, 1000.0)
    print("A high temperature flattens the distribution: ", hot_logits)
    utils.allclose(hot_logits, 0.001 * logits)

```

### Frequency Penalty

The frequency penalty is simple as well: count the number of occurrences of each token, then subtract `freq_penalty` for each occurrence. Hint: use `t.bincount` to do this in a vectorized way.


```python
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    """
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    pass


if MAIN:
    bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
    input_ids = tokenizer(bieber_prompt, return_tensors="pt")["input_ids"][0]
    logits = t.ones(tokenizer.vocab_size)
    penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
    assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
    assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"

```

## Sampling with `Categorical`

PyTorch provides a [`distributions` package](https://pytorch.org/docs/stable/distributions.html#distribution) with a number of convenient methods for sampling from various distributions.

We'll be using package more in future days, but for now we just need [`t.distributions.categorical.Categorical`](https://pytorch.org/docs/stable/distributions.html#categorical). Use this to implement `sample_basic`, which just samples from the provided logits (which may have already been modified by the temperature and frequency penalties).

Note that this will be slow since we aren't batching the samples, but don't worry about speed for now.


```python
def sample_basic(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    pass


if MAIN:
    N = 20000
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    utils.allclose_atol(counts, probs, atol=0.01)

```

## Sampling - Manual Testing

Run the below cell to get a sense for the `temperature` and `freq_penalty` arguments. Play with your own prompt and try other values.

Note: your model can generate newlines or non-printing characters, so calling `print` on generated text sometimes looks awkward on screen. You can call `repr` on the string before printing to have the string escaped nicely.


```python
if MAIN:
    N_RUNS = 1
    your_prompt = "Jingle bells, jingle bells, jingle all the way"
    cases = [
        ("High freq penalty", dict(freq_penalty=100.0)),
        ("Negative freq penalty", dict(freq_penalty=-1.0)),
        ("Too hot!", dict(temperature=2.0)),
        ("Pleasantly cool", dict(temperature=0.7)),
        ("Pleasantly warm", dict(temperature=0.9)),
    ]
    for (name, kwargs) in cases:
        for i in range(N_RUNS):
            output = sample_tokens(my_gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
            print(f"Sample {i} with: {name} ({kwargs}):")
            print(f"Your model said: {repr(output)}")

```

## Top-K Sampling

Conceptually, the steps in top-k sampling are:

- Find the `top_k` largest probabilities
- Set all other probabilities to zero
- Normalize and sample

Your implementation should stay in log-space throughout (don't exponentiate to obtain probabilities). Do you need to normalize?

<details>

<summary>Solution - Normalization</summary>

Categorical accepts unnormalized logits, so normalizing them would be redundant.

</details>


```python
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    """
    pass


if MAIN:
    k = 3
    probs = t.linspace(0, 0.4, 5)
    unnormalized_logits = probs.log() + 1.2345
    samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[:-k] = 0
    expected /= expected.sum()
    print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
    utils.allclose_atol(counts, expected, atol=0.01)

```

### Top-K Sampling - Example

The [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) famously included an example prompt about unicorns. Now it's your turn to see just how cherry picked this example was.

The paper claims they used top_k=40 and best of 10 samples. Note that this is quite slow - we'll see how to improve the speed after the next section.


```python
if MAIN:
    your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    output = sample_tokens(my_gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
    print(f"Your model said: {repr(output)}")

```

## Top-p aka Nucleus Sampling

Conceptually, in top-p we:

- Sort the probabilities from largest to smallest
- Find the cutoff point where the cumulative probability first equals or exceeds `top_p`. We do the cutoff inclusively, keeping the first probability above the threshold.
- If the number of kept probabilities is less than `min_tokens_to_keep`, keep that many tokens instead.
- Set all other probabilities to zero
- Normalize and sample

Optionally, refer to the paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf) for some comparison of different methods.


```python
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    pass


if MAIN:
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print(
        "top_p of 0.5 or lower should only return token 2: ",
        counts,
    )
    assert counts[0] == 0 and counts[1] == 0

# %%
if MAIN:
    N = 2000
    unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
    samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
    print(
        "top_p in (0.5, 0.8] should return tokens 1 and 2: ",
        counts,
    )
    assert counts[0] == 0

# %%
if MAIN:
    N = 20_000
    top_p = 0.71
    probs = t.Tensor([0, 0.1, 0.2, 0.3, 0.4])
    samples = t.tensor([sample_top_p(probs.log(), top_p) for _ in range(N)])
    counts = t.bincount(samples, minlength=len(probs)) / N
    expected = probs.clone()
    expected[0:3] = 0
    expected /= expected.sum()
    print(
        "Checking empirical frequencies (try to increase N if this test fails): ",
        counts,
    )
    utils.allclose_atol(counts, expected, atol=0.01)
    print("Success!")

```

### Top-P Sampling Example


```python
if MAIN:
    your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
    output = sample_tokens(my_gpt, tokenizer, your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
    print(f"Your model said: {repr(output)}")

```

## Efficient Text Generation Via Caching

The text generation we've done so far is needlessly re-computing certain values, which is very noticeable when you try to generate longer sequences.

Suppose you're generating text, and you've already run GPT on the sentence "My life motto:". Now you want to run the model on the sentence "My life motto: Always". Which computations from the first sentence can you reuse?

<details>
<summary>Show Answer</summary>

At each attention layer, the only things the attention layer needs from the previous sequence positions are the key and value vectors.

</details>

Modify your GPT-2 to optionally use a cache. When you run your GPT on "My life motto:", it should store the necessary values in the cache. Then in the next forward pass with just " Always" as input, it should load the cached values instead of recomputing them (and update the cache). This only needs to work with a single input sequence (batch size of 1), and you can assume that after the first forward pass, the input will be just one token.

The design of the cache is completely up to you - discuss possible designs with your partner before writing code. It should be possible to have only one GPT2 instance and many different cache instances at one time. Imagine that you want to use one instance to serve multiple users submitting requests for text generation like in [AI Dungeon](https://aidungeon.io/).

Some example considerations:

- Which GPT-2 classes need to interact with the cache?
- Should the cache be mutable and be updated in place, or does updating actually just create a separate instance?
- Is it possible for other programmers to incorrectly use your cache? Is there a way to prevent this failure mode or at least detect this and complain loudly?

If you're not convinced your API design makes sense, ask a TA before implementing it.

<details>
<summary>Help, my cache is silently computing the wrong thing!</summary>

A good debugging strategy is to use `nn.Module.register_forward_hook` to store the inputs and outputs to each module where you suspect a bug. Run your module in cached and uncached modes and check that the values you expect to be identical actually are.

Check your positional encoding and your attention mask. Try adding more assert statements to verify assumptions in your code.

</details>


```python
"TODO: Implement your cache here."


def sample_tokens_with_cache(
    model: GPT2,
    tokenizer,
    initial_text: str,
    max_tokens_generated=30,
    temperature=1.0,
    freq_penalty=2.0,
    stop_at_eos=True,
    top_k=0,
    top_p=0.0,
    cache=None,
) -> str:
    """Does the exact same thing as sample_tokens, but using cache to be faster."""
    pass


if MAIN:
    w2d3_test.test_identical_output_with_cache(
        my_gpt,
        tokenizer,
        "It is pitch black. You are likely to be eaten by a grue.",
        sample_tokens,
        sample_tokens_with_cache,
    )

```

## Beam Search

In beam search, we maintain a list of size `num_beams` completions which are the most likely completions so far as measured by the product of their probabilities. Since this product can become very small, we use the sum of log probabilities instead.

At each iteration, we run the batch of completions through the model and take the log-softmax to obtain `vocab_size` log-probs for each completion, or `num_beams * vocab_size` possible next completions in total.

If we kept all of these, then we would have `num_beams * vocab_size * vocab_size` completions after the next iteration which is way too many, so instead we sort them by their score and loop through from best (highest) log probability to worst (lowest).

For each next completion, if it ends in the end of sequence (EOS) token then we add it to a list of finished completions along with its score. Otherwise, we add it to the list of "to be continued" completions. The iteration is complete when the "to be continued" list has `num_beams` entries in it.

If our finished list now contains at least `num_return_sequences` completions, then we are done. If the length of the completion is now `len(prompt) + max_new_tokens`, then we are also done. Otherwise, we go to the next iteration.

Implement `beam_search` below. This is fully deterministic (make sure your model is in eval mode), so your version should be able to exactly reproduce the reference solution. Don't worry about speed or using your cache at this point and feel free to use a for loop over the sorted next-completions.

<details>

<summary>I'm confused about ranking the next completions from best to worst.</summary>

Flatten the model output to `num_beams * vocab_size` first, and then repeat the current logprobs of shape `num_beams` so you can add elementwise. This will give the logp of each possible completion, which you can then feed to `argsort` or `topk`.

Given some index into the flattened output, you can recover the original beam index and token id by using floor division and modulus, respectively.

</details>

<details>

<summary>I'm confused about what to do at the very start.</summary>

On every iteration after the first, you'll have completions of shape (num_beams, seq) but on the first iteration you'll just pass (1, seq). The probability of the initial prompt is just 1, so its log-prob is just 0.0.

</details>


```python
def beam_search(
    model, input_ids: t.Tensor, num_return_sequences: int, num_beams: int, max_new_tokens: int, tokenizer, verbose=False
) -> list[tuple[float, t.Tensor]]:
    """
    input_ids: (seq, ) - the prompt

    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    """
    assert num_return_sequences <= num_beams
    pass


if MAIN:
    your_prompt = "I don't want to rule the universe. I just think"
    input_ids = tokenizer(your_prompt, return_tensors="pt", return_attention_mask=False)["input_ids"][0]
    beam_out = w2d3_test.test_beam_search(
        beam_search,
        model=my_gpt,
        input_ids=input_ids,
        num_return_sequences=2,
        num_beams=6,
        max_new_tokens=20,
        tokenizer=tokenizer,
        verbose=True,
    )
    print("Final Completions: ")
    for (score, tokens) in beam_out:
        print(f"{score:.4f}: {repr(tokenizer.decode(tokens))}")

```

## Bonus

Congratulations on finishing the main content for the day! If you like, generate some text and share your favorite completions on Slack!

### Larger GPT-2 Models

Try loading some larger versions of GPT-2 and compare outputs on the same prompts.

### Batched Sampling with Temperature

Write a batched equivalent of `sample_tokens_with_cache` that can generate multiple continuations in parallel for a single prompt. It should produce the same output as a for loop around `sample_tokens_with_cache` (up to sampling randomness) but be faster.

Hint: refer to the pre-course exercises, in particular `sample_distribution` and the `gather` exercises. You may need to upgrade your cache implementation to support this use case.

### Cached Beam Search

Write an equivalent of `beam_search` that uses your cache implementation.

### PyTorch JIT

Learn about PyTorch's [JIT](https://pytorch.org/docs/stable/jit.html) and try applying it to your model, and measure the speed increase. It's normal for it to take some fiddling to get it to work.

### Top-K and Top-P Together

It's possible to use both top-k and top-p at the same time, by implementing them to take logits and return modified logits (and then letting `sample_basic` do the actual sampling). Refactor the implementation to do this. Which order should they be applied in? If you're uncertain, check the [HuggingFace](https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py#L3349) implementation to see what they do.

</details>
