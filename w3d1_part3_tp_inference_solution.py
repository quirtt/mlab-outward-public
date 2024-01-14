# %%
"""
# W3D1 Part 3 - Tensor Parallel

We've seen that data parallel training can allow you to scale training to many GPUs by dividing up the batch elements. It can be used on any model with minimal adjustments to the code. 

However, data parallel alone has limitations: 

- The model must be small enough that one GPU can run it with a batch size of at least 1.
- The maximum number of GPUs you can use is limited by the batch size; if you want to do inference on a batch size of 1, data parallel won't help.

In this part we'll see how to gain speedups when data parallel can't help.

<!-- toc -->

## Running Large Models

As of this writing, an A100 GPU in the 80GB RAM configuration is the most RAM in a single device that you can readily purchase. Large models like GPT-3 175B which have 175B * 2 bytes per `float16` = 350GB of weights cannot even be loaded on an A100 to do inference, let alone also fit optimizer state and gradients for training.

Obviously, it would be nice if we could just put more RAM on each device, but the engineering challenges are substantial. In fact, the 80GB A100 actually has 6x16GB memory modules totalling 96GB of RAM, but it's so difficult to make them without defects that only 5 are active on any given chip. Even the next generation H100 GPU also has 80GB of RAM in this same 5 of 6 configuration.

To solve this, there are two popular ways to partition a single instance of the model across multiple GPUs: pipeline parallelism (which we won't be implementing today) and tensor parallelism. The term "model parallelism" is ambiguous; sometimes it refers broadly to any technique where the model is partitioned, but sometimes it's used more narrowly as a synonym for tensor parallelism. It's worth clarifying what's intended if you hear or read this term.

### Pipeline Parallel

In pipeline parallelism, we divide up the layers across GPUs. So the first GPU might compute the embedding, then the second GPU might compute the first attention layer, and so on with GPUs passing activations forward in the forward pass, and passing gradients back in the backward pass. Systems like [PipeDream](https://www.microsoft.com/en-us/research/uploads/prod/2019/08/pipedream.pdf) use this strategy, but it has some drawbacks. The implementation is quite complicated, because layers can be of greatly different sizes. Naively, most of the GPUs are idling most of the time because GPU 5 can't do anything for a given input until GPU 4 finishes. This can be optimized by splitting up the batch into smaller "microbatches" and streaming them through, at the cost of even more implementation complexity. 

### Tensor Parallel

In tensor parallelism, we divide up individual parameter tensors across GPUs. In some cases like the transformer's embedding weight, this could be necessary because the parameter itself is too large to fit on one GPU.

Once divided, each GPU does as much as possible on its own, and we insert the minimum amount of communication between GPUs needed.

Today, we'll build up a tensor parallel implementation of the GPT architecture and use it to perform inference on 4 GPUs simultaneously. We will need tensor parallel versions of these layers that you've previously implemented:

- Linear
- Embedding
- UnidirectionalAttention

To start, we'll test with a simulated multi-GPU setup and then once our code is working, move up to a real machine with multiple GPUs.
"""
# %%
r"""
## Tensor Parallel - Linear Layer

A `Linear(in_channels, out_channels)` has a weight matrix of shape `(out_channels, in_channels)`, and the forward method computes $y = x {W}^\intercal + b$. 

The fact that the weight is transposed is an implementation detail: it means that columns of ${W}^\intercal$ are contiguous in memory which allows faster multiplication.

We will implement two different methods for splitting up the calculation: partitioning either rows or columns of ${W}^\intercal$ across devices. To be specific, for `Linear(3, 4)` the weight multiplication could look like this:

Partition columns, concatenating results to form output:
$$
\begin{equation*}
\begin{gather*}
\left[
\begin{array}{ccc}
x_0 & x_1 & x_2 
\end{array}
\right]
\left[
\begin{array}{cc:cc}
w_{00} & w_{01} & w_{02} & w_{03} \\
w_{10} & w_{11} & w_{12} & w_{13} \\
w_{20} & w_{21} & w_{22} & w_{23} \\
\end{array}
\right] \\
\begin{array}{cc}
\hspace{7.4em}\text{\scriptsize GPU 0}&\hspace{2.2em} \text{\scriptsize GPU 1}
\end{array}
\end{gather*}
=
\begin{gather*}
\left[
\begin{array}{c}
\sum_i w_{i0} x_i \\
\sum_i w_{i1} x_i
\end{array} \\
\right] \\
\left[
\begin{array}{c}
\sum_i w_{i0} x_i \\
\sum_i w_{i1} x_i
\end{array} \\
\right]
\end{gather*}

\begin{array}{c}
\text{\scriptsize GPU 0} \\ \\
\text{\scriptsize GPU 1}
\end{array}

\end{equation*}
$$

Partition rows, adding elementwise to combine contributions:
$$
\begin{equation*}
\begin{gather*}
\begin{array}{c}
\end{array}\\

\left[
\begin{array}{cc:c}
x_0 & x_1 & x_2 
\end{array}
\right] \\

\begin{array}{cc}
\hspace{1em}\text{\scriptsize GPU 0} & \text{\scriptsize GPU 1}
\end{array}
\end{gather*}

\left[
\begin{array}{cccc}
w_{00} & w_{01} & w_{02} & w_{03} \\
w_{10} & w_{11} & w_{12} & w_{13} \\
\hdashline
w_{20} & w_{21} & w_{22} & w_{23} \\
\end{array}
\right]

\begin{array}{c}
\\[0.5pt]
\text{\scriptsize GPU 0} \\[4pt]
\text{\scriptsize GPU 1}
\end{array}

=
\begin{gather*}
\left[
\begin{array}{c}
w_{00} x_0 + w_{10} x_1 \\
w_{01} x_0 + w_{11} x_1 \\
w_{02} x_0 + w_{12} x_1 \\
w_{03} x_0 + w_{13} x_1
\end{array} \\
\right] \\
\text{\scriptsize GPU 0}
\end{gather*}
+
\begin{gather*}
\left[
\begin{array}{c}
w_{20} x_2 \\
w_{21} x_2 \\
w_{22} x_2 \\
w_{23} x_2
\end{array} \\
\right] \\
\text{\scriptsize GPU 1}
\end{gather*}
\end{equation*}
$$

In the first scheme, each device needs the full input `x` and is solely responsible for a subset of the output elements. Concatenating all the subsets gives the full output.

In the second scheme, each device can take a partition of `x` and computes partial sums for every output element. Summing all the partial sums gives the full output. 

Exercise: we've described partitioning the weight. In each scheme, how should you partition the bias?

<details>

<summary>Solution - Partitioning the Bias</summary>

In the first scheme, if we want each rank to have the final output for a subset, then we should partition the bias along the output dimension as well.

In the second scheme, two reasonable ideas are:

- store the entire bias on rank 0 and just add it there before communicating.
- partition the bias and have each rank add their slice at the appropriate offset. 

The second way distributes the work evenly, but in practice both the storage for the bias and the computation are negligible.

</details>

## FakeDistributed class

It can be painful to debug when using the real `torch.distributed` module, so for testing we've provided a replacement class `FakeDistributed` which supports everything needed for today.

In particular, you can run your script in the debugger and have full visibility into what's happening on every rank, because `FakeDistributed` actually uses threads under the hood.

To keep straight whether you're using the real or fake one, I recommend `import torch.distributed as real_dist` and then explicitly pass the version you want to use to each function instead of relying on variables defined globally.

## Memory Mapping

The `mmap_parameter` helper uses the operating system's "memory mapping" capability to load weights from disk. Going into detail of how this works is out of scope, but for today you need to know that this is fast, and that you shouldn't write to the memory map.

This is faster than using `t.load` because the bytes are already on disk in exactly the format we need them, so there's no overhead of deserialization. Deserialization is actually a bottleneck when you want to load large models, especially if you're trying to do comparisons between different large models and need to swap back and forth.

Memory mapped files are lazy and only read from disk as needed. You can even mmap something that doesn't fit in CPU RAM, and if you only read slices that do fit, it will still work.

Even better, when you read from the memory map, the operating system will transparently cache the chunk, and then later if you read it again it will probably hit the cache and be available at RAM speed. When running multiple processes, all of them can benefit from this cache without you having to do anything special.
"""

# %%
import logging
import os
import pprint
import threading
from functools import partial
from typing import Callable, Optional, Type, Union

import torch as t
from einops import rearrange
from torch import nn
from torch.nn import functional as F

import utils
import w2d3_part2_sampling_solution
import w3d1_test
import w3d1_utils
from w2d3_part1_loading_solution import ACTIVATION_FUNCTIONS, GPTConfig, UnidirectionalAttention
from w3d1_fake_distributed import AbstractDistributed, FakeDistributed
from w3d1_utils import CONFIGS, DATA_ROOT, STAT_FILENAME, UniAttnWeights, init_on_device, mmap_parameter

MAIN = __name__ == "__main__"
HIDDEN_SIZE = 16


try:
    print("Loading OPT data files. It's normal to see a UserWarning here.")
    folder = os.path.join(DATA_ROOT, "mlab/opt-125m")
    test_linear_weight = mmap_parameter(folder, "layers.0.fc1.weight")
    test_linear_bias = mmap_parameter(folder, "layers.0.fc1.bias")
    test_qkv_weight = mmap_parameter(folder, "layers.0.self_attn_qkv.weight")
    test_qkv_bias = mmap_parameter(folder, "layers.0.self_attn_qkv.bias")
    test_out_proj_weight = mmap_parameter(folder, "layers.0.self_attn.out_proj.weight")
    test_out_proj_bias = mmap_parameter(folder, "layers.0.self_attn.out_proj.bias")
    test_embed_tokens_weight = mmap_parameter(folder, "embed_tokens.weight")
except OSError as e:
    print(
        "Failed to load OPT-125M data files. Try running w3d1_preload.py first, then contact a TA if the problem persists."
    )
    raise e

"""
### Helper function - part()

Today we're going to be slicing a lot of tensors into equally sized pieces. Implement the helper function `part` to make this more safe and concise.

Recall that `slice` is a built-in type containing `start`, `stop`, and `step` fields which can be integers or `None`. Given `x=[1,2,3,4,5,6,7]`, writing `x[1:5:2]` is syntactic sugar for `x[slice(1, 5, 2)]`.
"""
# %%
def part(n: int, rank: int, world_size: int, must_be_even=True) -> slice:
    """For a sequence of n elements, return a slice object representing rank's partition.

    must_be_even: if True, raise ValueError unless all partitions are equal length.
    """
    "SOLUTION"
    start, mod = divmod(n * rank, world_size)
    if must_be_even and mod != 0:
        raise ValueError(f"{n*rank} does not evenly divide {world_size}")
    end = n * (rank + 1) // world_size
    return slice(start, end)


if MAIN:
    assert part(10, 0, 2) == slice(0, 5)
    assert part(10, 1, 2) == slice(5, 10)

    x = t.rand((10, 2))
    row_partitions = [x[part(10, 0, 2)], x[part(10, 1, 2)]]
    utils.assert_all_equal(x, t.cat(row_partitions, 0))

    col_partitions = [x[:, part(2, 0, 2)], x[:, part(2, 1, 2)]]
    utils.assert_all_equal(x, t.cat(col_partitions, 1))


# %%
"""
### Linear - Split Columns

Now we will implement the first strategy in the diagram, taking care to keep all the dimensions straight. 

Implement `linear_split_columns` using a call to `part`.
"""
# %%
def linear_split_columns(weight: t.Tensor, bias: t.Tensor, rank: int, world_size: int) -> tuple[t.Tensor, t.Tensor]:
    """Given the weight and bias of a nn.Linear, return just the slice of the weight and bias needed for the specified rank.

    Assume all slices are of equal size.

    weight: (out_channels, in_channels)
    bias: (out_channels,)

    Return (weight_slice, bias_slice)
    """
    "SOLUTION"
    out_channels, in_channels = weight.shape
    s = part(out_channels, rank, world_size)
    return weight[s], bias[s]


if MAIN:
    w3d1_test.test_linear_split_columns(linear_split_columns, test_linear_weight, test_linear_bias)

# %%
"""
Next, implement the `LinearSplitColumns` module.

We're going to have one instance of this per GPU, each with a different weight slice. 
In forward, make a call to `dist.all_gather` and then combine the results from all GPUs together.

Don't read any global variables; use the provided dist from the constructor.
"""
# %%
class LinearSplitColumns(t.nn.Module):
    """Input and output are exactly like nn.Linear.

    Multiple distributed instances will collaborate to compute the result.
    Note that out_channels is the TOTAL number of output channels, NOT the dimension of the weight slice.
    """

    dist: AbstractDistributed
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dist: AbstractDistributed,
    ):
        "SOLUTION"
        super().__init__()
        self.dist = dist
        self.world_size = dist.get_world_size()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = t.nn.Parameter(t.empty((self.out_channels // self.world_size, in_channels)))
        self.bias = t.nn.Parameter(t.empty(self.out_channels // self.world_size))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the same y = x W^t + b as a regular Linear by collaborating."""
        "SOLUTION"
        # Equivalent but slower: out_chunk = x @ self.weight.T + self.bias
        out_chunk = F.linear(x, self.weight, self.bias)
        chunks = [t.zeros_like(out_chunk) for _ in range(self.world_size)]
        self.dist.all_gather(chunks, out_chunk)
        return t.cat(chunks, dim=-1)


def child_test_linear(
    dist: FakeDistributed,
    module: Union[Type["LinearSplitColumns"], Type["LinearSplitRows"]],
    split_fn: Callable,
    batch_size=2,
    seq_len=3,
    dtype=t.float64,
) -> None:
    """This function is called on each thread/process.

    Using random placeholder data, it verifies that multiple modules have the same input/output as the serial version.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.logger.info(f"{rank}: starting")

    test_weight = test_linear_weight.to(dtype=dtype)
    test_bias = test_linear_bias.to(dtype=dtype)

    out_channels, in_channels = test_weight.shape
    reference = nn.Linear(in_channels, out_channels)
    reference.weight = nn.Parameter(test_weight)
    reference.bias = nn.Parameter(test_bias)

    yours = module(in_channels, out_channels, dist)
    w, b = split_fn(test_weight, test_bias, rank, world_size)
    yours.weight = nn.Parameter(w)
    yours.register_parameter("bias", b if b is None else nn.Parameter(b))

    if rank == 0:
        x = t.randn((batch_size, seq_len, in_channels), dtype=dtype)  # Pretend input data
    else:
        x = t.zeros((batch_size, seq_len, in_channels), dtype=dtype)
    dist.broadcast(x, 0)

    with t.inference_mode():
        actual = yours(x)

    expected = reference(x)
    utils.allclose_atol(actual, expected, 1e-5)
    dist.logger.info(f"{rank}: Output matched serial version!")


def launch_threads(target: Callable, num_threads: int, local_dist: FakeDistributed, *args, **kwargs) -> None:
    threads = []
    for rank in range(num_threads):
        thread_args = (local_dist,) + args
        thread_kwargs = dict(**kwargs, rank=rank)
        thread_target = local_dist.with_rank(target)
        thread = threading.Thread(
            target=thread_target, args=thread_args, kwargs=thread_kwargs, name=f"Rank{rank}Thread"
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


if MAIN:
    for num_threads in [1, 4]:
        local_dist = FakeDistributed(world_size=num_threads)
        local_dist.logger.info(f"Testing LinearSplitColumns with {num_threads} ranks...")
        target = partial(child_test_linear, module=LinearSplitColumns, split_fn=linear_split_columns)
        launch_threads(target, num_threads, local_dist)
        if local_dist.exceptions:
            local_dist.logger.error("One or more threads raised an exception; see the log and the exceptions field.")
            break

# %%
"""
### Linear - Split Rows

Now implement splitting in the other dimension. The test assumes you've either partitioned the bias evenly between ranks or put the full bias on rank 0 and None on the other ranks; if you want to use another strategy, you may need to replace the test with your own.
"""
# %%
def linear_split_rows(
    weight: t.Tensor, bias: t.Tensor, rank: int, world_size: int
) -> tuple[t.Tensor, Optional[t.Tensor]]:
    """Given the weight and bias of a nn.Linear, return just the slice of the weight and bias needed for the specified rank.

    You can assume all slices are of equal size.

    weight: (out_channels, in_channels)
    bias: (out_channels,)

    Return (weight_slice, bias_slice)
    """
    "SOLUTION"
    _, in_channels = weight.shape
    ws = part(in_channels, rank, world_size)
    return weight[:, ws], (bias if rank == 0 else None)


if MAIN:
    w3d1_test.test_linear_split_rows(linear_split_rows, test_linear_weight, test_linear_bias)

# %%
if "SKIP":
    # Alternate solution
    def linear_split_rows_split_bias(
        weight: t.Tensor, bias: t.Tensor, rank: int, world_size: int
    ) -> tuple[t.Tensor, Optional[t.Tensor]]:
        """Given the weight and bias of a nn.Linear, return just the slice of the weight and bias needed for the specified rank.

        You can assume all slices are of equal size.

        weight: (out_channels, in_channels)
        bias: (out_channels,)

        Return (weight_slice, bias_slice)
        """
        "SOLUTION"
        out_channels, in_channels = weight.shape
        ws = part(in_channels, rank, world_size)
        bs = part(out_channels, rank, world_size)
        return weight[:, ws], bias[bs]

    if MAIN:
        w3d1_test.test_linear_split_rows(linear_split_rows_split_bias, test_linear_weight, test_linear_bias)

# %%
"""
## LinearSplitRows

As before, use the provided dist from the constructor.

Optional exercise: which is preferable when using low-precision floats like float16, LinearSplitRows or LinearSplitColumns?

<details>

<summary>Solution - Numerical Stability</summary>

LinearSplitColumns is as stable as the serial version, because internally F.linear will compute each term using a higher precision accumulator and only round at the end.

For LinearSplitRows, a naive implementation will round each partial sum to fp16, all-reduce, and then round the final result again to fp16, which will make the result quite different.

In the bonus section you can implement a smarter method.
</details>
"""
# %%
class LinearSplitRows(t.nn.Module):
    """Like nn.Linear, but input can be either the full input or just our partition of the input.

    Multiple distributed instances will collaborate to compute the result.
    Note that in_channels is the TOTAL number of input channels, NOT the dimension of the weight slice and similarly for out_channels.
    """

    # Solution assumes that linear_split_rows is used

    weight: nn.Parameter
    bias: nn.Parameter
    dist: AbstractDistributed

    def __init__(self, in_channels: int, out_channels: int, dist: AbstractDistributed):
        "SOLUTION"
        super().__init__()
        self.dist = dist
        self.world_size = dist.get_world_size()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = t.nn.Parameter(t.empty((out_channels, in_channels // self.world_size)))
        self.bias = t.nn.Parameter(t.empty(out_channels))

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Compute the same y = x W^t + b as a regular Linear by collaborating.

        x: shape of either (..., in_channels) or (..., in_channels // world_size)
        """
        "SOLUTION"
        rank = self.dist.get_rank()
        channels = x.shape[-1]
        if channels == self.in_channels:
            xs = part(channels, rank, self.world_size)
            x_slice = x[..., xs]
            self.dist.logger.debug(f"Slicing x {x.shape} -> {x_slice.shape}")
        elif channels == self.in_channels // self.world_size:
            x_slice = x
            self.dist.logger.debug(f"Got sliced x: {x.shape}")
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")

        partial_sums = F.linear(
            x_slice.double(), self.weight.double(), self.bias.double() if self.bias is not None else None
        )
        self.dist.all_reduce(partial_sums)
        return partial_sums


if MAIN:
    # TBD: debug this
    for num_threads in [1, 4]:
        local_dist = FakeDistributed(world_size=num_threads)
        local_dist.logger.info(f"Testing LinearSplitRows with {num_threads} ranks...")
        target = partial(child_test_linear, module=LinearSplitRows, split_fn=linear_split_rows, dtype=t.float32)
        launch_threads(target, num_threads, local_dist)
        if local_dist.exceptions:
            local_dist.logger.error("One or more threads raised an exception; see the log and the exceptions field.")
            break


# %%
"""
### Embedding

The embedding weight is the largest single weight in the network because of the long dimension `vocab_size`, aka `num_embeddings` in `nn.Embedding`. We don't speed up our computation in wall time by splitting it, but we may need to split it anyway to spread out the memory usage. 

Again, we have two possible dimensions to split on. 

Splitting on the `num_embeddings` dimension, it means that each device has the full embedding vector for a subset of specific tokens.

Splitting on the hidden dimension means that each device has a subset of embedding dimensions for all tokens.

For now, implement it splitting on the vocabulary dimension and don't worry about doing the most efficient implementation. Your implementation should work even if `num_embeddings` doesn't evenly divide the world size.

<details>

<summary>I'm confused on the implementation!</summary>

From the rank and world size, your module can compute the start and end range of tokens that it's responsible for. For example, with `world_size=2` and `num_embeddings=10`, the second rank is responsible for tokens [5, 10) and its local weight[3] is the embedding vector for token 5+3=8.

A straightforward solution is to initialize a (*, embedding_dim) matrix of zeros, fill in the vectors that we are locally responsible for, and all-reduce.

This does transmit a lot of zeros over the network, but you can optimize this in the bonus section.

</details>
"""
# %%
class EmbeddingSplitVocab(t.nn.Module):
    """Input and output are exactly like nn.Embedding.

    Multiple distributed instances will collaborate to compute the result.

    Again, num_embeddings is the total number (the vocab size), not the size of the local weight.
    """

    weight: nn.Parameter

    def __init__(self, num_embeddings: int, embedding_dim: int, dist: AbstractDistributed):
        "SOLUTION"
        super().__init__()
        self.dist = dist
        self.world_size = dist.get_world_size()
        self.weight = nn.Parameter(t.randn(num_embeddings // self.world_size, embedding_dim))

        s = part(num_embeddings, dist.get_rank(), self.world_size)
        self.start = s.start
        self.stop = s.stop

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (*)

        Return (*, embedding_dim)
        """
        "SOLUTION"
        not_ours_mask = (x < self.start) | (x >= self.stop)
        masked = t.where(not_ours_mask, 0, x - self.start)
        out = self.weight[masked]
        out[not_ours_mask] = 0.0
        self.dist.all_reduce(out)
        return out


# %%
def embedding_split_vocab(weight: t.Tensor, rank: int, world_size: int) -> t.Tensor:
    """Given the weight of a nn.Embedding, return just the slice of the weight needed for the specified rank.

    Assume all slices are of equal size.

    weight: (in_channels, out_channels) aka (vocab_size, embedding_size)

    Return weight_slice
    """
    "SOLUTION"
    num_embeddings, embedding_dim = weight.shape
    ws = part(num_embeddings, rank, world_size)
    return weight[ws]


if MAIN:
    w3d1_test.test_embedding_split_vocab(embedding_split_vocab, test_embed_tokens_weight[:, :10])


# %%
def child_test_embedding(
    dist: FakeDistributed,
    module: Type[EmbeddingSplitVocab],
    split_fn: Callable,
    batch_size=20,
    seq_len=30,
) -> None:
    """This function is called on each thread/process.

    Using random placeholder data, it verifies that multiple modules have the same input/output as the serial version.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.logger.info(f"{rank}: starting")

    small_embed = test_embed_tokens_weight[:, :10]  # Make test faster
    num_embeddings, embedding_dim = small_embed.shape
    reference = nn.Embedding(num_embeddings, embedding_dim)
    reference.weight = nn.Parameter(small_embed)

    yours = module(num_embeddings, embedding_dim, dist)
    w = split_fn(small_embed, rank, world_size)
    yours.weight = nn.Parameter(w)

    if rank == 0:
        x = t.randint(0, num_embeddings, (batch_size, seq_len), dtype=t.int64)  # Pretend input data
    else:
        x = t.zeros((batch_size, seq_len), dtype=t.int64)
    dist.broadcast(x, 0)

    with t.inference_mode():
        actual = yours(x)

    expected = reference(x)
    utils.allclose_atol(actual, expected, 1e-5)
    dist.logger.info(f"{rank}: Output matched serial version!")


if MAIN:
    for num_threads in [1, 4]:
        local_dist = FakeDistributed(world_size=num_threads)
        local_dist.logger.info(f"Testing EmbeddingSplitVocab with {num_threads} ranks...")
        target = partial(child_test_embedding, module=EmbeddingSplitVocab, split_fn=embedding_split_vocab)
        launch_threads(target, num_threads, local_dist)
        if local_dist.exceptions:
            local_dist.logger.error("One or more threads raised an exception; see the log and the exceptions field.")
            break

# %%
"""
### UnidirectionalAttentionSplit

We could implement self-attention in terms of our tensor parallel Linear layers, but this uses two communication calls (QKV and O). 

Exercise: how can we partition the weights to do just one communication call?

<details>

<summary>Solution - UnidirectionalAttention weight partition</summary>

Each attention head already produces its output independently without communication to the other heads.

This means that a natural way to partition is along the head dimension. We can just reuse a regular UnidirectionalAttention instance with a subset of heads inside, and then an all-reduce at the end to sum the outputs from each subset.

</details>

<details>

<summary>Unidirectional bias partition</summary>

Again, the bias uses so little compute that it doesn't matter much what we do as long as it's correct. 

To reuse our UnidirectionalAttention forward method without changes, we could put the full bias tensor on rank 0 and use None on the other ranks.

</details>

<details>

<summary>My shapes aren't matching up and I'm confused!</summary>

In the serial version, it was safe to assume that `num_heads * head_size == hidden_size` and use these two expressions interchangeably. We're still going to assume this for the full model, but on the partitions this no longer holds.

Now that we are dividing up the `num_heads`, on each rank have a new term `local_num_heads = num_heads // world_size`. Check that you aren't using `hidden_size` when you really need `local_num_heads * head_size` or vice versa.

</details>
"""
# %%
import math

from einops import rearrange
from fancy_einsum import einsum


class UnidirectionalAttentionSplit(t.nn.Module):
    """Input and output are exactly like UnidirectionalAttention.

    Multiple distributed instances will collaborate to compute the result.

    Again, constructor parameters are the total size, not the local size. Assume the number of heads evenly divides the world size.
    """

    inner: UnidirectionalAttention

    def __init__(self, hidden_size: int, num_heads: int, dist: AbstractDistributed, dropout=0.0):
        assert hidden_size % num_heads == 0
        "SOLUTION"
        super().__init__()
        self.hidden_size = hidden_size
        self.dist = dist
        self.world_size = dist.get_world_size()
        assert num_heads % self.world_size == 0
        local_num_heads = num_heads // self.world_size
        head_size = hidden_size // num_heads
        self.inner = UnidirectionalAttention(hidden_size, local_num_heads, head_size, dropout=dropout)

    def forward(self, x: t.Tensor, cache=None):
        "SOLUTION"
        local_out = self.inner(x, cache=cache)
        self.dist.all_reduce(local_out)
        return local_out


# %%
def uni_split_heads(weights: UniAttnWeights, rank: int, world_size: int) -> UniAttnWeights:
    """Split the QKV and output_proj by attention head.

    As in GPT, the qkv_weight consists of the Q, K, and V parts concatenated in that order.
    """
    "SOLUTION"
    # QKV weight is stored as ((qkv head hidden), (in_channels))
    qkv = rearrange(
        weights.qkv_weight,
        "(qkv head head_dim) hidden -> qkv head head_dim hidden",
        qkv=3,
        head=weights.total_num_heads,
        hidden=weights.hidden_size,
    )
    qkv_p = part(weights.total_num_heads, rank, world_size)
    qkv_weight = rearrange(
        qkv[:, qkv_p],
        "qkv head head_dim hidden -> (qkv head head_dim) hidden",
    )

    if weights.qkv_bias is None:
        qkv_bias = None
    else:
        bias = rearrange(
            weights.qkv_bias,
            "(qkv head head_dim) -> qkv head head_dim",
            qkv=3,
            head=weights.total_num_heads,
        )
        qkv_bias = rearrange(bias[:, qkv_p], "qkv head head_dim -> (qkv head head_dim)")

    # out_proj_weight is (hidden_size, num_heads * head_size)
    ow_p = part(weights.hidden_size, rank, world_size)
    out_proj_weight = weights.out_proj_weight[:, ow_p]
    out_proj_bias = weights.out_proj_bias if rank == 0 else None

    return UniAttnWeights(
        weights.total_num_heads, weights.hidden_size, qkv_weight, qkv_bias, out_proj_weight, out_proj_bias
    )


if MAIN:
    attn_weights = UniAttnWeights(12, 768, test_qkv_weight, test_qkv_bias, test_out_proj_weight, test_out_proj_bias)
    w3d1_test.test_uni_split_heads(uni_split_heads, attn_weights)


# %%
def child_test_attn(
    dist: FakeDistributed,
    module: Type[UnidirectionalAttentionSplit],
    split_fn: Callable,
    batch_size=2,
    seq_len=3,
    num_heads=12,
    hidden_size=768,
    dtype=t.float64,
) -> None:
    """This function is called on each thread/process.

    Using random placeholder data, it verifies that multiple modules have the same input/output as the serial version.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.logger.info(f"{rank}: starting")

    weights = UniAttnWeights(
        num_heads,
        hidden_size,
        test_qkv_weight,
        test_qkv_bias,
        test_out_proj_weight,
        test_out_proj_bias,
    ).to(dtype=dtype)
    assert weights.qkv_bias is not None
    assert weights.out_proj_bias is not None

    reference = UnidirectionalAttention(hidden_size, num_heads, dropout=0.0)
    reference.qkv_proj.weight = nn.Parameter(weights.qkv_weight)
    reference.qkv_proj.bias = nn.Parameter(weights.qkv_bias)
    reference.output_proj.weight = nn.Parameter(weights.out_proj_weight)
    reference.output_proj.bias = nn.Parameter(weights.out_proj_bias)

    yours = module(hidden_size, num_heads, dist)
    split_w: UniAttnWeights = split_fn(weights, rank, world_size)
    yours.inner.qkv_proj.weight = nn.Parameter(split_w.qkv_weight)
    b = split_w.qkv_bias
    yours.inner.qkv_proj.register_parameter("bias", b if b is None else nn.Parameter(b))
    yours.inner.output_proj.weight = nn.Parameter(split_w.out_proj_weight)
    ob = split_w.out_proj_bias
    yours.inner.output_proj.register_parameter("bias", ob if ob is None else nn.Parameter(ob))

    if rank == 0:
        x = t.randn((batch_size, seq_len, hidden_size), dtype=dtype)  # Pretend input data
    else:
        x = t.zeros((batch_size, seq_len, hidden_size), dtype=dtype)
    dist.broadcast(x, 0)

    with t.inference_mode():
        actual = yours(x)

    expected = reference(x)
    utils.allclose_atol(actual, expected, 1e-5)
    dist.logger.info(f"{rank}: Output matched serial version!")


if MAIN:
    for num_threads in [1, 4]:
        local_dist = FakeDistributed(world_size=num_threads)
        local_dist.logger.info(f"Testing UnidirectionalAttentionSplit with {num_threads} ranks...")
        target = partial(child_test_attn, module=UnidirectionalAttentionSplit, split_fn=uni_split_heads)
        launch_threads(target, num_threads, local_dist)
        if local_dist.exceptions:
            local_dist.logger.error("One or more threads raised an exception; see the log and the exceptions field.")
            break


# %%
"""
## Tensor Parallel OPT

OPT has the same architecture as GPT with some minor tweaks like using ReLU instead of GeLU. We've provided a model definition which is the same as your GPT, except with our new tensor parallel layers.

Note that since the embedding is split, the unembedding operation needs an `all_gather`.
"""
# %%
class GPT2Block_TensorParallel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        layer_norm_epsilon: float,
        activation_function: str,
        dist: AbstractDistributed,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = UnidirectionalAttentionSplit(hidden_size, num_heads, dist, dropout=dropout)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.linear1 = LinearSplitColumns(hidden_size, hidden_size * 4, dist)
        self.nonlinearity = ACTIVATION_FUNCTIONS[activation_function]
        self.linear2 = LinearSplitColumns(hidden_size * 4, hidden_size, dist)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: t.Tensor, cache=None) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        "SOLUTION"
        x = x + self.attn(self.ln1(x), cache=cache)
        x = x + self.dropout(self.linear2(self.nonlinearity(self.linear1(self.ln2(x)))))
        return x


class GPT2_TensorParallel(nn.Module):
    def __init__(self, config: GPTConfig, dist: AbstractDistributed):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.dist = dist
        self.token_embedding = EmbeddingSplitVocab(config.vocab_size, config.hidden_size, dist)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks: utils.StaticModuleList[GPT2Block_TensorParallel] = utils.StaticModuleList(
            [
                GPT2Block_TensorParallel(
                    config.hidden_size,
                    config.num_heads,
                    config.dropout,
                    config.layer_norm_epsilon,
                    config.activation_function,
                    dist,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, x: t.Tensor, cache=None) -> t.Tensor:
        """
        x: shape (batch, seq), dtype t.int64 - the token ids

        Return: shape (batch, seq, vocab_size), dtype t.float32- the output logits
        """
        "SOLUTION"
        _, S = x.shape
        if cache is None:
            pos_start = 0
        else:
            CB, C_HEADS, CS, C_HEADSIZE = cache[0].k.shape
            assert CS == 0 or S == 1, "Pass in one seq at a time after loading cache"
            pos_start = CS
        pos = t.arange(pos_start, pos_start + S).to(x.device)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=cache[i] if cache else None)
        x = self.ln(x)
        unembed = t.einsum("bnl, vl -> bnv", x, self.token_embedding.weight)
        unembeds = [t.empty_like(unembed) for i in range(self.dist.get_world_size())]
        self.dist.all_gather(unembeds, unembed)
        out = t.cat(unembeds, dim=-1)
        return out


"""
## Load with OPT

The provided `fast_load_gpt` does weight loading, and the provided `child_run_opt` runs the model in multiple threads.

Modify these two functions if necessary, and then test your OPT on successively more realistic cases:

- `FakeDistributed` and CPU
- `FakeDistributed` and multiple GPUs
- Real `torch.distributed`

Play around and have fun with your OPT! You should be able to use your sampling code from GPT day, though it may create tensors on the wrong devices.

- For models that do fit on one GPU, do you see a benefit in forward pass speed? What if we have larger batch sizes like a beam search?
- How big a model can you load now? `w3d1_preload` should have prepared weight files for various model sizes. You can also try loading even larger models by modifying and re-running `w3d1_preload`.
- How does speed compare between using multiple processes with `torch.distributed` compared to multiple threads with `FakeDistributed`?
- Did your cache implementation "just work" or did it require modifications?


<details>

<summary>I'm getting 'RuntimeError: "LayerNormKernelImpl" not implemented for 'Half''</summary>

Half means float16, and this means on your installation, LayerNorm doesn't support float16 on whatever device you're using. If you're testing on CPU, you can cast your model to float32.

</details>

<details>

<summary>What is this meta device thing? Do I need to understand it?</summary>

You don't have to understand it, but it's an interesting and underdocumented feature of PyTorch that has a few use cases.

When a tensor has a device of "meta", it still has a shape, but no underlying storage. Operations on a storage are defined to produce another meta tensor with the correct output shape, but are otherwise a no-op.

The reason we're using these today is that by default, running the constructor of PyTorch's built-in `nn.Module` subclasses will cause random weight initialization to occur, and this ends up being surprisingly expensive on larger models and making the model take a long time to load. 

If you're trying to load a model using `from_pretrained` and see your CPU pegged at 100% and your disk idling, this is probably what is happening.

By initializing the module on the meta device instead, the RNG operations become a no-op and we avoid spending a bunch of time generating random initial weights that have no purpose (since we immediately overwrite them with our pretrained weights).

</details>
"""
# %%
def fast_load_gpt(gpt: GPT2_TensorParallel, folder: str, device=t.device("cuda:0"), dist=None) -> GPT2_TensorParallel:
    """Rapidly load a OPT model from parameters saved with w3d1_utils.save_state_dict."""
    if dist is None:
        rank = 0
        world_size = 1
        print("Loading GPT with dist = None")
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        dist.logger.info(f"{rank}: loading GPT with world_size {world_size} and dist = {dist}")

    gpt.requires_grad_(False)

    def load_weight_bias(dest: Union[nn.Linear, nn.Embedding, nn.LayerNorm], param_prefix: str) -> None:
        """Load weights and biases for a module that has identical copies on each rank (not split)."""
        src_weight = mmap_parameter(folder, f"{param_prefix}.weight")
        if src_weight.shape != dest.weight.shape:
            raise ValueError(f"Src: {src_weight.shape}, Dest: {dest.weight.shape}")
        init_on_device(dest, "weight", src_weight.to(device))

        bias = getattr(dest, "bias", None)
        if bias is not None:
            src_bias = mmap_parameter(folder, f"{param_prefix}.bias")
            if bias.shape != src_bias.shape:
                raise ValueError(f"Src: {src_bias.shape}, Dest: {bias.shape}")
            init_on_device(dest, "bias", src_bias.to(device))

    def load_embedding_split_vocab(dest: EmbeddingSplitVocab, param_prefix: str) -> None:
        src_weight = mmap_parameter(folder, f"{param_prefix}.weight")
        split_weight = embedding_split_vocab(src_weight, rank, world_size)
        init_on_device(dest, "weight", split_weight.to(device))

    def load_fc_columns(dest: LinearSplitColumns, param_prefix: str) -> None:
        src_weight = mmap_parameter(folder, f"{param_prefix}.weight")
        src_bias = mmap_parameter(folder, f"{param_prefix}.bias").to(device)
        split_weight, split_bias = linear_split_columns(src_weight, src_bias, rank, world_size)
        init_on_device(dest, "weight", split_weight.to(device))
        init_on_device(dest, "bias", split_bias.to(device))

    def load_attn_split_heads(dest: UnidirectionalAttention, param_prefix: str) -> None:
        src_qkv_weight = mmap_parameter(folder, f"{param_prefix}.self_attn_qkv.weight")
        src_o_weight = mmap_parameter(folder, f"{param_prefix}.self_attn.out_proj.weight")
        src_qkv_bias = mmap_parameter(folder, f"{param_prefix}.self_attn_qkv.bias")
        src_o_bias = mmap_parameter(folder, f"{param_prefix}.self_attn.out_proj.bias")
        src_weights = UniAttnWeights(
            gpt.config.num_heads, gpt.config.hidden_size, src_qkv_weight, src_qkv_bias, src_o_weight, src_o_bias
        )
        split_weights = uni_split_heads(src_weights, rank, world_size)

        init_on_device(dest.qkv_proj, "weight", split_weights.qkv_weight.to(device))
        if split_weights.qkv_bias is not None:
            init_on_device(dest.qkv_proj, "bias", split_weights.qkv_bias.to(device))
        init_on_device(dest.output_proj, "weight", split_weights.out_proj_weight.to(device))
        if dest.output_proj is not None:
            if split_weights.out_proj_bias is not None:
                init_on_device(dest.output_proj, "bias", split_weights.out_proj_bias.to(device))
            else:
                dest.output_proj.register_parameter("bias", None)

    load_embedding_split_vocab(gpt.token_embedding, "embed_tokens")
    load_weight_bias(gpt.pos_embedding, "embed_positions")
    load_weight_bias(gpt.ln, "final_layer_norm")
    for (i, my_block) in enumerate(gpt.blocks):
        load_weight_bias(my_block.ln1, f"layers.{i}.self_attn_layer_norm")
        load_attn_split_heads(my_block.attn.inner, f"layers.{i}")
        load_weight_bias(my_block.ln2, f"layers.{i}.final_layer_norm")
        load_fc_columns(my_block.linear1, f"layers.{i}.fc1")
        load_fc_columns(my_block.linear2, f"layers.{i}.fc2")

    for (name, p) in gpt.named_parameters():
        if p.device == "meta":
            print("WARNING: still on meta device: ", name)
        if p.device != device:
            print("WARNING: Did not send to device: ", name)
    return gpt


def child_run_opt(dist: FakeDistributed, devices: list[str], opt_name: str, prompt: str) -> None:
    """Load OPT in multiple threads and run inference."""
    config = CONFIGS[opt_name]
    rank = dist.get_rank()
    local_device = t.device(devices[rank])
    folder = w3d1_utils.mlab_weight_dir(opt_name)
    with w3d1_utils.init_on_meta_device():
        model = GPT2_TensorParallel(config, dist)
    fast_load_gpt(model, folder, device=local_device, dist=dist)
    if local_device.type == "cpu":
        print("Running on CPU: converting to float32 to avoid LayerNormKernelImpl not implemented for Half error")
        model = model.float()

    # All tokenize - this is cheap though we could also broadcast the
    # length and then broadcast the token ids
    tokenizer = w3d1_utils.get_tokenizer(opt_name)
    local_device = next(model.parameters()).device
    out = w2d3_part2_sampling_solution.sample_tokens(model, tokenizer, prompt, temperature=0)  # type: ignore
    dist.logger.info(f"{rank}: {opt_name} said: {out}")


# %%
"""
For reference, when prompted with w3d1_utils.DEFAULT_PROMPT, my OPT-125m said:

"I'm not conscious. I'm just trying to get a better understanding of what's going on.
I'm sorry, but I'm not"
"""


def run_opt(devices: list[str]) -> None:
    num_threads = len(devices)
    local_dist = FakeDistributed(world_size=num_threads)
    local_dist.logger.setLevel(logging.INFO)
    local_dist.logger.info(f"Testing OPT with {opt_name} and {num_threads} ranks...")
    target = partial(child_run_opt, devices=devices, opt_name=opt_name, prompt=w3d1_utils.DEFAULT_PROMPT)
    launch_threads(target, num_threads, local_dist)


if MAIN:
    print("Models available: ")
    pprint.pprint(CONFIGS)
    opt_name = list(CONFIGS)[0]

    print("Testing FakeDistributed on 1 CPU")
    run_opt(["cpu"])


# %%
if MAIN:
    print("Testing FakeDistributed on the below devices: ")
    devices = ["cuda:0", "cuda:0", "cuda:0", "cuda:0"]
    run_opt(devices)

# %%

"""

## Bonus

Congratulations on completing the day's main content!

### Float16 for LinearSplitRows

Go back and try to get your LinearSplitRows implementation to produce the same (or at least closer) results when float16 is used for the weights and inputs. 

### Optimizing the MLP

In the MLP, we have the sequence `Linear` -> `GELU` or `RELU` -> `Linear`. Replacing these with our tensor parallel `Linear`, we have a `all_reduce` or `all_gather` as part of each `Linear`. 

Find a way to eliminate the first communication completely and only sync at the end of the MLP.

<details>

<summary>Solution - Optimal Splits</summary>

If we split on columns first, then each device has a subset of complete outputs. Instead of communicating, we can just apply GELU to the complete outputs and feed the resulting subset directly into a LinearSplitRows, which is capable of accepting a subset as input.

If your LinearSplitRows isn't numerically stable enough, you can work in float32 or float64 for this.

</details>

### Optimizing the Embedding

Explore optimizing the partition along the vocabulary dimension, or different approaches like partitioning along the embedding dimension. What's the fastest way to do it?

### Tensor Parallel Training

We focused on inference today and left out some details that would be required to do training with tensor parallel. Primarily. we didn't implement backpropagation across GPUs. 

In the backwards pass, you'll need to communicate the gradients to the appropriate places. One way to do this is by writing a [torch.autograd.Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function).


### 2D Parallel Training (challenging)

You can "nest" tensor parallel inside of data parallel to combine their benefits. Suppose each tensor was split up on 2 GPUs with tensor parallel - then from the perspective of data parallel, that 2-GPU unit is just a single logical device.

The implementation is tricky:

- Instead of just using the rank, derive a "dp_rank" in {0, 1} and "tp_rank" in {0, 1} from the rank. 
- Communications aren't all-to-all anymore, as the tensor parallel communications only need to happen within the pair. You can do this in various ways, and may want to use [process groups](https://pytorch.org/docs/stable/distributed.html#groups), though FakeDistributed doesn't implement these yet. 
- It's often helpful when testing parallelism to generate a small set of random training data and see if you can get the model to memorize the labels.

### Uneven Size Partitions

For simplicity, we assumed that all our partitions were of equal size. Usually, we try to make our model dimensions and number of devices such that this works out, but it's not always possible. 

Go through and test if your code works if the partitions are unequal. We can always partition so that only the last partition might be smaller. For example, with 3 GPUs and 10 rows, the partitions would be [4, 4, 2].
"""

# %%
