"""
# W3D1 - Parallelism, Part 2

Below is a training loop for a ResNet variant on CIFAR10. It's already been optimized with some fancy tricks and on my machine trains to 94% accuracy in 30s (or around 18 seconds on an A100). Let's see how much faster we can get it by using multiple GPUs!

<!-- toc -->

"""

import collections
import copy
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import w3d1_model
import wandb
from w3d1_utils import (
    CustomOptimizer,
    data_augmentation,
    load_cifar10,
    update_ema,
    validation,
)

IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)


@dataclass(frozen=True)
class Config:
    epochs: int
    batch_size: int
    momentum: float
    weight_decay: float
    weight_decay_bias: float
    ema_update_freq: int


DEFAULT_CONFIG = Config(
    epochs=10,
    batch_size=256,
    momentum=0.9,
    weight_decay=0.256,
    weight_decay_bias=0.004,
    ema_update_freq=5,
)
# %%
"""
## Parallelizing the Training Loop

Modify the training loop so it can either run using `dist` or without it. The function `dist.is_initialized()` can be used to tell you if you're using `dist`.

You shouldn't need to modify `w3d1_utils` or `w3d1_model`, just the `train` and `rank_process` functions. Suggested order of operations:

- Implement `rank_process` to perform the necessary `dist` initialization boilerplate before calling train().
- Handle the data loading in a manner of your choice. The current implementation loads the entire training set to GPU memory, which saves time transferring it on every epoch. Your solution should be at least as performant.
    - A simple starting point is: each rank loads all the data to CPU, preprocesses it, and then puts a slice of it in GPU memory where the slice depends on the rank. Then each rank can shuffle and iterate through its own slice without ever having to communicate training data.
    - Better is to preprocess once and save the slices in separate files, so each process can load only the file(s) that contain data for its slice.
- Ensure the models start out in an identical state on each rank.
    - Method 1: compute the initial state on rank 0 and broadcast the parameters to all the others.
    - Method 2: seed the RNG the same on each rank, so that the random initialization is the same. In this case, you'll need to take care of the `first_layer_weights` parameter whose initialization is based on training data statistics.
- Add `dist.all_reduce` after `backward` to sum up the gradients.
- Ensure your code is clear about what is a batch versus a minibatch.

A common gotcha is that if different processes have a different number of minibatches due to unequal division of the training data, you'll get out of sync at the end of the first epoch. Watch out for this and consider logging the data size and number of minibatches on each process.

Note that the best hyperparameters depends on the number of GPUs. For example, I found it beneficial to increase the batch size with the GPU count.
"""
# %%
def train(config: Config, is_leader=True, seed=0, wandb_mode="disabled") -> None:
    batch_size = config.batch_size
    rank = 0

    if "SOLUTION":
        is_dist = dist.is_initialized()
        if is_dist:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert world_size > 1
            assert config.batch_size % world_size == 0
            batch_size = config.batch_size // world_size

    if is_leader:
        wandb.init(project="w1d4", config=asdict(config), mode=wandb_mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type != "cpu" else torch.float32
    start_time = time.perf_counter()
    torch.manual_seed(seed + rank)
    torch.backends.cudnn.benchmark = True  # type: ignore
    train_data, train_targets, valid_data, valid_targets = load_cifar10(device, dtype)
    first_layer_weights = w3d1_model.patch_whitening(train_data[:10000, :, 4:-4, 4:-4])  # (27, 3, 3, 3)
    train_model = w3d1_model.ResNetBagOfTricks(first_layer_weights, c_in=3, c_out=10, scale_out=0.125)
    train_model.to(dtype)
    for module in train_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()
    train_model.train(True)
    train_model.to(device)
    if is_leader:
        valid_model = copy.deepcopy(train_model)

    if "SOLUTION":
        if is_dist:
            handles = []
            for p in train_model.parameters():
                handles.append(dist.broadcast(p, 0, async_op=True))
            for h in handles:
                h.wait()

    optimizer = CustomOptimizer(
        train_model.parameters(),
        config.momentum,
        config.weight_decay,
        config.weight_decay_bias,
    )

    if is_leader:
        print(f"Preprocessing: {time.perf_counter() - start_time:.2f} seconds")

    train_time = 0.0
    batch_count = 0
    valid_acc = 0.0

    if "SOLUTION":
        if is_dist:
            data_start = rank * len(train_data) // world_size
            data_end = (rank + 1) * len(train_data) // world_size
            print(f"{rank}: data runs from {data_start} to {data_end}. Local batch size: {batch_size}")
            # Slice makes a copy and keeps the full data - clone allows full data to be GC'd
            train_data = train_data[data_start:data_end].clone()
            train_targets = train_targets[data_start:data_end].clone()
            print("Train data: ", train_data.shape)

    if is_leader:
        print("\nepoch    batch    train time [sec]    validation accuracy")

    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        # Note: because we seeded with seed+rank, the randperm
        # will be different per rank. Otherwise it would be the same.
        indices = torch.randperm(len(train_data), device=device)
        data = data_augmentation(train_data[indices], batch_size)
        targets = train_targets[indices]

        for i in range(0, len(data), batch_size):
            if i + batch_size > len(data):
                break  # discard partial batches

            inputs = data[i : i + batch_size]
            target = targets[i : i + batch_size]
            batch_count += 1
            train_model.zero_grad()
            logits = train_model(inputs)
            loss = w3d1_model.label_smoothing_loss(logits, target, alpha=0.2)
            loss_sum = loss.sum()
            loss_sum.backward()

            if "SOLUTION":
                if is_dist:
                    for p in train_model.parameters():
                        by_dtype = collections.defaultdict(list)
                        if p.grad is not None and p.requires_grad:
                            by_dtype[p.grad.dtype].append(p.grad)
                        for v in by_dtype.values():
                            dist.all_reduce_coalesced(v)

            optimizer.step()

            if is_leader and (i // batch_size % config.ema_update_freq) == 0:
                update_ema(train_model, valid_model, 0.99**config.ema_update_freq)

        train_time += time.perf_counter() - start_time
        if is_leader:
            valid_acc = validation(
                valid_model,
                valid_data,
                valid_targets,
                batch_size,
                epoch,
                batch_count,
                train_time,
            )
            wandb.log(
                dict(
                    elapsed=train_time,
                    train_loss=loss_sum,
                    val_acc=valid_acc,
                    epoch=epoch,
                    examples_seen=batch_count * batch_size,
                )
            )

    if is_leader:
        print("Final validation accuracy: ", valid_acc)
        wandb.finish()


def rank_process(rank: int, world_size: int):
    """Initialize the store, process group, and call train with the default config."""
    "SOLUTION"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    store = dist.TCPStore("127.0.0.1", 29500, world_size, rank == 0)
    dist.init_process_group("nccl", store=store, rank=rank, world_size=world_size)
    train(DEFAULT_CONFIG, is_leader=rank == 0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("n_gpu", type=int, nargs="?", default=1)
    args = parser.parse_args()
    world_size = args.n_gpu
    if world_size == 1:
        train(DEFAULT_CONFIG, is_leader=True)
    else:
        mp.spawn(rank_process, args=(world_size,), nprocs=world_size, join=True)


"""
## Onto Part 3

Congratulations! At this point you have a correct implementation of data parallelism that provides some speedup in wall-clock time over a single GPU, without having to touch the model at all.

My un-optimized solution was able to improve from 30 seconds on 1 V100 GPU to 22 seconds on 2 V100 GPU, but 4 V100 still took 22 seconds, which indicates that there wasn't enough work to saturate 4 GPUs. However, I was able reach 11 seconds on 4 V100 by doubling the batch size without significantly reducing accuracy. A 2.7x speedup on 4 GPU is respectable, but it's possible to do substantially better.

I recommend moving onto Part 3 now, but if you have time at the end of the day, come back to this part and see how much you can improve!

## Bonus

<details>

<summary>Bonus - Ideas for Optimization</summary>

To go faster, one optimization is to start communicating gradients as soon as they're available, without waiting for `backward` to complete. This interleaving can hide some of the communication latency. You could also perform the optimizer step for a parameter as soon as the all_reduce is done, instead of waiting for all the grads to be available.

Another optimization is that many small calls to `all_reduce` have substantial overhead. You can try `all_reduce_coalesced` instead.

As well as improving wall clock time, you can improve memory usage. Read about the [ZeroRedundancyOptimizer](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html) and try to do the same sharding of the optimizer states.

Another cool idea is to use lossy compression on the gradients to reduce the amount of data that needs to be communicated. The [PowerSGD](https://arxiv.org/pdf/1905.13727.pdf) paper explores this.

</details>

<details>

<summary>Bonus - DistributedDataParallel API</summary>

In the actual PyTorch, the class [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP for short) encapsulates all the distributed code to further reduce the changes needed in the user's training loop. In the simplest case, you can just wrap the model object in a DDP object and that's it.

This is intended to make things very easy for the user, but is actually sometimes slower than a more intrusive solution. Refer to the optional reading in Part 1 and get an idea of how DDP is implemented. If it sounds interesting, try implementing parts of it yourself.

</details>
"""
# %%
# if "SKIP":
#     # Implementation of PowerSGD that someone wrote in MLAB1 (maybe Tao?)
#     # TBD lowpri: test for correctness and benchmark
#     class LowRankCompressionDistributedSGD:
#         def __init__(
#             self,
#             params: list[t.nn.Parameter],
#             compression_rank: int,
#             dist_size,
#             lr: float,
#             momentum: float,
#         ):
#             self.cr = compression_rank
#             self.lr = lr / dist_size
#             self.mu = momentum
#             params = [param for param in params if param.requires_grad]
#             self.device = params[0].device
#             # all compressed parameters have to be rank 2, for conv "output (input h w)" works best, but ideally it'd be "largest (all others)"
#             self.compressed_params = [p.view(p.shape[0], -1).to(self.device) for p in params if len(p.shape) > 1]
#             self.raw_params = [p for p in params if len(p.shape) <= 1]
#             self.params = params

#             self.momentum = [None for p in self.params]
#             self.errors = [torch.zeros_like(p).to(self.device) for p in self.compressed_params]
#             # why is this variance 1??? is it???
#             self.qs = [torch.randn(p.shape[-1], self.cr).to(self.device) for p in self.compressed_params]
#             self.ps = [torch.zeros(p.shape[0], self.cr).to(self.device) for p in self.compressed_params]

#         def share_raw_grads(self):
#             for reduction in [dist.all_reduce(p.grad, async_op=True) for p in self.raw_params]:
#                 reduction.wait()

#         def decompress(self):
#             for param, q, p in zip(self.compressed_params, self.qs, self.ps):
#                 t.matmul(p, q.permute(1, 0), out=param.grad)

#         def add_error(self):
#             for param, error in zip(self.compressed_params, self.errors):
#                 param.grad.add_(error)

#         def orthonormalize(self, matrix):
#             for column in range(matrix.shape[1]):
#                 for pre_column in range(column):
#                     matrix[:, column] -= (
#                         pre_column
#                         * t.einsum("i,i->", matrix[:, column], matrix[:, pre_column])
#                         / matrix[:, pre_column].norm()
#                     )
#                 matrix[:, column] = F.normalize(matrix[:, column], dim=0)

#         def compress_gradients_and_store_error(self):
#             reductions = []
#             for i, (param, q, p) in enumerate(zip(self.compressed_params, self.qs, self.ps)):
#                 t.matmul(param.grad, q, out=p)
#                 reductions.append(dist.all_reduce(p, async_op=True))
#             for reduction in reductions:
#                 reduction.wait()

#             reductions = []
#             for i, (param, q, p) in enumerate(zip(self.compressed_params, self.qs, self.ps)):
#                 self.orthonormalize(p)
#                 t.matmul(param.grad.permute(1, 0), p, out=q)

#                 decompressed = t.einsum("nr, mr -> nm", p, q)
#                 self.errors[i] = param.grad - decompressed
#                 reductions.append(dist.all_reduce(q, async_op=True))

#             for reduction in reductions:
#                 reduction.wait()

#         def zero_grad(self):
#             for p in self.params:
#                 if p.grad is not None:
#                     p.grad.zero_()

#         def step(self):
#             for param in self.params:
#                 assert isinstance(param.grad, t.Tensor)
#             with torch.no_grad():
#                 # setting grad of my views on parameters because they don't automatically get grad
#                 for param, cp in zip([x for x in self.params if len(x.shape) > 1], self.compressed_params):
#                     cp.grad = param.grad.view(param.grad.shape[0], -1)

#                 self.add_error()
#                 self.compress_gradients_and_store_error()
#                 self.share_raw_grads()
#                 self.decompress()

#                 for i, p in enumerate(self.params):
#                     g = p.grad
#                     if self.momentum[i] is not None:
#                         self.momentum[i] = self.mu * self.momentum[i] + g
#                     else:
#                         self.momentum[i] = g
#                     g = self.momentum[i]
#                     p -= self.lr * g
