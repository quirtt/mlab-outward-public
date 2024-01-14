# %%
"""
# W3D1 - Parallelism

Today's material is divided into three parts.

- Part 1: practice with `torch.distributed` and understand how the different backends and operations work.
- Part 2: implement data parallelism to speed up a training loop.
- Part 3: implement tensor parallelism and perform inference using a model that is too large for one GPU.

<!-- toc -->

## Readings

- [GPUs for Deep Learning](https://lambdalabs.com/blog/best-gpu-2022-sofar/): give this a skim to get a sense of how different GPUs compare to each other. Sometimes, it's more cost effective to use a smaller number of more powerful GPUs, and other times the opposite is true. Some hardware configurations of GPU+[interlink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) can scale close to linearly in ideal conditions, while others become bottlenecked on communication.
- [Fast Multi-GPU collectives with NCCL](https://developer.nvidia.com/blog/fast-multi-gpu-collectives-nccl/): Introduces the `nccl` library that we'll use today. You should understand the "ring" algorithm for broadcasting data.
- Optional - [PyTorch Distributed: Experiences on Accelerating Data Parallel Training](https://arxiv.org/pdf/2006.15704.pdf): this isn't necessary to understand today's material, but if you're interested in the engineering and API design aspect of the PyTorch data parallel module, this gives a lot of good detail.

Modern deep learning almost always requires multiple devices for training, as the scale of experiments has grown much more quickly than the capabilities of an individual device. The two basic problems with single device training are that (a) the model may not fit on one device, in which case it can't be trained at all, and (b) even if the model does fit on one device, the experiment would take a long time to run in wall clock time.

However, we don't yet know how to optimally train models in parallel, because there are a vast number of possible configurations and it requires a lot of effort on both the software and hardware sides. As of 2022, large scale experiments like [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) that are running on 6144 TPU v4 are able to achieve only around 46% of the potential maximum hardware utilization. This might seem low, but it reflects substantial engineering progress - 46% is already more than double GPT-3's utilization.

The problem is so hard because different parts of the system can be bottlenecked at different times on compute, GPU memory bandwidth, or communication between devices. The possibilities for dividing the computation are also combinatorial - for each operation you could think of partitioning the input among any subset of the dimensions, and then either merging the results or leaving the output partitioned for the next stage.

The three simplest parallelization strategies are:

- Data parallelism: each GPU is responsible for a slice of inputs along the batch dimension.
- Tensor parallelism: each GPU is responsible for a slice of weights (for example, the weights for a subset of attention heads).
- Pipeline parallelism: each GPU is responsible for a layer or set of layers.

The PaLM paper combines 12-way tensor parallelism and 256-way data parallelism. This means that for a batch size of 2048 and 48 attention heads, a single device would be responsible for computing the output of 48/12 = 4 attention heads on 2048/256 = 8 training examples.

## Data Parallelism

We're going to start today with data parallelism. It's not the most efficient, but it's comparatively simple to understand and implement, and can be further optimized and/or used in combination with other techniques.

We start with N identical GPUs and:

- Copy identical weights to each
- Divide our batch of size B into N "minibatches" of size M=B//N. We'll assume that N evenly divides B for today.
- Each GPU runs a forward and backward pass on its minibatch to obtain the local gradients for its minibatch.
- Synchronize the gradients with all-reduce.

"All-reduce" means that each GPU will send and receive minibatch gradients until all devices have the same sum. This sum of gradients is exactly identical to if we'd run a single forward pass on the full batch B, with some special treatment needed around batch normalization and dropout layers.

Batch normalization is a special case because it normally computes a mean over the full batch, but now each device is computing a mean over the minibatch. If you want dropout to be the same, you need to carefully initialize the random number generator on each device.

Assuming the special cases are handled and all devices have an identical sum of gradients, each GPU can now independently compute an identical optimizer step, which will deterministically modify the parameters to the same result. This completes a single iteration and all the devices are still in sync.

### Pros of Data Parallelism

The best thing is that any model that fits on 1 GPU can be "wrapped" in a data parallel version without thinking too hard about it. Because the batch elements are already independent of each other (again, except for batch norm), your devices only need to communicate once per batch at the end to sum their gradients.

In comparison, tensor parallelism and pipeline parallelism require you to send activations around during both forward and backwards passes, and there's some cleverness involved to minimize this amount of communication.

### Cons of Data Parallelism

One issue is that the communication between GPUs can easily get saturated - all of the GPUs want to send data at once when summing gradients. This can be mitigated somewhat by sending gradients of the later layers immediately as they're computed, interleaved with computing gradients of the earlier layers. It's also mitigated by using special extremely fast interconnects like NVLink.

Another issue is that if your model doesn't run on 1 GPU even with a minibatch size of 1, you can't use data parallelism alone; you are forced to use one of the other two strategies, possibly combined with data parallelism.

In terms of GPU memory, data parallelism is wasteful because all the parameters are duplicated N times, as is the optimizer state. This can be mitigated by splitting the optimizer state across devices, at the cost of more communication.

As N grows, the minibatch size B/N becomes too small to fully utilize the GPU. You can increase the total batch size B somewhat to compensate, but large batches tend to generalize more poorly so there's a problem-dependent cap to increasing B.

Let's get to it and build our own data parallel implementation using the `torch.distributed` package!

## torch.distributed

The `torch.distributed` package (`dist` for short) provides functions to communicate between different devices, which could mean multiple CPUs/GPUs on the same computer or on different computers. If we only wanted our code to run on a single computer with multiple GPUs, we don't need `torch.distributed` at all. We could just use `tensor.to()` to move data to and from the appropriate devices, and if needed we could use threads and shared memory or multiple process for concurrent operations.

Today, we're using `torch.distributed` so that everything you do would transparently scale to multiple computers connected by a network.

The actual implementation of an operation like broadcast or all-reduce is delegated to one of three backends:

- [`gloo`](https://github.com/facebookincubator/gloo/tree/main/gloo) is developed by Facebook. Its advantages are that it supports both CPU and GPU, but as you'll see the GPU version is much slower than NVIDIA's `nccl`. Usually, it has better error messages than `nccl` so it can be useful to debug against `gloo` before switching to `nccl`.
- [`nccl`](https://github.com/NVIDIA/nccl) (pronounced 'nickel') is developed by NVIDIA. It only works on NVIDIA GPUs, but it's been highly optimized and specialized to NVIDIA's GPUs and other products like [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) and NVSwitch.
- [`mpi`](https://github.com/open-mpi/ompi) stands for Message Passing Interface. Unlike the other two, this isn't a specific library but rather an old school open standard from the 90s. This is primarily aimed at clusters with thousands of CPUs, and we won't be using this today.

### Lockstep Multiprocessing

The programming paradigm with `dist` is called **lockstep multiprocessing**. It's called "lockstep" because whenever one process calls a communication function from the `dist` package, every process has to call that function or your program will deadlock. The multiprocessing refers to the fact that there are multiple processes on potentially different computers. For today, we'll use the common configuration where the processes each have a unique number called the `rank`, running from 0 to `world_size`. We give each process exclusive access to a single GPU.

All the processes except rank 0 are equal peers; in `dist`, rank 0 has additional responsibilities by convention. It's going to set up a key-value store that other processes can connect to and get and set values. A common way to do this is by providing the hostname and port of the machine running the rank 0 process to each other process. Today we'll do this for you as part of the utility code.

The store is an instance of [`torch.distributed.TCPStore`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore) and isn't how we actually send tensors around. It's needed to have all the processes agree on their ranks and how big `world_size` is, and to figure out the topology of the network. (TBD: can we see output of nccl doing this?)

Rank 0 is also used by convention in your code when you have work that only one rank needs to do. For example, it makes sense for rank 0 to save model checkpoints to disk instead of all the ranks doing it redundantly.

### GPU Topology

If you're doing this on an AWS box with a `p3.16xl` instance, your 8 V100 GPUs are arranged like this:

<p align="center">
    <img src="w3d1_p3_topology.png"/>
</p>

The thick arrows are NVIDIA's fast interconnects called `NVLink`. GPUs can simultaneously send and receive data at 25GB/s over each NVLink, meaning that a GPU with 6 NVLinks can send at a max of 150GB/s and simultaneously receive a max of 150GB/s. This is much faster than the PCIe and QPI connections, which are hardware dependent but let's call it 32GB/s and 20GB/s respectively.

Run `nvidia-smi topo -m` at the terminal to see the topology of your machine. For example, if you have 4 V100s it might look like this:

```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity
GPU0     X      NV1     NV1     NV2     0-31            N/A
GPU1    NV1      X      NV2     NV1     0-31            N/A
GPU2    NV1     NV2      X      NV2     0-31            N/A
GPU3    NV2     NV1     NV2      X      0-31            N/A
```

NV1 means there is one NVLink connecting the devices, and NV2 means there are two. NV12 is equivalent to NV6 and means that there is a bonded set of of 6 NVLinks

In the topology in the image above, suppose that GPU0 wants to `broadcast` a 1GB tensor to all the other GPUs. Doing this the naive way where GPU0 is only sending to each other GPU and each other GPU is only receiving, roughly how long will this take?

<details>

<summary>Solution - Naive Broadcast</summary>

First, start the slow transfer to GPU4, GPU5, and GPU7 over the QPI (a slow connection between CPUs), which takes 3GB at 20 GB/s, or 150ms.

During this time, you can complete the transfer to 1, 2, 3, and 6 simultaneously at the maximum bandwidth of a single NVLink, 1GB at 25 GB/s, 40ms.

The total time is then just the slow transfer total 150ms.

</details>

Suppose we chunk the transfer into 100 chunks of 10MB, and allow each GPU to both send and receive. Describe an efficient way to perform the broadcast and compute roughly how long it would take.

<details>

<summary>Solution - Efficient Broadcast</summary>

A reasonable ring order would be 0, 2, 4, 6, 7, 5, 3, 1 with NVLink at each step. Then the transfer from 0->2 will complete in 1/25 s. Since each GPU can simultaneously be sending and receiving, the send from 2->4 can start after a negligible delay of 1/25/100 s. The whole operation should complete in roughly 40ms, about 4 times faster than the naive version.

</details>

It's the backend's job to figure out the most efficient way to perform your broadcast given the topology, but it's your job to ensure that your operating system reports the correct topology. For example, if you're running inside a virtual machine or a container, `nccl` might see a virtual topology that doesn't reflect the real topology. If your system is running slowly, this is one thing you can check.

### Your First Broadcast

Now that you have an appreciation for the complexity behind the scenes of doing a broadcast, let's walk through the steps of making a broadcast at the PyTorch level.

First, we have to launch a process for each rank that we want. We're using `torch.multiprocessing` (`mp` for short) to create the processes. You can only launch processes on the same machine with this, but this is enough for our needs and it's easy to do this from the REPL. To launch processes on multiple machines, PyTorch provides something called [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html#launcher-api) but there are many other alternatives available.

Within each child process, we have to call `dist.init_process_group` first thing, which does the rendevous procedure and sets up the store. You can have multiple process groups and processes can belong to more than one group, which is useful if you're nesting multiple kinds of parallelism.

Once initialized, you can make use of `dist.get_rank()` and `dist.get_world_size()`.

Each receiving process has to allocate a tensor of the appropriate size. Sometimes, you'll need to communicate this size first but we can hardcode it for demonstration purposes. Then the tensor is sent by calling `dist.broadcast(tensor, 0)`. The 0 is the "source rank" meaning that when rank 0 calls `broadcast(tensor, 0)`, it'll be sending the data in `tensor` and when each other rank calls `broadcast(tensor, 0)`, it'll actually be receiving data into `tensor`, overwriting whatever is currently there.

It's confusing that the same function call with the same arguments can be either a send or a receive operation based on the rank of the currently running process, but you'll get used to it.

**Example**

GPU 0:
```python
x = torch.tensor([1])
dist.broadcast(x, 0)
# x = [1]
```

GPU 1:
```python
x = torch.tensor([2])
dist.broadcast(x, 0)
# x = [1]
```

### Distributed Logging

Using `print` statements is problematic when you have a bunch of processes involved that could potentially be on different computers. Even on the same computer, you might not see your print statements in some scenarios because the standard output stream isn't connected to something you can see, or the prints might appear but overlapping when multiple processes print at once to the same stream. The Python `logging` module will just work, even in the distributed scenario by sending log messages over TCP.

Today you can use `logger.info(message: str)` to log a message and it will append to a local file `w3d1_log.txt`. Use info for messages that you'd want to see when the model is running normally, and debug for messages that you'd only want to see if something was wrong.

### FakeDistributed Class

The unit tests run against a `distributed` replacement called `FakeDistributed` which actually runs multiple threads in the same process.

You'll use this class more in Part 3, but for now this is all you need to know about it.
"""
# %%
import logging
import logging.handlers
import os
import signal
import sys
import time
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Union

import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
from matplotlib import pyplot as plt

import w3d1_test

if not dist.is_available():
    print("torch.distributed is not available - exiting.")
    sys.exit(0)

IS_CI = os.getenv("IS_CI")
MAIN = __name__ == "__main__"
if MAIN:
    # Using fork gives "process 0 terminated with signal SIGSEGV" because of our listener process
    # Spawn is slower but also consistent between Linux/Windows/OSX.
    mp.set_start_method("spawn")

current_pids: list[int] = []
logger = logging.getLogger()
logger.setLevel("INFO")

# %%
"""
## Utilities

Read through for a general idea but don't worry about understanding every line.
"""
# %%
def log_receiver_process(queue: mp.Queue, log_filename="./w3d1_log.txt") -> None:
    """Dequeue records from other processes and write them to the specified file."""
    root = logging.getLogger()
    handler = logging.FileHandler(log_filename, mode="a")
    formatter = logging.Formatter(fmt="%(asctime)s.%(msecs)03d %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    while True:
        try:
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            logger.exception("Exception in log handler")
            traceback.print_exc(file=sys.stderr)


def make_logger(log_queue: mp.Queue) -> logging.Logger:
    h = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


def rank_process(
    rank: int,
    world_size: int,
    log_queue: mp.Queue,
    target_func: Callable,
    args: tuple,
    backend: str,
) -> None:
    """Runs in the child process. Initializes dist, the store, and the logger and calls target_func"""
    global logger, store
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    store = dist.TCPStore("127.0.0.1", 29502, world_size, rank == 0)
    dist.init_process_group(backend, store=store, rank=rank, world_size=world_size)
    logger = make_logger(log_queue)  # type: ignore
    target_func(*args)


def spawn_multiple(func: Callable, args: tuple, world_size: int, backend: str) -> None:
    """Spawns world_size processes, each with a different rank, that run func(*args)."""
    queue = mp.Queue(-1)
    listener = mp.Process(target=log_receiver_process, args=(queue,))
    listener.start()
    # mp.spawn prepends the rank automatically, so don't provide it here
    all_args = (world_size, queue, func, args, backend)
    context = mp.spawn(rank_process, all_args, nprocs=world_size, join=False)
    assert context is not None
    current_pids.extend(context.pids())  # Allows kill() to work later if needed
    while not context.join():
        pass
    listener.kill()


def kill() -> int:
    """Kill all child processes spawned by spawn_multiple() and not yet killed.

    Similar to "kill -9 pid" on Linux.

    Very useful when those processes are deadlocked.
    Return the number of processes successfully killed."""
    killed = []
    for pid in current_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            killed.append(pid)
    for pid in killed:
        current_pids.remove(pid)
    return len(killed)


# %%
"""
### Your First Broadcast

Give it a try now! Implement `broadcast_from_0_cpu`, which will run in each process with a different value for `dist.get_rank()`.

Sometimes unsaved changes to your .py file may not be run when you run a code block; make sure to save your .py file before running.
"""
# %%


def broadcast_from_0_cpu(shape: tuple, dist=dist):
    """Broadcast a random tensor of the given shape from rank 0 to others, and log the rank along with the sum of the tensor.

    Remember that only rank 0 generates the random tensor, so use dist.get_rank() to distinguish these cases. Example:
    If the tensor generated is [[-2, 1], [-.5, -1]], log 'Rank 27: got sum -2.5'

    Args:
    - shape: shape of the tensor to create and broadcast
    - dist: the torch.distributed package. This argument is included for testing purposes, you should ignore it.
    """
    "SOLUTION"
    rank = dist.get_rank()
    if rank == 0:
        tensor = t.randn(shape)
    else:
        tensor = t.empty(shape)
    dist.broadcast(tensor, 0)
    logger.info(f"Rank {rank}: got sum {tensor.sum()}")


if MAIN:
    w3d1_test.test_broadcast_from_device(broadcast_from_0_cpu, device="cpu")

# %%
if MAIN and not IS_CI:
    world_size = min(4, os.cpu_count() or 4)
    print(f"Testing broadcast on {world_size} CPUs")
    args = ((1024, 1024),)
    spawn_multiple(broadcast_from_0_cpu, args, world_size=world_size, backend="gloo")


# %%
"""
### Using the Correct GPU

There are multiple ways to ensure your processes are using the correct GPU. Setting the environment variable `CUDA_VISIBLE_DEVICES=2` in the child process means that PyTorch will ignore the existence of any device other than physical GPU 2, which it will call `cuda:0`.

We've done this for you today in the utility code, so your code literally won't be able to see any of the other GPUs and will report `t.cuda.device_count()` as 1. This prevents you from accidentally using the wrong device, which leads to all kinds of confusing bugs.

Implement `broadcast_from_0_gpu`. It's the same as before except create your tensors on CUDA.
"""
# %%


def broadcast_from_0_gpu(shape: tuple, dist=dist):
    """Broadcast a random tensor of the given shape from rank 0 to others, and log the rank along with the sum of the tensor."""
    "SOLUTION"
    rank = dist.get_rank()
    logger.info(f"{rank}: Starting with visible devices: {t.cuda.device_count()}")
    if rank == 0:
        tensor = t.randn(shape, device="cuda")
    else:
        tensor = t.empty(shape, device="cuda")
    dist.broadcast(tensor, 0)
    logger.info(f"Rank {rank}: got sum {tensor.sum()}")


if MAIN:
    w3d1_test.test_broadcast_from_device(broadcast_from_0_gpu, device="cuda")


# %%
if MAIN and not IS_CI:
    world_size = t.cuda.device_count()
    print(f"Testing broadcast on {world_size} GPU")
    args = ((1024, 1024),)
    spawn_multiple(broadcast_from_0_gpu, args, world_size=world_size, backend="gloo")


# %%
"""
## Error Checking With `dist`

First with `gloo` and then with `nccl`, try deliberately making the receiving GPU tensor too small for the data. I've done this many times by accident. What happens?

<details>

<summary>Spoiler - Receiving Buffer too Small</summary>

On my machine, `gloo` raises with a relatively clear error message (although it doesn't indicate the line where the error occurred) and releases its resources.

`nccl` is much worse - it silently causes all the receiving processes to hang until the timeout is reached. The default timeout is something like 30 minutes, but can be configured in the call to `init_process_group`. Interrupting the kernel and even restarting the kernel doesn't remove those processes. To fix it, interrupt the kernel, then use the `kill()` utility to remove those processes and get your GPU memory back.

You can verify that all the processes were indeed killed and GPU memory was released by running `nvidia-smi`. If, for some reason, they weren't killed, try running `pkill python`. (Warning: this command will also kill your Jupyter kernel, so make sure to save all your unsaved changes before running it.)

You can get additional debugging information from `nccl` by setting environment variables:
```python
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_FILE'] = './filename.%h.%p'
```

In my experience, this information is only occasionally useful.

</details>

Now try making the number of bytes correct, but the dtype incorrect. For example, make it a `float16` buffer with double the number of elements, or a `float64` with half the number of elements. What happens?

<details>

<summary>Spoiler - Wrong Dtype</summary>

Both `gloo` and `nccl` copy the bytes in while ignoring the discrepancy. The correct number of bytes are received, but the bytes will be interpreted as the wrong type by the receiver. If you're lucky, one of the bit patterns will be a representation of `NaN` and the sum will be `NaN`, but with decent probability you'll just get a sum of nonsense numbers.

</details>

Fortunately, PyTorch provides a way to detect both of these issues by setting the environment variable `TORCH_DISTRIBUTED_DEBUG=DETAIL`. This does decrease performance, but when debugging it's extremely useful to have enabled.
"""

# %%
"""
## Benchmarking `gloo` and `nccl`

We will now compare the performance of `gloo` and `nccl` on the broadcast operation. To do so, you will write the following:

1. A benchmarking case that sets up and calls the operation to be benchmarked.

2. A benchmarking function that is run on each device, and instantiates and benchmarks the provided case.

Then you'll make a plot with a line for each backend showing throughput (GB/s sent by rank 0) versus the data size. Data size should range between 100KB and around 10GB. Note that you can't allocate the full amount of your GPU memory because some of it is used by PyTorch and the CUDA driver.

First, here are some helper functions and the abstract class for the benchmarking cases. You don't need to write any code here.
"""
# %%


def bytes(n_elems: int):
    """Convert number of float32 elements transfered to bytes"""
    bytes_per_elem = 4
    return bytes_per_elem * n_elems


def throughput(n_elems: int, duration: float):
    """Convert number of elements transferred and duration to throughput (GB/s)"""
    bytes_to_gb = 1 / 2**30
    return bytes(n_elems) * bytes_to_gb / duration


class BenchmarkCase(ABC):
    """
    An abstract class representing a distributed operation to be benchmarked.
    An instance of this class will be instantiated in each process, then the call method will be called and benchmarked.
    """

    @abstractmethod
    def __init__(self, shape: tuple, rank: int, device: Union[t.device, str]) -> None:
        """Set up the test case"""

    @property
    @abstractmethod
    def tensor(self) -> t.Tensor:
        """Returns the tensor used in the distributed operation"""

    @abstractmethod
    def __call__(self, dist=dist) -> None:
        """Call the test case. This is the function to be benchmarked."""

    @property
    def log_info(self) -> str:
        """Additional info to be logged, besides the benchmark results"""
        return f"Tensor Sum: {self.tensor.sum()}"


"""
### Results

To get the timings back into your REPL, create a `mp.Queue` and pass it in for `result_queue`, then have process rank 0 do the timing and enqueue the results. You can use `time.perf_counter()` for a high resolution clock. To add results to the queue, use `result_queue.put(object)`, and to retrieve them use `result_queue.get()`.

Gotcha: when sending data with `dist.broadcast` on GPU, it will return as soon as the data has been queued up on the GPU, which doesn't mean that the receiver has gotten it. This is because GPU operations are performed [asynchronously](https://pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution).

To ensure that the transfer is really complete, add a `t.cuda.synchronize()` after the broadcast and before stopping the timer. This will sleep until all CUDA kernels on the current device are complete.
"""

# %%


class BenchmarkBroadcast(BenchmarkCase):
    def __init__(self, shape: tuple, rank: int, device: Union[t.device, str]) -> None:
        """
        Create a random tensor of the given shape on the rank 0 process and empty tensors on the others
        """
        "SOLUTION"
        if rank == 0:
            self._tensor = t.randn(shape, device=device)
        else:
            self._tensor = t.empty(shape, device=device)

    @property
    def tensor(self) -> t.Tensor:
        "SOLUTION"
        return self._tensor

    def __call__(self, dist=dist) -> None:
        """Broadcast a tensor from the rank 0 process to others"""
        "SOLUTION"
        dist.broadcast(self.tensor, 0)


def benchmark(case: type[BenchmarkCase], shapes: list[tuple], result_queue: mp.Queue, dist=dist) -> None:
    """
    Benchmark the time it takes to perform a distributed operation, and log the time elapsed.
    This function (benchmark) will be run on each spawned process.
     - case: The class of the BenchmarkCase subclass to be benchmarked, which will be instantiated and called..
     - shapes: a list of tensor shapes to benchmark the case for
     - result_queue: a queue to log results to. Log results as tuples of the form (backend: String, shape: tuple, time_elapsed: float)
    """
    "SOLUTION"
    rank = dist.get_rank()
    device = t.device("cuda:0") if t.cuda.is_available() else t.device("cpu")
    for shape in shapes:
        func = case(shape, rank, device)
        logger.info(f"Before func\tRank {rank}\t{func.log_info}")
        start_time = time.perf_counter()
        func()
        if t.cuda.is_available():
            t.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        logger.info(f"After func\tRank {rank}\t{func.log_info}")
        if rank == 0:
            result = (dist.get_backend(), shape, elapsed)
            logger.info(str(result))
            result_queue.put(result)


if MAIN:
    w3d1_test.test_benchmark_broadcast_single(BenchmarkBroadcast)

# %%
if MAIN:
    w3d1_test.test_benchmark_broadcast_multiple(BenchmarkBroadcast)
# %%
if MAIN:
    w3d1_test.test_benchmark(benchmark)

"""
With 4 V100s, `gloo` was marginally faster for the 100K transfer and got absolutely stomped on all the rest, ranging from 70-120 times slower than `nccl`. I don't actually know why this is so bad, but I would speculate that NVLink is maybe not being used at all and the data is going over PCI Express. This means that in practice, `gloo` can't be used for deep learning on NVIDIA GPUs.

For the remainder of today, we'll be using `nccl`, but if you get any weird errors then it's worth trying `gloo` to see if you get a more informative error message.

"""
# %%
"""
## Benchmarking Broadcast For Real

The tests above ran in the `FakeDistributed` simulator which uses multiple threads instead of multiple processes. Next run the benchmark on multiple processes for real.

Create an `mp.Queue` instance and use it to store your performance results: tuples of form `(backend, shape, time_elapsed)`. Recall that `spawn_multiple()` takes as arguments a function (here: `benchmark`), the arguments to that function, the world size, and the backend. You can use t.cuda.device_count() to access the number of GPUs. 

When all benchmark calls finish, create and populate a list `results` with popped out from your queue. This may take a minute to run; reduce the size of shapes for testing purposes as needed.
"""
if MAIN and not IS_CI:
    print("Benchmarking broadcast on multi-GPU")
    shapes = [(256 * 10**power,) for power in range(2, 8)]
    results: list[tuple[str, tuple, float]] = []
    if "SOLUTION":
        world_size = t.cuda.device_count()
        result_queue = mp.Queue(-1)
        args = (BenchmarkBroadcast, shapes, result_queue)
        spawn_multiple(benchmark, args, world_size=world_size, backend="gloo")
        spawn_multiple(benchmark, args, world_size=world_size, backend="nccl")
        for _ in range(2 * len(shapes)):
            results.append(result_queue.get())

if MAIN and not IS_CI:
    assert results
    bytes_transferred = [bytes(shape[0]) for (backend, shape, dt) in results if backend == "gloo"]
    gloo_throughput = [throughput(shape[0], dt) for (backend, shape, dt) in results if backend == "gloo"]
    nccl_throughput = [throughput(shape[0], dt) for (backend, shape, dt) in results if backend == "nccl"]

    fig, ax = plt.subplots()
    ax.loglog(bytes_transferred, gloo_throughput, label="gloo")
    ax.loglog(bytes_transferred, nccl_throughput, label="nccl")
    ax.set(xlabel="Bytes transferred", ylabel="Throughput (GB/s)")
    fig.legend()


# %%
"""
## All-Reduce

We've seen how to send the initial parameters from rank 0 to the other processes using `broadcast`. This same procedure is also how you would send module buffers like batch norm statistics. We only need one more communication to have a working data parallel implementation, which is sending around each GPU's local gradients until each GPU has the sum of all gradients.

This operation is called "all-reduce", and is the communication equivalent of matrix multiply - an expensive, frequently used operation that has been heavily optimized, both in terms of programmer effort on the software side and architecture design on the hardware side. It's fair to say that we couldn't scale deep learning without efficient all-reduce.

**Example**

GPU 0:
```python
x = torch.tensor([1])
dist.all_reduce(x)
# x = [3]
```

GPU 1:
```python
x = torch.tensor([2])
dist.all_reduce(x)
# x = [3]
```

<details>

<summary>Floating Point Rounding Error</summary>

Your implementation should behave identically to the real one, which guarantees the sum to be identical on every rank. This requires some care due to floating point rounding error; floating point addition is not associative in general and the difference is larger in lower precision arithmetic.

Will these three operations result in the same sum? 

```python
if MAIN:
    a = t.tensor(0.1, dtype=t.float16)
    b = t.tensor(0.2, dtype=t.float16)
    c = t.tensor(0.3, dtype=t.float16)
    print((a + b) + c)
    print((a + (b + c)))
    print(t.tensor([0.1, 0.2, 0.3], dtype=t.float16).sum())
```
</details>

### Naive All-Reduce

Implement all_reduce_broadcast in a naive manner: call `broadcast` `world_size` times, then sum the results. The only constraint on the order of summation is that the sum must end up bitwise identical on every rank.
"""

# %%


def all_reduce_broadcast(tensor: t.Tensor, dist=dist) -> None:
    "SOLUTION"
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tensors = [tensor if i == rank else t.empty_like(tensor) for i in range(world_size)]
    for i, cache_tensor in enumerate(tensors):
        dist.broadcast(cache_tensor, i)
    if t.cuda.is_available():
        t.cuda.synchronize()
    # add the tensors to 'tensor' in place in the right order
    before_sum = t.zeros_like(tensor)
    for cache_tensor in tensors[:rank]:
        before_sum.add_(cache_tensor)
    tensor.add_(before_sum)
    for cache_tensor in tensors[(rank + 1) :]:
        tensor.add_(cache_tensor)


if MAIN:
    w3d1_test.test_all_reduce_broadcast(all_reduce_broadcast)

# %%
"""
Now, set up BenchmarkCases to compare the perfomance of your `all_reduce_broadcast` and `dist.all_reduce`.

How much slower is the naive implementation than `dist.all_reduce`?
"""
# %%
class BenchmarkAllReduce(BenchmarkCase):
    """A Benchmark case to benchmark dist.all_reduce"""

    def __init__(self, shape: tuple, rank: int, device: Union[t.device, str]) -> None:
        "SOLUTION"
        self._tensor = t.randn(shape, device=device)

    @property
    def tensor(self) -> t.Tensor:
        "SOLUTION"
        return self._tensor

    def __call__(self, dist=dist) -> None:
        "SOLUTION"
        dist.all_reduce(self.tensor)


class BenchmarkAllReduceBroadcast(BenchmarkCase):
    """A Benchmark case to benchmark all_reduce_broadcast"""

    def __init__(self, shape: tuple, rank: int, device: Union[t.device, str]) -> None:
        "SOLUTION"
        self._tensor = t.randn(shape, device=device)

    @property
    def tensor(self) -> t.Tensor:
        "SOLUTION"
        return self._tensor

    def __call__(self, dist=dist) -> None:
        "SOLUTION"
        all_reduce_broadcast(self.tensor, dist=dist)


if MAIN:
    w3d1_test.test_benchmark_all_reduce(BenchmarkAllReduce)
# %%
if MAIN:
    w3d1_test.test_benchmark_all_reduce(BenchmarkAllReduceBroadcast)

# %%
if MAIN and not IS_CI:
    shapes = [(int(256 * 10**i),) for i in range(3, 7)]
    results_dist: list[tuple[str, tuple, float]] = []
    print("Benchmarking dist allreduce on multi-GPU")
    if "SOLUTION":
        world_size = t.cuda.device_count()
        result_queue = mp.Queue(-1)
        args = (BenchmarkAllReduce, shapes, result_queue)
        spawn_multiple(benchmark, args, world_size=world_size, backend="nccl")
        results_dist.extend([result_queue.get() for _ in range(len(shapes))])

# %%
if MAIN and not IS_CI:
    shapes = [(int(256 * 10**i),) for i in range(3, 7)]
    results_naive: list[tuple[str, tuple, float]] = []
    print("Benchmarking naive broadcast-based allreduce on multi-GPU")
    if "SOLUTION":
        world_size = t.cuda.device_count()
        result_queue = mp.Queue(-1)
        args = (BenchmarkAllReduceBroadcast, shapes, result_queue)
        spawn_multiple(benchmark, args, world_size=world_size, backend="nccl")
        results_naive.extend([result_queue.get() for _ in range(len(shapes))])

if MAIN and not IS_CI:
    # Plot the results
    assert results_naive and results_dist
    bytes_transferred = [bytes(shape[0]) for (backend, shape, dt) in results_naive]
    naive_throughput = [throughput(shape[0], dt) for (backend, shape, dt) in results_naive]
    dist_throughput = [throughput(shape[0], dt) for (backend, shape, dt) in results_dist]
    fig, ax = plt.subplots()
    ax.loglog(bytes_transferred, naive_throughput, label="naive")
    ax.loglog(bytes_transferred, dist_throughput, label="dist")
    ax.set(xlabel="Bytes transferred", ylabel="Throughput (GB/s)")
    fig.legend()

# %%

"""
## Onto Part 2

Now you're ready for Part 2, where we'll adapt a real training loop for data parallelism and try to improve its performance.
"""

# %%
