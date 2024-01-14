import time
from typing import Union

import torch as t
import torch.multiprocessing as mp
from einops import rearrange

from utils import report, assert_all_equal
from w3d1_fake_distributed import FakeDistributed, launch
from w3d1_utils import UniAttnWeights

dist = FakeDistributed(world_size=4)

SUM_ANNOUNCER = "after broadcast sum"


def get_sum(log: str, sum_announcer=SUM_ANNOUNCER, sum_offset=1):
    loc = log.find(sum_announcer)
    loc += len(sum_announcer) + sum_offset
    sum = float(log[loc : (loc + 10)])
    return sum


@report
def test_broadcast_from_device(broadcast_from_0_cpu, device="cpu") -> None:
    if device == "cuda" and not t.cuda.is_available():
        return

    dist = FakeDistributed(world_size=4)
    shape = (256,)
    args = (shape, dist)
    launch(broadcast_from_0_cpu, world_size=4, args=args, alt_dist=dist)
    log_out = dist.log.getvalue()
    sum = get_sum(log_out)
    for r in range(dist.get_world_size()):
        assert f"{r}: broadcast with shape torch.Size([256]), src=0" in log_out
        assert f"{r}: {SUM_ANNOUNCER} {sum}" in log_out


def benchmark(case: type, shapes: list[tuple], result_queue: mp.Queue, dist, logger) -> None:
    rank = dist.get_rank()
    device = t.device("cuda:0") if t.cuda.is_available() else t.device("cpu")
    for shape in shapes:
        func = case(shape, rank, device)
        logger.info(f"Before func\tRank {rank}\t{func.log_info}")
        start_time = time.perf_counter()
        func(dist=dist)
        elapsed = time.perf_counter() - start_time
        logger.info(f"After func\tRank {rank}\t{func.log_info}")
        if rank == 0:
            result = (dist.get_backend(), shape, elapsed)
            logger.info(str(result))
            result_queue.put(result)


class BenchmarkTest:
    def __init__(self, shape: tuple, rank: int, device: Union[t.device, str]) -> None:
        if rank == 0:
            self._tensor = t.randn(shape, device=device)
        else:
            self._tensor = t.empty(shape, device=device)

    @property
    def tensor(self) -> t.Tensor:
        return self._tensor

    @property
    def log_info(self) -> str:
        """Additional info to be logged, besides the benchmark results"""
        return f"Tensor Sum: {self.tensor.sum()}"

    def __call__(self, dist=dist) -> None:
        """Broadcast a tensor from the rank 0 process to others"""
        dist.broadcast(self.tensor, 0)


@report
def test_benchmark_broadcast_single(BenchmarkBroadcast) -> None:
    dist = FakeDistributed(world_size=4)
    results_queue = mp.Queue(-1)
    shapes = [(2**x,) for x in range(2, 3)]
    args = (BenchmarkBroadcast, shapes, results_queue, dist, dist.logger)
    launch(benchmark, world_size=4, args=args, alt_dist=dist)
    log_out = dist.log.getvalue()
    sum = get_sum(log_out)
    for r in range(dist.get_world_size()):
        for s in shapes:
            assert f"{r}: broadcast with shape torch.Size([{int(s[0])}]), src=0" in log_out
            assert f"{r}: {SUM_ANNOUNCER} {sum}" in log_out


@report
def test_benchmark_broadcast_multiple(BenchmarkBroadcast) -> None:
    dist = FakeDistributed(world_size=4)
    results_queue = mp.Queue(-1)
    shapes = [(2**x,) for x in range(2, 5)]
    args = (BenchmarkBroadcast, shapes, results_queue, dist, dist.logger)
    launch(benchmark, world_size=4, args=args, alt_dist=dist)
    log_out = dist.log.getvalue()
    for r in range(dist.get_world_size()):
        for s in shapes:
            assert f"{r}: broadcast with shape torch.Size([{int(s[0])}]), src=0" in log_out


@report
def test_benchmark(benchmark):
    results_queue = mp.Queue(-1)
    shapes = [(2**x,) for x in range(2, 5)]
    args = (BenchmarkTest, shapes, results_queue)
    kwargs = dict(dist=dist)
    launch(benchmark, world_size=4, args=args, kwargs=kwargs, alt_dist=dist)
    log_out = dist.log.getvalue()
    for r in range(dist.get_world_size()):
        for s in shapes:
            assert f"{r}: broadcast with shape torch.Size([{int(s[0])}]), src=0" in log_out
    result = results_queue.get()
    assert isinstance(result[0], str)
    assert isinstance(result[1], tuple)
    assert isinstance(result[2], float)


@report
def test_benchmark_all_reduce(BenchmarkAllReduce) -> None:
    dist = FakeDistributed(world_size=4)
    results_queue = mp.Queue(-1)
    shapes = [(2**x,) for x in range(2, 3)]
    args = (BenchmarkAllReduce, shapes, results_queue, dist, dist.logger)
    launch(benchmark, world_size=4, args=args, alt_dist=dist)
    log_out = dist.log.getvalue()
    sum = get_sum(log_out, "After func", 19)
    for r in range(dist.get_world_size()):
        assert f"Rank {r}\tTensor Sum: {sum}" in log_out


@report
def test_all_reduce_broadcast(all_reduce_broadcast) -> None:
    class BenchmarkAllReduceBroadcast:
        def __init__(self, shape: tuple, rank: int, device: Union[t.device, str]) -> None:
            self._tensor = t.randn(shape, device=device)

        @property
        def tensor(self) -> t.Tensor:
            return self._tensor

        def __call__(self, dist=dist) -> None:
            all_reduce_broadcast(self.tensor, dist=dist)

        @property
        def log_info(self) -> str:
            """Additional info to be logged, besides the benchmark results"""
            return f"Tensor Sum: {self.tensor.sum()}"

    test_benchmark_all_reduce(BenchmarkAllReduceBroadcast)


@report
def test_linear_split_columns(linear_split_columns, weight, bias) -> None:
    for n in [1, 2, 4]:
        slices = [linear_split_columns(weight, bias, i, n) for i in range(n)]
        assert_all_equal(t.cat([w for w, b in slices], dim=0), weight)
        assert_all_equal(t.cat([b for w, b in slices], dim=0), bias)


@report
def test_linear_split_rows(linear_split_rows, weight, bias) -> None:
    for n in [1, 2, 4]:
        ws = []
        bs = []
        for i in range(n):
            w, b = linear_split_rows(weight, bias, i, n)
            ws.append(w)
            if b is not None:
                bs.append(b)

        assert_all_equal(t.cat(ws, dim=1), weight)
        assert_all_equal(t.cat(bs, dim=0), bias)


@report
def test_embedding_split_vocab(embedding_split_vocab, weight) -> None:
    for n in [1, 2, 4]:
        slices = [embedding_split_vocab(weight, i, n) for i in range(n)]
        assert_all_equal(t.cat(slices, dim=0), weight)


@report
def test_uni_split_heads(uni_split_heads, attn_weights: UniAttnWeights):
    for n in [1, 2, 4]:
        parts: list[UniAttnWeights] = [uni_split_heads(attn_weights, rank, n) for rank in range(n)]

        actual_qkv = t.cat(
            [
                rearrange(
                    p.qkv_weight,
                    "(qkv head dim) hidden -> qkv head dim hidden",
                    qkv=3,
                    head=attn_weights.total_num_heads,
                )
                for p in parts
            ],
            dim=1,
        )
        actual_qkv = rearrange(actual_qkv, "qkv head dim hidden -> (qkv head dim) hidden")
        assert_all_equal(actual_qkv, attn_weights.qkv_weight)

        if attn_weights.qkv_bias is not None:
            actual_qkv_bias = t.cat(
                [
                    rearrange(
                        p.qkv_bias,
                        "(qkv head dim) -> qkv head dim",
                        qkv=3,
                        head=attn_weights.total_num_heads,
                    )
                    for p in parts
                    if p.qkv_bias is not None
                ],
                dim=1,
            )
            actual_qkv_bias = rearrange(actual_qkv_bias, "qkv head dim -> (qkv head dim)")
            assert_all_equal(actual_qkv_bias, attn_weights.qkv_bias)

        actual_out_proj = t.cat([p.out_proj_weight for p in parts], dim=1)
        assert_all_equal(actual_out_proj, attn_weights.out_proj_weight)

        if attn_weights.out_proj_bias is not None:
            bs: list[t.Tensor] = []
            for p in parts:
                if p.out_proj_bias is not None:
                    bs.append(p.out_proj_bias)
            assert_all_equal(t.cat(bs), attn_weights.out_proj_bias)
