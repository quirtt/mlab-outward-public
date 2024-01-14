import logging
import logging.handlers
import threading
from io import StringIO
from typing import Callable, Optional, Protocol
import torch as t


__threadsafe_logger = None
__logger_buffer = StringIO()


def make_threadsafe_logger() -> tuple[logging.Logger, StringIO]:
    global __threadsafe_logger
    if __threadsafe_logger is None:
        __threadsafe_logger = logging.getLogger("w3d1_fake_distributed")
        __threadsafe_logger.setLevel(logging.DEBUG)
        # Allows us to assert against log messages in tests - note this is fragile

        log_streamhandler = logging.StreamHandler(__logger_buffer)
        log_streamhandler.setLevel(logging.DEBUG)
        __threadsafe_logger.addHandler(log_streamhandler)
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.DEBUG)
        __threadsafe_logger.addHandler(streamhandler)
    return __threadsafe_logger, __logger_buffer


class Op:
    def __init__(self, dist: "FakeDistributed"):
        self.dist = dist
        self.lock = threading.Lock()
        self._barrier = threading.Barrier(self.dist.get_world_size())
        self.n_executed = 0

    @property
    def rank(self) -> int:
        return self.dist.get_rank()

    def _wait(self):
        with self.lock:
            self.n_executed += 1
        self._barrier.wait()


class BroadcastOp(Op):
    def __init__(self, dist):
        super().__init__(dist)
        self._tensor: Optional[t.Tensor] = None

    def execute(self, tensor, src):
        self.dist.logger.debug(f"{self.rank}: broadcast with shape {tensor.shape}, src={src}")
        if self.rank == src:
            self._tensor = tensor
        self._wait()
        assert self._tensor is not None
        if self.rank != src:
            tensor.copy_(self._tensor)
        self.dist.logger.debug(f"{self.rank}: after broadcast sum {tensor.sum()}")


class AllGatherOp(Op):
    def __init__(self, dist):
        super().__init__(dist)
        self._all_gathers: dict[int, t.Tensor] = {}

    def execute(self, tensor_list, tensor):
        self.dist.logger.debug(f"{self.rank}: all_gather with shape {tensor.shape}.")
        self._all_gathers[self.rank] = tensor
        self._wait()
        for r in range(self.dist.get_world_size()):
            tensor_list[r].copy_(self._all_gathers[r])
        self.dist.logger.debug(f"{self.rank}: after all_gather sum {tensor.sum()}")


class AllReduceOp(Op):
    def __init__(self, dist):
        super().__init__(dist)
        self._total: Optional[t.Tensor] = None
        self.n_executed = 0

    def execute(self, tensor) -> None:
        self.dist.logger.debug(f"{self.rank}: all_reduce with shape {tensor.shape}.")
        with self.lock:
            self._total = tensor if self._total is None else self._total + tensor
        self._wait()
        tensor.copy_(self._total)
        self.dist.logger.debug(f"{self.rank}: after all_reduce sum {tensor.sum()}")


class AbstractDistributed(Protocol):
    """Common interface for code that runs under either FakeDistributed or torch.distributed."""

    logger: logging.Logger
    log: StringIO

    def get_rank(self) -> int:
        ...

    def get_world_size(self) -> int:
        ...

    def all_gather(self, tensor_list: list[t.Tensor], tensor: t.Tensor, group=None, async_op=False) -> None:
        ...

    def broadcast(self, tensor: t.Tensor, src: int, group=None, async_op=False) -> None:
        ...

    def all_reduce(self, tensor: t.Tensor, group=None, async_op=False) -> None:
        ...

    def barrier(self, group=None, async_op=False) -> None:
        ...


def _check_implemented(group, async_op):
    assert group is None, "process groups not implemented for FakeDistributed"
    assert async_op is False, "async op not implemented for FakeDistributed"


class FakeDistributed:
    """Provides an interface like torch.distributed but in one thread.

    - Configures a logger automatically
    - Logs info on communication ops
    - Tracks exceptions raised in child threads
    """

    logger: logging.Logger
    log: StringIO
    exceptions: dict[str, Exception]

    def __init__(self, world_size: int):
        self.logger, self.log = make_threadsafe_logger()
        self.log.truncate(0)
        self.world_size = world_size
        self.exceptions = {}
        self.local_rank = threading.local()
        self.ops: list[Op] = []
        self.sequence_numbers = [-1 for _ in range(self.world_size)]
        self.lock = threading.Lock()
        self._barrier = threading.Barrier(self.world_size)

    def with_rank(self, func: Callable) -> Callable:
        """Return a wrapper that executes in the child thread and sets the rank before calling func."""

        def with_rank_func(*args, **kwargs):
            rank = kwargs.pop("rank")
            self.local_rank.val = rank
            try:
                func(*args, **kwargs)
            except Exception as e:
                self.logger.exception(f"Rank {rank} raised an exception: ")
                self.exceptions[rank] = e

        return with_rank_func

    def get_rank(self) -> int:
        return self.local_rank.val

    def _set_rank(self, rank: int) -> None:
        # Only test harness should call this - you can't do it on a real distributed.
        self.local_rank.val = rank

    def get_world_size(self):
        return self.world_size

    def get_backend(self) -> str:
        return self.__class__.__name__

    def _get_op(self, cls):
        rank = self.get_rank()
        with self.lock:
            self.sequence_numbers[rank] += 1
            seq = self.sequence_numbers[rank]
            if len(self.ops) == seq:
                op = cls(self)
                self.ops.append(op)
            else:
                op = self.ops[seq]
            assert isinstance(op, cls), f"Supposed to call: {op.__class__.__name__} next"
        return op

    def all_gather(self, tensor_list: list[t.Tensor], tensor: t.Tensor, group=None, async_op=False) -> None:
        assert len(tensor_list) == self.get_world_size(), "Expected list with one Tensor per device"
        _check_implemented(group, async_op)
        op = self._get_op(AllGatherOp)
        op.execute(tensor_list, tensor)

    def broadcast(self, tensor: t.Tensor, src: int, group=None, async_op=False) -> None:
        _check_implemented(group, async_op)
        assert 0 <= src < self.get_world_size(), "Src out of range!"
        op = self._get_op(BroadcastOp)
        op.execute(tensor, src)

    def all_reduce(self, tensor: t.Tensor, group=None, async_op=False) -> None:
        _check_implemented(group, async_op)
        op = self._get_op(AllReduceOp)
        op.execute(tensor)

    def barrier(self, group=None, async_op=False) -> None:
        _check_implemented(group, async_op)
        self._barrier.wait()


def launch(target, world_size: int, args=None, kwargs=None, alt_dist=None):
    global dist
    dist = alt_dist if alt_dist is not None else FakeDistributed(world_size=world_size)
    threads = []
    for r in range(dist.get_world_size()):
        thread_target = dist.with_rank(target)
        thread_args = args if args is not None else ()
        thread_kwargs = kwargs if kwargs is not None else {}
        thread_kwargs["rank"] = r
        thread = threading.Thread(target=thread_target, args=thread_args, kwargs=thread_kwargs, name=f"Rank{r}Thread")
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    # This should pass typecheck
    fake: AbstractDistributed = FakeDistributed(1)
    fake_dist: AbstractDistributed

    # TBD lowpri: this should too if we impl async_op
    # import torch.distributed as fake_dist
