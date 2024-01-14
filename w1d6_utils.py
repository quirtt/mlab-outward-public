import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import torch

# must be called before using pycuda. (theoretically torch.cuda.init() is
# supposed to work, but it doesn't)
x = torch.tensor(1).cuda()


class Holder(drv.PointerHolderBase):  # type: ignore
    """A wrapper for t.Tensor so that PyCUDA knows the pointer to its data."""

    def __init__(self, tensor: torch.Tensor):
        super(Holder, self).__init__()
        self.tensor = tensor
        self.gpudata = tensor.data_ptr()

    def get_pointer(self) -> int:
        return self.tensor.data_ptr()


def ceil_divide(a: int, b: int) -> int:
    return (a + b - 1) // b


def load_module(filename: str, no_extern_c=False) -> SourceModule:
    with open(filename) as f:
        return SourceModule(f.read(), no_extern_c=no_extern_c)
