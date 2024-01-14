from dataclasses import dataclass
import os
import re
import time
import torch
import torch as t
import torchvision
import numpy as np
from torch import nn
from typing import Iterable, Optional
import sys
from tqdm.auto import tqdm

###############
## Part 2 Utils


def preprocess_data(np_data: np.ndarray, device, dtype) -> torch.Tensor:
    # Convert to torch float16 tensor
    data = torch.tensor(np_data, device=device).to(dtype)

    # Normalize
    mean = torch.tensor([125.31, 122.95, 113.87], device=device).to(dtype)
    std = torch.tensor([62.99, 62.09, 66.70], device=device).to(dtype)
    data = (data - mean) / std

    # Permute data from NHWC to NCHW format
    data = data.permute(0, 3, 1, 2)

    return data


def load_cifar10(device, dtype, data_dir="~/data"):
    if sys.platform == "win32":
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    train = torchvision.datasets.CIFAR10(root=data_dir, download=True)
    valid = torchvision.datasets.CIFAR10(root=data_dir, train=False)
    train_data = preprocess_data(train.data, device, dtype)
    valid_data = preprocess_data(valid.data, device, dtype)
    train_targets = torch.tensor(train.targets).to(device)
    valid_targets = torch.tensor(valid.targets).to(device)
    # Pad 32x32 to 40x40
    train_data = nn.ReflectionPad2d(4)(train_data)
    return train_data, train_targets, valid_data, valid_targets


def data_augmentation(data, batch_size) -> t.Tensor:
    data = torch.cat(
        [random_crop(data[i : i + batch_size], crop_size=(32, 32)) for i in range(0, len(data), batch_size)]
    )
    data[: len(data) // 2] = torch.flip(data[: len(data) // 2], [-1])
    return data


def update_ema(train_model, valid_model, rho):
    # The trained model is not used for validation directly. Instead, the
    # validation model weights are updated with exponential moving averages.
    train_weights = train_model.state_dict().values()
    valid_weights = valid_model.state_dict().values()
    for train_weight, valid_weight in zip(train_weights, valid_weights):
        if valid_weight.dtype in [torch.float16, torch.float32]:
            valid_weight *= rho
            valid_weight += (1 - rho) * train_weight


def random_crop(data, crop_size):
    crop_h, crop_w = crop_size
    h = data.size(2)
    w = data.size(3)
    x = torch.randint(w - crop_w, size=(1,))[0]
    y = torch.randint(h - crop_h, size=(1,))[0]
    return data[:, :, y : y + crop_h, x : x + crop_w]


def validation(valid_model, valid_data, valid_targets, batch_size, epoch, batch_count, train_time):
    valid_correct = []
    valid_model.train(False)
    for i in range(0, len(valid_data), batch_size):

        # Test time agumentation: Test model on regular and flipped data
        regular_inputs = valid_data[i : i + batch_size]
        flipped_inputs = torch.flip(regular_inputs, [-1])

        logits1 = valid_model(regular_inputs).detach()
        logits2 = valid_model(flipped_inputs).detach()

        # Final logits are average of augmented logits
        logits = torch.mean(torch.stack([logits1, logits2], dim=0), dim=0)

        # Compute correct predictions
        correct = logits.max(dim=1)[1] == valid_targets[i : i + batch_size]

        valid_correct.append(correct.detach().type(torch.float64))

    # Accuracy is average number of correct predictions
    valid_acc = torch.mean(torch.cat(valid_correct)).item()

    print(f"{epoch:5} {batch_count:8d} {train_time:19.2f} {valid_acc:22.4f}")
    return valid_acc


class CustomOptimizer:
    def __init__(
        self, params: Iterable[t.nn.parameter.Parameter], momentum: float, weight_decay: float, weight_decay_bias: float
    ):
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.weight_decay_bias = weight_decay_bias

        self.lr_schedule = torch.cat(
            [
                torch.linspace(0e0, 2e-3, 194),
                torch.linspace(2e-3, 2e-4, 582),
            ]
        )
        self.lr_schedule_for_biases = 64.0 * self.lr_schedule
        params = list(params)
        self.weights = [(w, torch.zeros_like(w)) for w in params if w.requires_grad and len(w.shape) > 1]
        self.biases = [(w, torch.zeros_like(w)) for w in params if w.requires_grad and len(w.shape) <= 1]
        self.num_steps = 0

    def step(self) -> None:
        lr_index = min(self.num_steps, len(self.lr_schedule) - 1)
        lr = self.lr_schedule[lr_index]
        lr_bias = self.lr_schedule_for_biases[lr_index]
        update_nesterov(self.weights, lr, self.weight_decay, self.momentum)
        update_nesterov(self.biases, lr_bias, self.weight_decay_bias, self.momentum)
        self.num_steps += 1


def update_nesterov(weights, lr, weight_decay, momentum):
    for weight, velocity in weights:
        if weight.requires_grad:
            gradient = weight.grad.data
            weight = weight.data

            gradient.add_(weight, alpha=weight_decay).mul_(-lr)
            velocity.mul_(momentum).add_(gradient)
            weight.add_(gradient.add_(velocity, alpha=momentum))


## END PART 2 UTILS
###################


###############
## PART 3 UTILS
from w2d3_part1_loading_solution import GPTConfig
from transformers import OPTForCausalLM, AutoTokenizer
from typing import cast, Any
from collections import defaultdict
from contextlib import contextmanager

MAIN = __name__ == "__main__"
DATA_ROOT = "./w3d1_data"
STAT_FILENAME = os.path.join(DATA_ROOT, "stats.json")
DEFAULT_PROMPT = "Hey, are you conscious? Can you talk to me?"
CONFIGS = {
    "facebook/opt-125m": GPTConfig(
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        max_position_embeddings=2048,
        vocab_size=50272,
        dropout=0.0,
        activation_function="relu",
    ),
    "facebook/opt-1.3b": GPTConfig(
        num_layers=24,
        num_heads=32,
        hidden_size=2048,
        max_position_embeddings=2048,
        vocab_size=50272,
        dropout=0.0,
        activation_function="relu",
    ),
    "facebook/opt-2.7b": GPTConfig(
        num_layers=32,
        num_heads=32,
        hidden_size=2560,
        max_position_embeddings=2048,
        vocab_size=50272,
        dropout=0.0,
        activation_function="relu",
    ),
    "facebook/opt-6.7b": GPTConfig(
        num_layers=32,
        num_heads=32,
        hidden_size=4096,
        max_position_embeddings=2048,
        vocab_size=50272,
        dropout=0.0,
        activation_function="relu",
    ),
    "facebook/opt-13b": GPTConfig(
        num_layers=40,
        num_heads=40,
        hidden_size=5120,
        max_position_embeddings=2048,
        vocab_size=50272,
        dropout=0.0,
        activation_function="relu",
    ),
    "facebook/opt-30b": GPTConfig(
        num_layers=48,
        num_heads=56,
        hidden_size=7168,
        max_position_embeddings=2048,
        vocab_size=50272,
        dropout=0.0,
        activation_function="relu",
    ),
    "facebook/opt-66b": GPTConfig(
        num_layers=64,
        num_heads=72,
        hidden_size=9216,
        max_position_embeddings=2048,
        vocab_size=50272,
        dropout=0.0,
        activation_function="relu",
    ),
}


def parameter_size_bytes(opt_name: str):
    m = re.match(r"facebook/opt-(.+?)(m|b)", opt_name)
    assert m is not None
    n, unit = m.groups()
    return 2 * float(n) * (2**20 if unit == "m" else 2**30)


def hf_cache_dir(opt_name: str) -> str:
    return os.path.join(DATA_ROOT, opt_name)


def get_tokenizer(opt_name: str, local_files_only=False):
    """
    As of transformers 4.21.1, fast tokenizers don't work on some of these models.
    """
    return AutoTokenizer.from_pretrained(opt_name, local_files_only=local_files_only, use_fast=False)


def load_hf_model(
    opt_name: str, device: t.device, stats: defaultdict, local_files_only=False, low_cpu_mem_usage=False
) -> tuple[OPTForCausalLM, Any]:
    print("Loading HuggingFace model", opt_name)
    # 13b only gives 'Currenty GPT2's fast tokenizer does NOT support adding a BOS token.'
    # What's up with that?
    tokenizer = get_tokenizer(opt_name, local_files_only=local_files_only)
    os.makedirs(hf_cache_dir(opt_name), exist_ok=True)

    start = time.perf_counter()
    pretrained: OPTForCausalLM = cast(
        OPTForCausalLM,
        OPTForCausalLM.from_pretrained(
            opt_name,
            cache_dir=hf_cache_dir(opt_name),
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only,
        ),
    )
    pretrained.eval()
    load_secs = time.perf_counter() - start
    stats[opt_name]["load_secs"] = load_secs
    print(f"Loaded pretrained to CPU in {load_secs:.3f}s")

    start = time.perf_counter()
    pretrained.to(device)
    t.cuda.synchronize()
    gpu_load_secs = time.perf_counter() - start
    stats[opt_name]["gpu_load_secs"] = gpu_load_secs
    print(f"Loaded pretrained to GPU in {gpu_load_secs:.3f}s")

    our_config = CONFIGS[opt_name]
    assert pretrained.config.num_hidden_layers == our_config.num_layers
    assert pretrained.config.num_attention_heads == our_config.num_heads
    assert pretrained.config.hidden_size == our_config.hidden_size
    assert pretrained.config.max_position_embeddings == our_config.max_position_embeddings
    assert pretrained.config.vocab_size == our_config.vocab_size
    assert pretrained.config.activation_function == our_config.activation_function
    return pretrained, tokenizer


def greedy_sample(model: OPTForCausalLM, input_ids: t.Tensor, tokenizer, num_tokens=30) -> str:
    generate_ids = model.generate(
        input_ids, min_length=len(input_ids) + num_tokens, max_new_tokens=num_tokens, num_beams=1, do_sample=False
    )
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def mlab_weight_dir(opt_name: str) -> str:
    return os.path.join(DATA_ROOT, opt_name.replace("facebook", "mlab"))


def get_path(folder: str, param_name: str) -> str:
    """Return the path for the given state dict key."""
    return f"{folder}/{param_name}.npy"


def save_state_dict(pretrained: OPTForCausalLM, folder: str) -> None:
    """Save all the decoder params to disk in npy format."""
    os.makedirs(folder, exist_ok=True)
    sd = pretrained.model.decoder.state_dict()
    # Concatenate qkv for each layer
    for i in range(pretrained.config.num_hidden_layers):
        wq = sd.pop(f"layers.{i}.self_attn.q_proj.weight")
        wk = sd.pop(f"layers.{i}.self_attn.k_proj.weight")
        wv = sd.pop(f"layers.{i}.self_attn.v_proj.weight")
        bq = sd.pop(f"layers.{i}.self_attn.q_proj.bias")
        bk = sd.pop(f"layers.{i}.self_attn.k_proj.bias")
        bv = sd.pop(f"layers.{i}.self_attn.v_proj.bias")
        sd[f"layers.{i}.self_attn_qkv.weight"] = t.cat((wq, wk, wv), dim=0)
        sd[f"layers.{i}.self_attn_qkv.bias"] = t.cat((bq, bk, bv), dim=0)

    for param_name, param in tqdm(sd.items()):
        arr = param.detach().cpu().numpy()

        if "embed_positions.weight" in param_name and param.shape[0] == 2050:
            print("Special case: removing extra part of positional embedding: ", param_name)
            arr = arr[2:]

        np.save(get_path(folder, param_name), arr, allow_pickle=False)


# %%
def mmap_parameter(folder: str, param_name: str) -> t.Tensor:
    """Memory map the parameter with the specified name.

    The returned tensor is a read only, lazy view into the disk.
    """
    arr = np.load(get_path(folder, param_name), mmap_mode="r")
    assert isinstance(arr, np.ndarray)
    tensor = t.as_tensor(arr)
    return tensor


@contextmanager
def init_on_meta_device():
    """Context manager that hooks register_parameter and replaces the param with one on the meta device."""
    orig_register_parameter = nn.Module.register_parameter

    def register_meta_parameter(module: nn.Module, name: str, param: nn.Parameter) -> None:
        orig_register_parameter(module, name, param)
        if param is not None:
            # see accelerate/big_modeling.py
            # TBD: Is there a reason we can't rely on param provided here and need the _parameters lookup?
            param_cls = type(module._parameters[name])
            assert issubclass(param_cls, nn.Parameter)
            kwargs = module._parameters[name].__dict__
            new_param = param_cls(module._parameters[name].to(t.device("meta")), **kwargs)  # type: ignore
            assert isinstance(new_param, nn.Parameter)
            module._parameters[name] = new_param

    try:
        nn.Module.register_parameter = register_meta_parameter  # type: ignore
        yield
    finally:
        nn.Module.register_parameter = orig_register_parameter


def init_on_device(module: nn.Module, name: str, x: t.Tensor) -> None:
    """Reinitialize the parameter with the specified name using the value and device.

    This works in the case where the parameter is currently on the meta device.
    """
    current = module._parameters[name]
    if current is not None and current.shape != x.shape:
        print("Overwriting parameter of shape {current.shape} with {x.shape}. ")
    param_cls = type(module._parameters[name])  # Handle subclasses of nn.Parameter
    assert issubclass(param_cls, nn.Parameter)
    kwargs = module._parameters[name].__dict__  # usually empty but just in case
    new_param = param_cls(x, **kwargs)
    assert isinstance(new_param, nn.Parameter), (new_param, type(new_param))
    module._parameters[name] = new_param


@dataclass
class UniAttnWeights:
    """Convenience class for passing around weights used for unidirectional attention."""

    total_num_heads: int
    hidden_size: int
    qkv_weight: t.Tensor
    qkv_bias: Optional[t.Tensor]
    out_proj_weight: t.Tensor
    out_proj_bias: Optional[t.Tensor]

    def to(self, dtype: t.dtype) -> "UniAttnWeights":
        return UniAttnWeights(
            self.total_num_heads,
            self.hidden_size,
            self.qkv_weight.to(dtype=dtype),
            self.qkv_bias.to(dtype=dtype) if self.qkv_bias is not None else None,
            self.out_proj_weight.to(dtype=dtype),
            self.out_proj_bias.to(dtype=dtype) if self.out_proj_bias is not None else None,
        )
