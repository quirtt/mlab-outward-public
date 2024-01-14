# %%
"""
W3D1 Model Preloader

This script prepares weights needed for Part 3. This downloads over 80GB of data and takes quite a while, so start it well before you need it.

It will load the Open Pretrained Transformers (OPT) models from HuggingFace and convert the weights to a format suitable for zero-copy memory mapping.

If you have the file w3c1_data/stats.json locally and this contains data, then this has already been run for you and you shouldn't run it again unless you want to play with it.
"""

import gc

# %%
import json
import os
import re
import time
from collections import defaultdict
import importlib.util

import numpy as np
import torch as t
from matplotlib import pyplot as plt
from torch import nn
from tqdm.auto import tqdm
from w3d1_utils import (
    CONFIGS,
    DEFAULT_PROMPT,
    load_hf_model,
    greedy_sample,
    save_state_dict,
    mlab_weight_dir,
    STAT_FILENAME,
    parameter_size_bytes,
)

MAIN = __name__ == "__main__"

spec = importlib.util.find_spec("accelerate")
if spec is None:
    print(
        "Accelerate not installed - large model loading will be excruciatingly slow. If you're in a hurry, pip install accelerate and try again."
    )
    low_cpu_mem_usage = False
else:
    low_cpu_mem_usage = True

if MAIN:
    # Skip 350m because architecture is different
    # Skip 30b, 66b - just a big download
    MODELS_TO_PRELOAD = list(CONFIGS)[:-2]
    print("Preloading models: ", MODELS_TO_PRELOAD)
    stats = defaultdict(dict)
    local_device = t.device("cuda", 0)
    for i, opt_name in enumerate(MODELS_TO_PRELOAD):
        if t.cuda.get_device_properties(local_device).total_memory < parameter_size_bytes(opt_name):
            print(f"Not enough memory to move {opt_name} to {local_device}, running on CPU.")
            local_device = t.device("cpu")
        stats[opt_name]["device"] = str(local_device)

        model, tokenizer = load_hf_model(
            opt_name, device=local_device, stats=stats, low_cpu_mem_usage=low_cpu_mem_usage
        )

        if local_device.type != "cpu":
            # 30b failed to run on CPU with
            # RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
            # Could probably fix this by casting to float32
            # But for now just don't do the test on CPU, it'd be slow anyway
            inputs = tokenizer(DEFAULT_PROMPT, return_tensors="pt")
            input_ids = inputs["input_ids"].to(local_device)

            start = time.perf_counter()
            model(input_ids)
            t.cuda.synchronize()
            forward_time = time.perf_counter() - start
            stats[opt_name]["forward_time"] = forward_time
            print(f"Cold forward pass took {forward_time:.3f}s")

            start = time.perf_counter()
            n_forwards = 10
            for i in range(n_forwards):
                model(input_ids)
            t.cuda.synchronize()
            forward_time_warm = (time.perf_counter() - start) / n_forwards
            stats[opt_name]["forward_time_warm"] = forward_time_warm
            print(f"Warm forward pass took {forward_time_warm:.3f}s")

            start = time.perf_counter()
            inputs = tokenizer(DEFAULT_PROMPT, return_tensors="pt")
            local_device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(local_device)
            out = greedy_sample(model, input_ids, tokenizer)
            generate_time = time.perf_counter() - start
            stats[opt_name]["generated_text"] = out
            stats[opt_name]["generate_time"] = generate_time
            print(f"Generate 30 tokens took: {forward_time:.3f}s\n{opt_name} said: {out}")

            stats[opt_name]["memory_reserved"] = t.cuda.memory_reserved(local_device)

        save_state_dict(model, mlab_weight_dir(opt_name))
        with open(STAT_FILENAME, "w") as f:
            json.dump(stats, f)

        del model
        del tokenizer
        gc.collect()


# %%
