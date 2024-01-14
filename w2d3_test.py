import time

import torch as t

from utils import allclose, allclose_scalar, assert_all_equal, report


@report
def test_unidirectional_attn(UnidirectionalAttention, batch_size=2, seq_len=5, hidden_size=6, num_heads=2):
    encodings = t.randn(batch_size, seq_len, hidden_size)
    import w2d3_part1_loading_solution

    ref = w2d3_part1_loading_solution.UnidirectionalAttention(hidden_size, num_heads, dropout=0.0)
    expected = ref(encodings)

    yours = UnidirectionalAttention(hidden_size, num_heads, dropout=0.0)
    for p in yours.parameters():
        p.requires_grad = False

    w2d3_part1_loading_solution._copy_weight_bias(yours.qkv_proj, ref.qkv_proj)
    w2d3_part1_loading_solution._copy_weight_bias(yours.output_proj, ref.output_proj)

    actual = yours(encodings)

    allclose(actual, expected)


@report
def test_gpt_block(GPT2Block):
    import w2d3_part1_loading_solution

    x = t.randn(1, 5, 48)
    _block = w2d3_part1_loading_solution.GPT2Block(
        hidden_size=48,
        layer_norm_epsilon=1e-4,
        dropout=0.0,
        num_heads=4,
        activation_function="relu",
    )
    _block.eval()
    _out = _block(x)

    block = GPT2Block(
        hidden_size=48,
        layer_norm_epsilon=1e-4,
        dropout=0.0,
        num_heads=4,
        activation_function="relu",
    )
    block.eval()
    for p in block.parameters():
        p.requires_grad = False

    w2d3_part1_loading_solution._copy_weight_bias(block.linear1, _block.linear1)
    w2d3_part1_loading_solution._copy_weight_bias(block.linear2, _block.linear2)
    w2d3_part1_loading_solution._copy_weight_bias(block.ln1, _block.ln1)
    w2d3_part1_loading_solution._copy_weight_bias(block.ln2, _block.ln2)
    w2d3_part1_loading_solution._copy_weight_bias(block.attn.qkv_proj, _block.attn.qkv_proj)
    w2d3_part1_loading_solution._copy_weight_bias(block.attn.output_proj, _block.attn.output_proj)

    out = block(x)

    allclose(out, _out)


@report
def test_gpt(GPT2):
    import w2d3_part1_loading_solution

    config = w2d3_part1_loading_solution.GPTConfig(
        num_layers=2,
        num_heads=4,
        vocab_size=100,
        hidden_size=64,
        max_position_embeddings=32,
        dropout=0.0,
        layer_norm_epsilon=1e-4,
    )
    x = t.randint(0, config.vocab_size, (1, 5))

    _gpt = w2d3_part1_loading_solution.GPT2(config)
    _gpt.eval()
    _output = _gpt(x)

    gpt = GPT2(config)
    gpt.eval()
    for p in gpt.parameters():
        p.requires_grad = False

    gpt.token_embedding.weight.copy_(_gpt.token_embedding.weight)
    gpt.pos_embedding.weight.copy_(_gpt.pos_embedding.weight)
    w2d3_part1_loading_solution._copy_weight_bias(gpt.ln, _gpt.ln)

    for my_block, ref_block in zip(gpt.blocks, _gpt.blocks):  # type: ignore
        w2d3_part1_loading_solution._copy_weight_bias(my_block.ln1, ref_block.ln1)
        w2d3_part1_loading_solution._copy_weight_bias(my_block.attn.qkv_proj, ref_block.attn.qkv_proj)
        w2d3_part1_loading_solution._copy_weight_bias(my_block.attn.output_proj, ref_block.attn.output_proj)
        w2d3_part1_loading_solution._copy_weight_bias(my_block.ln2, ref_block.ln2)
        w2d3_part1_loading_solution._copy_weight_bias(my_block.linear1, ref_block.linear1)
        w2d3_part1_loading_solution._copy_weight_bias(my_block.linear2, ref_block.linear2)

    output = gpt(x)

    allclose(_output, output)


@report
def test_load_pretrained_weights(model, tokenizer):
    def encode(text: str) -> t.Tensor:
        """Return a Tensor of shape (batch=1, seq)."""
        return tokenizer(text, return_tensors="pt")["input_ids"]

    prompt = "Former President of the United States of America, George"
    input_ids = encode(prompt)
    with t.inference_mode():
        logits = model(input_ids)[0, -1]
    topk = t.topk(logits, k=10).indices
    next_tokens = tokenizer.batch_decode(topk.reshape(-1, 1))
    print("Prompt: ", prompt)
    print("Your model's top 10 predictions: ", next_tokens)
    assert " Washington" in next_tokens
    assert " Bush" in next_tokens


@report
def test_sample_zero_temperature(model, tokenizer, sample_tokens):
    prompt = "Jingle bells, jingle bells, jingle all the way"
    print("Greedy decoding with prompt: ", prompt)
    output = sample_tokens(model, tokenizer, prompt, temperature=0, max_tokens_generated=8)
    print(f"Your model said: {output}")
    expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    assert output == expected

    print("Greedy decoding a second time (should be deterministic): ")
    output = sample_tokens(model, tokenizer, prompt, temperature=0, max_tokens_generated=8)
    print(f"Your model said: {output}")
    expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
    assert output == expected


@report
def test_identical_output_with_cache(model, tokenizer, initial_text, sample_tokens, sample_tokens_with_cache):
    t.manual_seed(1010)
    start = time.perf_counter()
    out = sample_tokens(
        model,
        tokenizer,
        initial_text,
        temperature=0,
        max_tokens_generated=15,
        stop_at_eos=False,
    )
    print(f"Elapsed: {time.perf_counter() - start:.3f}")
    print("Output without cache:", out)

    t.manual_seed(1010)
    start = time.perf_counter()
    out_cache = sample_tokens_with_cache(
        model,
        tokenizer,
        initial_text,
        temperature=0,
        max_tokens_generated=15,
        stop_at_eos=False,
    )
    print(f"Elapsed: {time.perf_counter() - start:.3f}")
    print("Output with cache: ", out_cache)

    assert out == out_cache


@report
def test_beam_search(beam_search_fn, **beam_search_kwargs):
    import w2d3_part2_sampling_solution

    print("Running reference solution: ")
    expected = w2d3_part2_sampling_solution.beam_search(**beam_search_kwargs)

    print("Running your solution: ")
    actual = beam_search_fn(**beam_search_kwargs)

    for (escore, eids), (ascore, aids) in zip(expected, actual):
        allclose_scalar(escore, ascore)
        assert_all_equal(eids, aids)

    return actual
