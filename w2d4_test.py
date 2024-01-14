import torch as t
import torch.nn as nn

from utils import allclose, allclose_atol, report


@report
def test_qk_attn(QK_attn, W_QK, attn_input, attn_pattern):
    with t.inference_mode():
        QK_attn_pattern = QK_attn(W_QK, attn_input)
        allclose(QK_attn_pattern, attn_pattern, rtol=1e-3)


@report
def test_ov_result_mix_before(OV_result_mix_before, W_OV, residual_stream_pre, attn_pattern, expected_head_out):
    with t.inference_mode():
        actual_head_out = OV_result_mix_before(W_OV, residual_stream_pre, attn_pattern)
        allclose(actual_head_out, expected_head_out, rtol=1e-1)


@report
def test_ov_result_mix_after(OV_result_mix_after, W_OV, residual_stream_pre, attn_pattern, expected_head_out):
    with t.inference_mode():
        actual_head_out = OV_result_mix_after(W_OV, residual_stream_pre, attn_pattern)
        allclose(actual_head_out, expected_head_out, rtol=1e-1)


@report
def test_logit_attribution(logit_attribution, model, cache_2, tokens_2, logits_2):
    with t.inference_mode():
        batch_index = 0
        embed = cache_2["hook_embed"][batch_index]
        l1_results = cache_2["blocks.0.attn.hook_result"][batch_index]
        l2_results = cache_2["blocks.1.attn.hook_result"][batch_index]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens_2[batch_index])
        # Uses fancy indexing to get a len(tokens_2[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits_2[batch_index, t.arange(len(tokens_2[0]) - 1), tokens_2[batch_index, 1:]]
        allclose(logit_attr.sum(1), correct_token_logits, rtol=1e-2)
