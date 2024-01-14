import torch as t
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import rearrange, repeat

from utils import allclose, allclose_atol, allclose_scalar, report


def flat(x: t.Tensor) -> t.Tensor:
    """Helper function for combining batch and sequence dimensions."""
    return rearrange(x, "b s ... -> (b s) ...")


def unflat(x: t.Tensor, max_seq: int) -> t.Tensor:
    """Helper function for separating batch and sequence dimensions."""
    return rearrange(x, "(b s) ... -> b s ...", s=max_seq)


@report
def test_random_mask(random_mask, input_size=10000, max_seq=128):
    print("Testing empirical frequencies - if this test fails try increasing the input size")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    input_ids = t.randint(0, tokenizer.vocab_size, (input_size, max_seq))
    select_frac = 0.85
    mask_frac = 0.75
    random_frac = 0.10

    masked, was_selected = random_mask(
        input_ids, tokenizer.mask_token_id, tokenizer.vocab_size, select_frac, mask_frac, random_frac
    )

    print('Checking fraction of tokens selected...')
    actual_selected_frac = was_selected.float().mean().item()
    allclose_scalar(actual_selected_frac, select_frac)

    print('Checking fraction of tokens masked...')
    actual_mask_frac = (masked == tokenizer.mask_token_id).float().mean().item()
    expected_mask_frac = select_frac * mask_frac
    allclose_scalar(actual_mask_frac, expected_mask_frac)

    print('Checking fraction of tokens masked OR randomized...')
    changed_frac = (masked != input_ids).float().mean().item()
    expected_changed_frac = select_frac * (mask_frac + random_frac)
    allclose_scalar(changed_frac, expected_changed_frac)


@report
def test_cross_entropy_selected(cross_entropy_selected):
    t.manual_seed(0)

    shape = (3, 4, 10)
    pred = t.randn(*shape)
    y = t.randint(0, 10, shape[:-1])

    # none selected
    selected = t.zeros(shape[:-1], dtype=t.int)
    theirs = cross_entropy_selected(pred, y, selected)
    assert theirs.isnan().all()

    # all selected
    selected = t.ones(shape[:-1], dtype=t.int)
    theirs = cross_entropy_selected(pred, y, selected)
    ours = F.cross_entropy(flat(pred), flat(y))
    print(theirs, ours)
    assert theirs == ours

    # some selected
    selected = (t.rand(shape[:-1]) > 0.5).int()
    theirs = cross_entropy_selected(pred, y, selected)
    s_pred = flat(pred)[flat(selected).bool()]
    s_y = flat(y)[flat(selected).bool()]
    ours = F.cross_entropy(s_pred, s_y)
    print(theirs, ours)
    assert theirs == ours
