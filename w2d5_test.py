import torch as t

from utils import allclose, report, allclose_atol, allclose_scalar


@report
def test_get_inputs(get_inputs, model, data):
    import w2d5_solution

    module = model.layers[1].linear2

    expected = w2d5_solution.get_inputs(model, data, module)
    actual = get_inputs(model, data, module)

    allclose(actual, expected)


@report
def test_get_outputs(get_outputs, model, data):
    import w2d5_solution

    module = model.layers[1].linear2

    expected = w2d5_solution.get_outputs(model, data, module)
    actual = get_outputs(model, data, module)

    allclose(actual, expected)


@report
def test_get_out_by_head(get_out_by_head, model, data):
    import w2d5_solution

    layer = 2

    expected = w2d5_solution.get_out_by_head(model, data, layer)
    actual = get_out_by_head(model, data, layer)

    allclose(actual, expected)


@report
def test_get_out_by_component(get_out_by_components, model, data):
    import w2d5_solution

    expected = w2d5_solution.get_out_by_components(model, data)
    actual = get_out_by_components(model, data)

    allclose_atol(actual, expected, 1e-4)


@report
def test_final_ln_fit(model, data, get_ln_fit):
    import w2d5_solution

    expected, exp_r2 = w2d5_solution.get_ln_fit(model, data, model.norm, 0)
    actual, act_r2 = get_ln_fit(model, data, model.norm, 0)

    allclose(t.tensor(actual.coef_), t.tensor(expected.coef_))
    allclose(t.tensor(actual.intercept_), t.tensor(expected.intercept_))
    allclose(act_r2, exp_r2)


@report
def test_pre_final_ln_dir(model, data, get_pre_final_ln_dir):
    import w2d5_solution

    expected = w2d5_solution.get_pre_final_ln_dir(model, data)
    actual = get_pre_final_ln_dir(model, data)
    similarity = t.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    allclose_scalar(similarity, 1.0)


@report
def test_get_WV(model, get_WV):
    import w2d5_solution

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        v = w2d5_solution.get_WV(model, layer, head)
        their_v = get_WV(model, layer, head)
        allclose(their_v, v)


@report
def test_get_WO(model, get_WO):
    import w2d5_solution

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        o = w2d5_solution.get_WO(model, layer, head)
        their_o = get_WO(model, layer, head)
        allclose(their_o, o)


@report
def test_get_pre_20_dir(model, data, get_pre_20_dir):
    import w2d5_solution

    expected = w2d5_solution.get_pre_20_dir(model, data)
    actual = get_pre_20_dir(model, data)

    allclose(actual, expected)


@report
def embedding_test(model, tokenizer, embedding_fn):
    import w2d5_solution

    open_encoding = w2d5_solution.embedding(model, tokenizer, "(")
    closed_encoding = w2d5_solution.embedding(model, tokenizer, ")")

    allclose(embedding_fn(model, tokenizer, "("), open_encoding)
    allclose(embedding_fn(model, tokenizer, ")"), closed_encoding)


@report
def qk_test(model, their_get_q_and_k):
    import w2d5_solution

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        q, k = w2d5_solution.get_Q_and_K(model, layer, head)
        their_q, their_k = their_get_q_and_k(model, layer, head)
        allclose(their_q, q)
        allclose(their_k, k)


@report
def test_qk_calc_termwise(model, tokenizer, their_get_q_and_k):
    import w2d5_solution

    embedding = model.encoder(tokenizer.tokenize(["()()()()"]).to(w2d5_solution.DEVICE)).squeeze()
    expected = w2d5_solution.qk_calc_termwise(model, 0, 0, embedding, embedding)
    actual = their_get_q_and_k(model, 0, 0, embedding, embedding)

    allclose(actual, expected)
