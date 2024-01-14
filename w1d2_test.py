import torch as t
from utils import report, allclose, assert_all_equal, allclose_atol


@report
def test_trace(trace_fn):
    """Trace should be equal to the sum along the main diagonal."""
    for n in range(10):
        assert trace_fn(t.zeros((n, n), dtype=t.long)) == 0
        assert trace_fn(t.eye(n, dtype=t.long)) == n

        x = t.randint(0, 10, (n, n))
        expected = t.trace(x)
        actual = trace_fn(x)
        assert actual == expected


@report
def test_trace_expand(trace_fn):
    """Trace should work on a view created by torch.expand."""
    x = t.tensor([1]).expand((5, 5))
    assert trace_fn(x) == 5


@report
def test_trace_transpose(trace_fn):
    """Trace should work on a view created by transpose."""
    x = t.arange(16).reshape((4, 4)).T
    assert trace_fn(x) == 30


@report
def test_matmul(matmul_fn):
    """A simple 2x2 matmul should work and can be verified by hand."""
    x = t.arange(4).reshape(2, 2)
    expected = t.tensor([[2, 3], [6, 11]])
    assert_all_equal(matmul_fn(x, x), expected)


@report
def test_matmul_transpose(matmul_fn):
    """Matmul of a transposed tensor should work."""
    x = t.arange(6).view((2, 3))
    expected = t.tensor([[5, 14], [14, 50]])
    assert_all_equal(matmul_fn(x, x.T), expected)


@report
def test_matmul_expand(matmul_fn):
    """Matmul should work on a view created by torch.expand."""
    x = t.arange(3).expand((2, 3))
    expected = t.tensor([[5, 5], [5, 5]])
    assert_all_equal(matmul_fn(x, x.T), expected)


@report
def test_matmul_skip(matmul_fn):
    """Matmul should work on a view of a larger tensor with all strides > 1."""
    big = t.arange(-8, 8)
    x = big.as_strided(size=(2, 2), stride=(2, 4), storage_offset=8)
    y = big.as_strided(size=(2, 2), stride=(3, 2), storage_offset=8)
    expected = t.tensor([[12, 20], [18, 34]])
    assert_all_equal(matmul_fn(x, y), expected)


@report
def test_conv1d_minimal(conv1d_minimal, n_tests=20):
    # TBD lowpri: check transpose and expand here as well - should be fine
    import numpy as np

    for _ in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 30)
        ci = np.random.randint(1, 5)
        co = np.random.randint(1, 5)
        kernel_size = np.random.randint(1, 10)

        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))

        my_output = conv1d_minimal(x, weights)

        torch_output = t.conv1d(
            x,
            weights,
            stride=1,
            padding=0,
        )
        allclose_atol(my_output, torch_output, 1e-4)


@report
def test_conv2d_minimal(conv2d_minimal, dtype, atol, n_tests=2):
    """Compare against torch.conv2d.

    Due to floating point rounding, they can be quite different in float32 but should be nearly identical in float64.
    """
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))

        x = t.randn((b, ci, h, w), dtype=dtype)
        weights = t.randn((co, ci, *kernel_size), dtype=dtype)
        my_output = conv2d_minimal(x, weights)
        torch_output = t.conv2d(x, weights)
        allclose_atol(my_output, torch_output, atol)


@report
def test_conv1d(my_conv, n_tests=10):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = np.random.randint(1, 5)
        padding = np.random.randint(0, 5)
        kernel_size = np.random.randint(1, 10)

        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))

        my_output = my_conv(x, weights, stride=stride, padding=padding)

        torch_output = t.conv1d(x, weights, stride=stride, padding=padding)
        allclose_atol(my_output, torch_output, 1e-4)


@report
def test_pad1d(pad1d):
    """Should work with one channel of width 4."""
    x = t.arange(4).float().view((1, 1, 4))
    actual = pad1d(x, 1, 3, -2.0)
    expected = t.tensor([[[-2.0, 0.0, 1.0, 2.0, 3.0, -2.0, -2.0, -2.0]]])
    assert_all_equal(actual, expected)


@report
def test_pad1d_multi_channel(pad1d):
    """Should work with two channels of width 2."""
    x = t.arange(4).float().view((1, 2, 2))
    actual = pad1d(x, 0, 2, -3.0)
    expected = t.tensor([[[0.0, 1.0, -3.0, -3.0], [2.0, 3.0, -3.0, -3.0]]])
    assert_all_equal(actual, expected)


@report
def test_pad2d(pad):
    """Should work with one channel of 2x2."""
    x = t.arange(4).float().view((1, 1, 2, 2))
    expected = t.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [2.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ]
    )
    actual = pad(x, 0, 1, 2, 3, 0.0)
    assert_all_equal(actual, expected)


@report
def test_pad2d_multi_channel(pad):
    """Should work with two channels of 2x1."""
    x = t.arange(4).float().view((1, 2, 2, 1))
    expected = t.tensor([[[[-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]], [[-1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]]]])
    actual = pad(x, 1, 0, 0, 1, -1.0)
    assert_all_equal(actual, expected)


@report
def test_conv2d(my_conv, dtype, atol, n_tests=2):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)

        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))

        x = t.randn((b, ci, h, w), dtype=dtype)
        weights = t.randn((co, ci, *kernel_size), dtype=dtype)
        my_output = my_conv(x, weights, stride=stride, padding=padding)
        torch_output = t.conv2d(x, weights, stride=stride, padding=padding)
        allclose_atol(my_output, torch_output, atol)


@report
def test_maxpool2d(my_maxpool2d, n_tests=20):
    import numpy as np

    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)

        none_stride = bool(np.random.randint(2))
        if none_stride:
            stride = None
        else:
            stride = tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)

        x = t.randn((b, ci, h, w))

        my_output = my_maxpool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        torch_output = t.max_pool2d(
            x,
            kernel_size,
            stride=stride,  # type: ignore (None actually is allowed)
            padding=padding,
        )
        allclose_atol(my_output, torch_output, 1e-4)


@report
def test_maxpool2d_module(MaxPool2d):
    """Should take the max over a 4x4 grid of the numbers [0..16) correctly."""
    m = MaxPool2d((2, 2), stride=2, padding=0)
    x = t.arange(16).reshape((1, 1, 4, 4)).float()
    expected = t.tensor([[5.0, 7.0], [13.0, 15.0]])
    assert_all_equal(m(x), expected)


@report
def test_conv2d_module(Conv2d):
    """Your weight should be called 'weight' and have an appropriate number of elements.

    TBD: anything else here? We already tested the functional form sufficiently beforehand.
    """
    m = Conv2d(4, 5, (3, 3))
    assert isinstance(m.weight, t.nn.parameter.Parameter), "Weight should be registered a parameter!"
    assert m.weight.nelement() == 4 * 5 * 3 * 3


@report
def test_batchnorm2d_module(BatchNorm2d):
    """The public API of the module should be the same as the real PyTorch version."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.num_features == num_features
    assert isinstance(bn.weight, t.nn.parameter.Parameter)
    assert isinstance(bn.bias, t.nn.parameter.Parameter)
    assert isinstance(bn.running_mean, t.Tensor)
    assert isinstance(bn.running_var, t.Tensor)
    assert isinstance(bn.num_batches_tracked, t.Tensor)


@report
def test_batchnorm2d_forward(BatchNorm2d):
    """For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps)."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.training
    x = t.randn((100, num_features, 3, 4))
    out = bn(x)
    assert x.shape == out.shape
    allclose_atol(out.mean(dim=(0, 2, 3)), t.zeros(num_features), 1e-6)
    allclose(out.std(dim=(0, 2, 3)), t.ones(num_features), rtol=1e-2)


@report
def test_batchnorm2d_running_mean(BatchNorm2d):
    """Over repeated forward calls with the same data in train mode, the running mean should converge to the actual mean."""
    bn = BatchNorm2d(3, momentum=0.6)
    assert bn.training
    x = t.arange(12).float().view((2, 3, 2, 1))
    mean = t.tensor([3.5000, 5.5000, 7.5000])
    num_batches = 20
    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - (((1 - bn.momentum) ** (i + 1)))) * mean
        allclose(bn.running_mean, expected_mean)
    assert bn.num_batches_tracked.item() == num_batches

    # Large enough momentum and num_batches -> running_mean should be very close to actual mean
    bn.eval()
    actual_eval_mean = bn(x).mean((0, 2, 3))
    allclose(actual_eval_mean, t.zeros(3))


# TBD lowpri: test running_var as well


@report
def test_flatten(Flatten):
    x = t.arange(24).reshape((2, 3, 4))
    assert Flatten(start_dim=0)(x).shape == (24,)
    assert Flatten(start_dim=1)(x).shape == (2, 12)
    assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4)
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (6, 4)


@report
def test_flatten_is_view(Flatten):
    """Normally, Flatten should be able to return a view meaning changes affect the original input."""
    x = t.arange(24).reshape((2, 3, 4))
    view = Flatten()(x)
    view[0][0] = 99
    assert x[0, 0, 0] == 99
    # TBD lowpri: test a case where flatten can't be a view


@report
def test_linear_forward(Linear):
    """Your Linear should produce identical results to torch.nn given identical parameters."""
    x = t.rand((10, 512))
    yours = Linear(512, 64)

    assert yours.weight.shape == (64, 512)
    assert yours.bias.shape == (64,)

    official = t.nn.Linear(512, 64)
    yours.weight = official.weight
    yours.bias = official.bias
    actual = yours(x)
    expected = official(x)
    allclose(actual, expected)


@report
def test_linear_parameters(Linear):
    m = Linear(2, 3)
    params = dict(m.named_parameters())
    assert len(params) == 2, f"Your model has {len(params)} recognized Parameters"
    assert list(params.keys()) == [
        "weight",
        "bias",
    ], "For compatibility with PyTorch, your fields should be named weight and bias."


@report
def test_linear_no_bias(Linear):
    m = Linear(3, 4, bias=False)
    assert m.bias is None, "Bias should be None when not enabled."
    assert len(list(m.parameters())) == 1


@report
def test_sequential(Sequential):
    from torch.nn import Linear, ReLU

    modules = [Linear(1, 2), ReLU(), Linear(2, 1)]
    s = Sequential(*modules)

    assert list(s.modules()) == [s, *modules], "The sequential and its submodules should be registered Modules."
    assert len(list(s.parameters())) == 4, "Submodules's parameters should be registered."


@report
def test_sequential_forward(Sequential):
    from torch.nn import Linear, ReLU

    modules = [Linear(1, 2), ReLU(), Linear(2, 1)]
    x = t.tensor([5.0])
    s = Sequential(*modules)
    actual_out = s(x)
    expected_out = modules[-1](modules[-2](modules[-3](x)))
    allclose(actual_out, expected_out)


@report
def test_same_predictions(your_model_predictions: list[int]):
    assert your_model_predictions == [367, 207, 103, 604, 865, 562, 628, 39, 980, 447]
