import torch as t
from torch.utils.data import DataLoader, TensorDataset
from utils import report, allclose, allclose_scalar, assert_all_equal, allclose_atol
import torch
from sklearn.datasets import make_moons
from torch.nn import functional as F


@report
def test_mlp(MLP):
    import w1d4_part1_solution

    x = t.randn(128, 2)
    t.manual_seed(534)
    mlp = MLP(2, 32, 2)

    params = list(mlp.parameters())
    if not params:
        raise AssertionError(
            "Your network doesn't return anything when mlp.parameters() is called. If you used a list for your layers, you need to use nn.ModuleList or nn.Sequential instead."
        )

    t.manual_seed(534)
    _mlp = w1d4_part1_solution.ImageMemorizer(2, 32, 2)

    allclose(mlp(x), _mlp(x))


@report
def test_tensor_dataset(TensorDataset):
    tensors = [t.rand((10, 20)), t.rand((10, 5)), t.arange(10)]
    dataset = TensorDataset(*tensors)
    assert len(dataset) == 10
    for index in [0, slice(0, 5, 1), slice(1, 5, 2)]:
        print("Testing with index:", index)
        expected = tuple(tensor[index] for tensor in tensors)
        actual = dataset[index]
        for e, a in zip(expected, actual):
            assert_all_equal(e, a)


@report
def test_train(train_one_epoch):
    import w1d4_part1_solution

    t.manual_seed(928)
    lr = 0.1
    momentum = 0.5
    X = t.rand(512, 2)
    Y = t.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    t.manual_seed(600)
    expected_model = w1d4_part1_solution.ImageMemorizer(2, 32, 3)
    expected_train_loss = w1d4_part1_solution.train_one_epoch(expected_model, dl)

    t.manual_seed(600)
    actual_model = w1d4_part1_solution.ImageMemorizer(2, 32, 3)
    actual_train_loss = train_one_epoch(actual_model, dl)

    allclose_scalar(actual_train_loss, expected_train_loss)

    x = t.randn(128, 2)
    allclose(actual_model(x.to(w1d4_part1_solution.device)), expected_model(x.to(w1d4_part1_solution.device)))


@report
def test_evaluate(evaluate):
    import w1d4_part1_solution

    # TBD: loss on random labels? On predict all zero? Not sure how good it is to do this.
    t.manual_seed(928)
    X = t.rand(512, 2)
    Y = t.rand(512, 3)
    dl = DataLoader(TensorDataset(X, Y), batch_size=128)

    model = w1d4_part1_solution.ImageMemorizer(2, 32, 3)
    w1d4_part1_solution.train_one_epoch(model, dl)
    _loss = w1d4_part1_solution.evaluate(model, dl)
    loss = evaluate(model, dl)
    allclose_scalar(_loss, loss)


def _get_moon_data(unsqueeze_y=False):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    if unsqueeze_y:  # better when the training regimen uses l1 loss, rather than x-ent
        y = y.unsqueeze(-1)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)


def _train_with_opt(model, opt):
    dl = _get_moon_data()
    for i, (X, y) in enumerate(dl):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()


@report
def test_sgd(SGD):
    import w1d4_part1_solution

    test_cases = [
        dict(lr=0.1, momentum=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.7, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, weight_decay=0.05),
        dict(lr=0.2, momentum=0.8, weight_decay=0.05),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = w1d4_part1_solution.ImageMemorizer(2, 32, 2)
        opt = torch.optim.SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = w1d4_part1_solution.ImageMemorizer(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, torch.Tensor)
        assert isinstance(w0_submitted, torch.Tensor)
        allclose_atol(w0_correct, w0_submitted, atol=1e-5)


@report
def test_rmsprop(RMSprop):
    import w1d4_part1_solution

    test_cases = [
        dict(lr=0.1, alpha=0.9, eps=0.001, weight_decay=0.0, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.5),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = w1d4_part1_solution.ImageMemorizer(2, 32, 2)
        opt = torch.optim.RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = w1d4_part1_solution.ImageMemorizer(2, 32, 2)
        opt = RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, torch.Tensor)
        assert isinstance(w0_submitted, torch.Tensor)
        allclose_atol(w0_correct, w0_submitted, atol=1e-5)


@report
def test_adam(Adam):
    import w1d4_part1_solution

    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = w1d4_part1_solution.ImageMemorizer(2, 32, 2)
        opt = torch.optim.Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = w1d4_part1_solution.ImageMemorizer(2, 32, 2)
        opt = Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, torch.Tensor)
        assert isinstance(w0_submitted, torch.Tensor)
        allclose_atol(w0_correct, w0_submitted, atol=1e-5)
