import warnings

import numpy as np

from utils import report


@report
def test_log_back(log_back):
    a = np.array([1, np.e, np.e**np.e])
    b = np.log(a)
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = log_back(grad_out, b, a)
    expected = [2.0, 2.0 / np.e, 2.0 / (np.e**np.e)]
    assert np.allclose(actual, expected)


@report
def test_unbroadcast(unbroadcast):
    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 20.0).all(), "Each element in the small array appeared 20 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 1, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 5.0).all(), "Each element in the small array appeared 5 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 4.0).all(), "Each element in the small array appeared 4 times in the large array."


@report
def test_multiply_back(multiply_back0, multiply_back1):
    a = np.array([1, 2, 3])
    b = np.array([2])
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back0(grad_out, c, a, b)
    expected = [4.0, 4.0, 4.0]
    assert np.allclose(actual, expected)
    actual = multiply_back1(grad_out, c, a, b)
    expected = [12.0]
    assert np.allclose(actual, expected)


@report
def test_multiply_back_float(multiply_back0, multiply_back1):
    a = np.array([1, 2, 3])
    b = 2
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back0(grad_out, c, a, b)
    expected = [4.0, 4.0, 4.0]
    assert np.allclose(actual, expected)

    a = np.array([1, 2, 3])
    b = 2
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back1(grad_out, c, b, a)
    expected = [4.0, 4.0, 4.0]
    assert np.allclose(actual, expected)


@report
def test_forward_and_back(forward_and_back):
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 1])
    c = np.array([10])
    dg_da, dg_db, dg_dc = forward_and_back(a, b, c)
    expected_dg_da = np.array([1, 1 / 2, 1 / 3])
    expected_dg_db = np.array([1 / 2, 1 / 3, 1])
    expected_dg_dc = np.array([0.13028834])
    assert np.allclose(dg_da, expected_dg_da)
    assert np.allclose(dg_db, expected_dg_db)
    assert np.allclose(dg_dc, expected_dg_dc)


@report
def test_back_func_lookup(BackwardFuncLookup):
    backward_funcs = BackwardFuncLookup()
    backward_funcs.add_back_func(np.log, 0, np.exp)
    assert backward_funcs.get_back_func(np.log, 0) == np.exp
    backward_funcs.add_back_func(np.multiply, 0, np.divide)
    assert backward_funcs.get_back_func(np.multiply, 0) == np.divide
    backward_funcs.add_back_func(np.multiply, 1, np.add)
    assert backward_funcs.get_back_func(np.multiply, 1) == np.add


@report
def test_log(Tensor, log_forward):
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = log_forward(a)
    assert np.allclose(b.array, [1, np.e])
    assert b.requires_grad == True, "Should require grad because input required grad."
    assert b.is_leaf == False
    assert b.recipe is not None
    assert len(b.recipe.parents) == 1 and b.recipe.parents[0] is a
    assert len(b.recipe.args) == 1 and b.recipe.args[0] is a.array
    assert b.recipe.kwargs == {}
    assert b.recipe.func is np.log
    c = log_forward(b)
    assert np.allclose(c.array, [0, 1])


@report
def test_log_no_grad(Tensor, log_forward):
    d = Tensor([1, np.e])
    e = log_forward(d)
    assert e.requires_grad == False, "Should not require grad because input did not."
    assert e.recipe is None
    assert np.allclose(e.array, [0, 1])


@report
def test_multiply(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = multiply(a, b)
    assert c.requires_grad == True, "Should require grad because input required grad."
    assert c.is_leaf == False
    assert c.recipe is not None
    assert len(c.recipe.parents) == 2 and c.recipe.parents[0] is a and c.recipe.parents[1] is b
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is a.array and c.recipe.args[1] is b.array
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 10, 20, 30]])
    np.allclose(c.array, expected)


@report
def test_multiply_no_grad(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=False)
    b = Tensor([[0], [1], [10]], requires_grad=False)
    c = multiply(a, b)
    assert c.requires_grad == False, "Should not require grad because input did not require grad."
    assert c.recipe is None
    expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 10, 20, 30]])
    np.allclose(c.array, expected)


@report
def test_multiply_float(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = 3
    c = multiply(a, b)
    assert c.requires_grad == True
    assert c.recipe is not None
    assert len(c.recipe.parents) == 1 and c.recipe.parents[0] is a
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is a.array and c.recipe.args[1] is b
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([0, 3, 6, 9])
    np.allclose(c.array, expected)

    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = 3
    c = multiply(b, a)
    assert c.requires_grad == True
    assert c.recipe is not None
    assert len(c.recipe.parents) == 1 and c.recipe.parents[1] is a
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is b and c.recipe.args[1] is a.array
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([0, 3, 6, 9])
    np.allclose(c.array, expected)


class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> list[Node]:
    return node.children


@report
def test_topological_sort_linked_list(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)

    expected = [z, y, x]
    for e, a in zip(expected, topological_sort(x, get_children)):
        assert e is a


@report
def test_topological_sort_branching(topological_sort):
    z = Node()
    y = Node()
    x = Node(y, z)
    w = Node(x)

    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw" or out == "yzxw"


@report
def test_topological_sort_rejoining(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    w = Node(z, x)

    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw"


@report
def test_topological_sort_cyclic(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    z.children = [x]

    try:
        topological_sort(x, get_children)
    except:
        assert True
    else:
        assert False


@report
def test_backprop(Tensor):
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = a.log()
    c = b.log()
    c.backward(end_grad=np.array([1.0, 1.0]))
    assert c.grad is None
    assert b.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, 1 / b.array / a.array)


@report
def test_backprop_branching(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert np.allclose(b.grad.array, a.array)


@report
def test_backprop_requires_grad_false(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=False)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert b.grad is None


@report
def test_backprop_float_arg(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = 2
    c = a * b
    d = 2
    e = d * c
    e.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert e.grad is None
    assert c.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([4.0, 4.0, 4.0]))


@report
def test_negative_back(Tensor):
    a = Tensor([-1, 0, 1], requires_grad=True)
    b = -a
    c = -b
    c.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    assert a.grad is not None
    assert np.allclose(a.grad.array, [1, 1, 1])


@report
def test_exp_back(Tensor):
    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    b.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    assert a.grad is not None
    assert np.allclose(a.grad.array, 1 / np.e, 0, np.e)

    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    c = b.exp()
    c.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))

    def d(x):
        return (np.e**x) * (np.e ** (np.e**x))

    assert a.grad is not None
    assert np.allclose(a.grad.array, *[d(x) for x in a.array])


@report
def test_reshape_back(Tensor):
    a = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
    b = a.reshape((3, 2))
    b.backward(end_grad=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))
    assert a.grad is not None and np.allclose(a.grad.array, np.ones(6))


@report
def test_permute_back(Tensor):
    a = Tensor(np.arange(24).reshape((2, 3, 4)), requires_grad=True)
    out = a.permute((2, 0, 1))
    out.backward(np.arange(24).reshape((4, 2, 3)))

    assert a.grad is not None
    assert np.allclose(
        a.grad.array,
        np.array(
            [
                [[0.0, 6.0, 12.0, 18.0], [1.0, 7.0, 13.0, 19.0], [2.0, 8.0, 14.0, 20.0]],
                [[3.0, 9.0, 15.0, 21.0], [4.0, 10.0, 16.0, 22.0], [5.0, 11.0, 17.0, 23.0]],
            ]
        ),
    )


@report
def test_expand(Tensor):
    a = Tensor(np.ones((2, 1, 3)), requires_grad=True)
    b = a.expand((5, 1, 2, 4, 3))
    b.backward(np.full_like(b.array, 10.0))
    assert a.grad is not None and a.grad.shape == a.array.shape
    assert (a.grad.array == 20 * 10.0).all()


@report
def test_expand_negative_length(Tensor):
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    b = a.expand((3, 2, -1))
    assert b.shape == (3, 2, 5)
    b.backward(end_grad=np.ones(b.shape))
    assert a.grad is not None and a.grad.shape == a.array.shape
    assert (a.grad.array == 6).all()


@report
def test_sum_keepdim_false(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum(0)
    c = b.sum(0)
    c.backward(np.array(2))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 2).all()


@report
def test_sum_keepdim_true(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum(1, keepdim=True)
    c = a.sum(0, keepdim=True)
    assert np.allclose(c.array, np.array([[5.0, 7.0, 9.0, 11.0, 13.0]]))
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 1).all()


@report
def test_sum_dim_none(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum()
    b.backward(np.array(4))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 4).all()


@report
def test_getitem_int(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[1]
    c = b.sum(0)
    c.backward(np.array(10.0))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([[0, 0, 0], [10, 10, 10]]))


@report
def test_getitem_tuple(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[(1, 2)]
    b.backward(np.array(10.0))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([[0, 0, 0], [0, 0, 10]]))


@report
def test_getitem_integer_array(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 2, 0])
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))


@report
def test_getitem_integer_tensor(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = Tensor(np.array([0, 1, 0, 1, 0])), Tensor(np.array([0, 0, 1, 2, 0]))
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))


@report
def test_add_broadcasted(Tensor):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a + b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == 4).all()


@report
def test_subtract_broadcasted(Tensor):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a - b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == -4).all()


@report
def test_truedivide_broadcasted(Tensor):
    a = Tensor([0, 6, 12, 18], requires_grad=True)
    b = Tensor([[1], [2], [3]], requires_grad=True)
    c = a / b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == (1 + 1 / 2 + 1 / 3)).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert np.equal(b.grad.array, np.array([[-36.0], [-9.0], [-4.0]])).all()


@report
def test_maximum(Tensor):
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([-1, 1, 3], requires_grad=True)
    out = a.maximum(b)
    assert np.allclose(out.array, [0, 1, 3])
    out.backward(end_grad=np.ones(out.shape))

    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad.array, [1, 0.5, 0])
    assert np.allclose(b.grad.array, [0, 0.5, 1])


@report
def test_maximum_broadcasted(Tensor):
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([[-1], [1], [3]], requires_grad=True)
    out = a.maximum(b)
    assert np.allclose(out.array, np.array([[0, 1, 2], [1, 1, 2], [3, 3, 3]]))
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([1.0, 1.5, 2.0]))
    assert b.grad is not None and np.allclose(b.grad.array, np.array([[0.0], [1.5], [3.0]]))


@report
def test_relu(Tensor):
    a = Tensor([-1, 0, 1], requires_grad=True)
    out = a.relu()
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([0, 0.5, 1.0]))


@report
def test_matmul2d(Tensor):
    a = Tensor(np.arange(-3, 3).reshape((2, 3)), requires_grad=True)
    b = Tensor(np.arange(-4, 5).reshape((3, 3)), requires_grad=True)
    out = a @ b
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad.array, np.array([[-9, 0, 9], [-9, 0, 9]]))
    assert np.allclose(b.grad.array, np.array([[-3, -3, -3], [-1, -1, -1], [1, 1, 1]]))


@report
def test_cross_entropy(Tensor, cross_entropy):
    logits = Tensor([[float("-inf"), float("-inf"), 0], [1 / 3, 1 / 3, 1 / 3], [float("-inf"), 0, 0]])
    true_labels = Tensor([2, 0, 0])
    expected = Tensor([0.0, np.log(3), float("inf")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = cross_entropy(logits, true_labels)
    assert np.allclose(actual.array, expected.array)


@report
def test_no_grad(Tensor, NoGrad):
    a = Tensor([1], requires_grad=True)
    with NoGrad():
        b = a + a
    c = a + a
    assert b.requires_grad == False
    assert b.recipe is None
    assert c.requires_grad
    assert c.recipe is not None


@report
def test_no_grad_nested(Tensor, NoGrad):
    a = Tensor([1], requires_grad=True)
    with NoGrad():
        with NoGrad():
            with NoGrad():
                b = a + a
    assert b.requires_grad == False
    assert b.recipe is None
