import cudnn
import pytest
from looseversion import LooseVersion


def test_conv_relu(jparams, testgraph):
    X = testgraph.tensor(dim=jparams["in_dim"], layout="NHWC")
    W = testgraph.tensor(dim=jparams["filter_dim"], layout="NHWC")

    conv_out = testgraph.conv_fprop(
        name="conv",
        image=X,
        weight=W,
        padding=jparams["padding"],
        stride=jparams["stride"],
        dilation=jparams["dilation"],
    )
    Y = testgraph.relu(input=conv_out)


# This test broke in merge commit 72b88c88f125382694a784bb4ce2535f62d09903
# def test_conv(jparams, testgraph):
#     X = testgraph.tensor(dim=jparams["in_dim"], data_type = cudnn.data_type.FLOAT, layout = "NHWC")
#     W = testgraph.tensor(dim=jparams["filter_dim"], data_type = cudnn.data_type.FLOAT, layout = "NHWC")

#     conv_out = testgraph.conv_fprop(name = "conv", image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])


def test_conv_bias_relu(jparams, testgraph):
    X = testgraph.tensor(dim=jparams["in_dim"], layout="NHWC")
    W = testgraph.tensor(dim=jparams["filter_dim"], layout="NHWC")
    bias = testgraph.tensor(dim=jparams["bias_dim"], layout="NHWC")

    conv_out = testgraph.conv_fprop(
        name="conv",
        image=X,
        weight=W,
        padding=jparams["padding"],
        stride=jparams["stride"],
        dilation=jparams["dilation"],
    )
    bias_out = testgraph.bias(name="bias", input=conv_out, bias=bias)
    relu_output = testgraph.relu(input=bias_out)


def test_conv_relu_bias_relu(jparams, testgraph):
    X = testgraph.tensor(dim=jparams["in_dim"], layout="NHWC")
    W = testgraph.tensor(dim=jparams["filter_dim"], layout="NHWC")
    bias = testgraph.tensor(dim=jparams["bias_dim"], layout="NHWC")

    conv_out = testgraph.conv_fprop(
        name="conv",
        image=X,
        weight=W,
        padding=jparams["padding"],
        stride=jparams["stride"],
        dilation=jparams["dilation"],
    )
    relu_1 = testgraph.relu(input=conv_out)
    bias_out = testgraph.bias(name="bias", input=relu_1, bias=bias)
    relu_output = testgraph.relu(input=bias_out)


def test_dgrad_add(jparams, testgraph):
    testgraph.set_compute_data_type(cudnn.data_type.FLOAT)
    testgraph.set_io_data_type(cudnn.data_type.FLOAT)
    wTensor = testgraph.tensor(dim=jparams["filter_dim"], layout="NHWC", data_type=cudnn.data_type.FLOAT)
    dyTensor = testgraph.tensor(dim=jparams["conv_out_dim"], layout="NHWC", data_type=cudnn.data_type.FLOAT)
    bTensor = testgraph.tensor(dim=jparams["dx_dim"], layout="NHWC", data_type=cudnn.data_type.FLOAT)

    dxTensor = testgraph.conv_dgrad(
        name="dgrad",
        loss=dyTensor,
        filter=wTensor,
        padding=jparams["padding"],
        stride=jparams["stride"],
        dilation=jparams["dilation"],
    )
    dxTensor.set_dim(jparams["dx_dim"])

    afterAdd = testgraph.add(name="add", a=dxTensor, b=bTensor)


def test_dgrad(jparams, testgraph):
    testgraph.set_compute_data_type(cudnn.data_type.FLOAT)
    testgraph.set_io_data_type(cudnn.data_type.FLOAT)
    wTensor = testgraph.tensor(dim=jparams["filter_dim"], layout="NHWC", data_type=cudnn.data_type.FLOAT)
    dyTensor = testgraph.tensor(dim=jparams["conv_out_dim"], layout="NHWC", data_type=cudnn.data_type.FLOAT)

    dxTensor = testgraph.conv_dgrad(
        name="dgrad",
        loss=dyTensor,
        filter=wTensor,
        padding=jparams["padding"],
        stride=jparams["stride"],
        dilation=jparams["dilation"],
    )
    dxTensor.set_dim(jparams["dx_dim"])


def test_batchnorm(jparams, testgraph):

    if (not "backend_version" in dir(cudnn)) or LooseVersion(cudnn.backend_version_string()) < "8.7":
        pytest.skip("BN not supported below cudnn 8.7")

    testgraph.set_io_data_type(cudnn.data_type.FLOAT)
    testgraph.set_heuristics([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    N, C, H, W = jparams["in_dim"]
    X = testgraph.tensor(dim=jparams["in_dim"], data_type=cudnn.data_type.HALF, layout="NHWC")
    scale = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    bias = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    in_running_mean = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    in_running_var = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)

    epsilon = testgraph.tensor_cpu_constant(1e-03, dim=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
    momentum = testgraph.tensor_cpu_constant(0.1, dim=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

    Y, saved_mean, saved_inv_var, out_running_mean, out_running_var = testgraph.batchnorm(
        name="BN",
        input=X,
        scale=scale,
        bias=bias,
        in_running_mean=in_running_mean,
        in_running_var=in_running_var,
        epsilon=epsilon,
        momentum=momentum,
    )

    # TODO: set_data_type. Also allow chaining by returning the tensor
    Y.set_data_type(cudnn.data_type.HALF)


def test_conv_batchnorm(jparams, testgraph):

    pytest.skip("BN conv fusion not yet supported")

    testgraph.set_io_data_type(cudnn.data_type.FLOAT)
    testgraph.set_heuristics([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])

    X = testgraph.tensor(dim=jparams["in_dim"], layout="NHWC")
    W = testgraph.tensor(dim=jparams["filter_dim"], layout="NHWC")

    conv_out = testgraph.conv_fprop(
        name="conv",
        image=X,
        weight=W,
        padding=jparams["padding"],
        stride=jparams["stride"],
        dilation=jparams["dilation"],
    )

    C = jparams["conv_out"][1]
    # X = testgraph.tensor(dim=jparams["in_dim"], data_type=cudnn.data_type.HALF, layout = "NHWC")
    scale = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    bias = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    in_running_mean = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    in_running_var = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)

    epsilon = testgraph.tensor_cpu_constant(1e-03, dim=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)
    momentum = testgraph.tensor_cpu_constant(0.1, dim=[1, 1, 1, 1], data_type=cudnn.data_type.FLOAT)

    Y, saved_mean, saved_inv_var, out_running_mean, out_running_var = testgraph.batchnorm(
        name="BN",
        input=conv_out,
        scale=scale,
        bias=bias,
        in_running_mean=in_running_mean,
        in_running_var=in_running_var,
        epsilon=epsilon,
        momentum=momentum,
    )

    # TODO: set_data_type. Also allow chaining by returning the tensor
    Y.set_data_type(cudnn.data_type.HALF)


def test_gemm(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(name="image", dim=[B, M, K], data_type=cudnn.data_type.HALF)
    weight = testgraph.tensor(name="weight", dim=[B, K, N], data_type=cudnn.data_type.HALF)

    gemm_output = testgraph.matmul(name="mb_matmul", A=image, B=weight, compute_data_type=cudnn.data_type.FLOAT)
    # gemm_output.set_stride([M*N, N, 1])


def test_gemm_relu(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(name="image", dim=[B, M, K], data_type=cudnn.data_type.HALF)
    weight = testgraph.tensor(name="weight", dim=[B, K, N], data_type=cudnn.data_type.HALF)

    gemm_output = testgraph.matmul(name="mb_matmul", A=image, B=weight, compute_data_type=cudnn.data_type.FLOAT)
    # Make intermediate tensor output row-major:
    gemm_output.set_stride([M * N, N, 1])
    relu_output = testgraph.relu(input=gemm_output)


def test_gemm_bias_relu(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(name="image", dim=[B, M, K], data_type=cudnn.data_type.HALF)
    weight = testgraph.tensor(name="weight", dim=[B, K, N], data_type=cudnn.data_type.HALF)
    bias = testgraph.tensor(name="weight", dim=[B, M, N], data_type=cudnn.data_type.HALF)

    gemm_output = testgraph.matmul(name="mb_matmul", A=image, B=weight, compute_data_type=cudnn.data_type.FLOAT)
    # Make intermediate tensor output row-major:
    gemm_output.set_stride([M * N, N, 1])

    bias_out = testgraph.bias(name="bias", input=gemm_output, bias=bias)
    relu_output = testgraph.relu(input=bias_out)


def test_reduction(jparams, testgraph):
    N, C, H, W = jparams["in_dim"]

    X = testgraph.tensor(dim=[N, C, H, W], layout="NHWC")

    Y = testgraph.reduction(input=X, mode=cudnn.reduction_mode.ADD)
    Y.set_dim([N, 1, H, W])
    Y.set_data_type(cudnn.data_type.FLOAT)


def test_conv_reduction(jparams, testgraph):
    N, C, H, W = jparams["in_dim"]
    K, C2, R, S = jparams["filter_dim"]
    assert C == C2

    padding = stride = dilation = [1, 1]
    X = testgraph.tensor(dim=[N, C, H, W], layout="NHWC")
    Weight = testgraph.tensor(dim=[K, C, R, S], layout="NHWC")
    Y0 = testgraph.conv_fprop(image=X, weight=Weight, padding=padding, stride=stride, dilation=dilation)

    Y = testgraph.reduction(input=Y0, mode=cudnn.reduction_mode.ADD)
    Y.set_dim([N, 1, H, W])
    Y.set_data_type(cudnn.data_type.FLOAT)
