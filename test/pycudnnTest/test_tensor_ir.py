from typing import Optional
import test_graph as tg
import torch
import utils
from utils import StatusCode
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple
import inspect
from functools import lru_cache
import math
import random
import string
from tensor_ir_utils import CompilerWithKernelCacheSingleton

from nv_tensor_ir._mlir import ir
from nv_tensor_ir._mlir.dialects import nv_tensor_ir

import nv_tensor_ir._mlir.extras.types as T
from data_types import DataType, convert_datatype

import sys

# Redirect stderr to stdout to preserve chronological order of output
sys.stderr = sys.stdout


_DEFAULT_CUBIN_CHIP_FALLBACK = "sm_100a"
_SUPPORTED_CUBIN_CHIP = {100: _DEFAULT_CUBIN_CHIP_FALLBACK, 107: "sm_107a"}


def _get_cubin_chip_for_compute_capability(compute_capability):
    return _SUPPORTED_CUBIN_CHIP.get(compute_capability, _DEFAULT_CUBIN_CHIP_FALLBACK)


@lru_cache(maxsize=1)
def get_default_cubin_chip():
    """Detect cubin_chip from the current GPU's compute capability."""
    try:
        cask_context = nv_tensor_ir.create_cask_context()
        cask_context.initialize_cuda_device()
        cc = cask_context.get_compute_capability()
        return _get_cubin_chip_for_compute_capability(cc)
    except Exception:
        return _DEFAULT_CUBIN_CHIP_FALLBACK


def get_element_bits(data_type):
    if data_type in [DataType.DOUBLE, DataType.INT64]:
        return 64
    elif data_type in [DataType.FLOAT, DataType.INT32]:
        return 32
    elif data_type in [
        DataType.FP8_E4M3,
        DataType.FP8_E8M0,
        DataType.FP8_E5M2,
        DataType.INT8,
    ]:
        return 8
    elif data_type in [DataType.HALF, DataType.BFLOAT16]:
        return 16
    else:
        raise ValueError(f"Unsupported data type: {data_type}")


def is_float_dtype(data_type):
    return data_type in [
        DataType.FLOAT,
        DataType.DOUBLE,
        DataType.HALF,
        DataType.BFLOAT16,
    ]


# Maximum shared memory per SM (bytes) by architecture, used to pre-filter
# CudaTile tile configs before invoking ptxas.  Values are the hard limits
# reported by ptxas ("uses too much shared data … max").
_SM_MAX_SMEM_BYTES = {
    "sm_80": 163_840,  # A100  160 KB
    "sm_86": 99_328,  # RTX 30xx  97 KB
    "sm_87": 99_328,
    "sm_89": 99_328,  # RTX 40xx  97 KB
    "sm_90": 231_424,  # H100  226 KB  (0x38c00)
    "sm_90a": 231_424,
    "sm_100": 232_448,  # B100  0x38c00 (from ptxas)
    "sm_100a": 232_448,  # 0x38c00 (from ptxas)
}
# Fall-back limit used for unknown chips.
_SM_MAX_SMEM_BYTES_DEFAULT = 231_424


# Operations that have no lowering pattern in TensorToCudaTile.cpp and will
# therefore produce kErrorCompilation when compiled with the CudaTile backend.
# Tests using these ops are waived for CudaTile until lowering support is added.
_CUDA_TILE_UNSUPPORTED_OPS = frozenset(
    [
        # ActivationForwardNode ops (no EluFwd/SwishFwd/SoftplusFwd conversion)
        "elu",
        "swish",
        "softplus",
        # ActivationBackwardNode ops
        "elu_backward",
        "swish_backward",
        "softplus_backward",
        # BinaryOperationNode backward ops
        "relu_backward",
        "tanh_backward",
        "sigmoid_backward",
        "gelu_backward",
        "gelu_approx_tanh_backward",
        # BinaryOperationNode ops without CudaTile lowering
        "atan2",
        # ScaledMatmulOp: no TensorToCudaTile lowering pattern yet; uses Collective backend
        "scaled_matmul",
    ]
)

# Operations whose CudaTile lowering only supports f32/f64 operands.
# These are waived when the operation's output type is f16.
# Note: the test-graph op_names here are "erf" and "gelu" (CUDNN_POINTWISE_GELU_FWD).
_CUDA_TILE_F32_ONLY_OPS = frozenset(
    [
        "erf",
        "gelu",
    ]
)


def is_integer_dtype(data_type):
    return data_type in [DataType.INT8, DataType.INT32, DataType.INT64]


def parse_int_list(value):
    """Parse comma-separated string into list of integers.

    Args:
        value: Comma-separated string (e.g., "1,2,3")

    Returns:
        List of integers

    Raises:
        ValueError: If parsing fails
    """
    try:
        return list(map(int, value.split(",")))
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Failed to parse integer list from '{value}': {e}")


def get_optional_arg(args, attr_name, parser=None, default=None):
    """Safely retrieve and optionally parse an argument attribute.

    Args:
        args: Arguments object
        attr_name: Name of the attribute to retrieve
        parser: Optional function to parse the value (e.g., int, bool, parse_int_list)
        default: Default value if attribute is None or doesn't exist

    Returns:
        Parsed value or default
    """
    value = getattr(args, attr_name, None)
    if value is None:
        return default
    return parser(value) if parser else value


def _get_arg_from_tensorir_args(tensorir_args, attr_name, converter=None, default=None):
    value = getattr(tensorir_args, attr_name, None)
    if value is None:
        return default
    return converter(value) if converter else value


def _parse_comma_separated_ints(value):
    return list(map(int, value.split(",")))


def _extract_config_properties(config):
    cga_tile_size = [config.cgaTileSize.m, config.cgaTileSize.n, config.cgaTileSize.k]
    mma_shape = [config.mmaShape.m, config.mmaShape.n, config.mmaShape.k]
    cluster_shape = [
        config.clusterShape.m,
        config.clusterShape.n,
        config.clusterShape.k,
    ]
    cta_count = config.mmaTileSize.m // config.ctaTileSize.m
    return cga_tile_size, mma_shape, cluster_shape, cta_count


def _create_config_list(
    cga_tile_size,
    mma_shape,
    cluster_shape,
    cta_count,
    stream_k,
    cubin_chip,
    matmul_element_bits,
):
    """
    Create a configuration list with all parameters.

    Returns:
        List containing all configuration parameters
    """
    return [
        cga_tile_size,
        mma_shape,
        cluster_shape,
        cta_count,
        stream_k,
        cubin_chip,
        matmul_element_bits,
    ]


class TensorIRArguments:
    """Helper class to encapsulate parsed TensorIR arguments."""

    def __init__(self, tensorir_args):
        """Parse and store all TensorIR arguments."""
        self.tile_size = _get_arg_from_tensorir_args(tensorir_args, "tile_size", _parse_comma_separated_ints)
        self.mma_shape = _get_arg_from_tensorir_args(tensorir_args, "mma_shape", _parse_comma_separated_ints)
        self.cluster_shape = _get_arg_from_tensorir_args(tensorir_args, "cluster_shape", _parse_comma_separated_ints)
        self.cta_count = _get_arg_from_tensorir_args(tensorir_args, "cta_count", int)
        self.stream_k = _get_arg_from_tensorir_args(tensorir_args, "stream_k", bool, False)
        self.cubin_chip = _get_arg_from_tensorir_args(tensorir_args, "cubin_chip", default=get_default_cubin_chip())

    def has_all_required(self):
        """Check if all required arguments are provided."""
        return all(
            x is not None
            for x in [
                self.mma_shape,
                self.tile_size,
                self.cluster_shape,
                self.cta_count,
            ]
        )

    def create_config(self, cga_tile_size, mma_shape, cluster_shape, cta_count, matmul_element_bits):
        """Create a configuration using the stored stream_k and cubin_chip."""
        return _create_config_list(
            cga_tile_size,
            mma_shape,
            cluster_shape,
            cta_count,
            self.stream_k,
            self.cubin_chip,
            matmul_element_bits,
        )

    def print_args(self):
        """Print the parsed arguments for debugging."""
        print(f"args_mma_shape {self.mma_shape}, args_cta_count {self.cta_count}, " f"args_stream_k {self.stream_k}, args_cubin_chip {self.cubin_chip}")


def generate_tensorir_compilation_configs(
    m,
    n,
    k,
    matmul_element_bits,
    tensorir_args,
    is8BitTransposeB,
    isUTCHMMA=False,
    isUTCIMMA=False,
    isBlockScaled=False,
    compiler_backend="Tile",
):
    """
    Generate multiple TensorIR compilation configurations.

    Returns a list of valid configurations based on problem size and constraints.
    """
    args = TensorIRArguments(tensorir_args)

    # If all required arguments are provided, return immediately
    if args.has_all_required():
        return [
            args.create_config(
                args.tile_size,
                args.mma_shape,
                args.cluster_shape,
                args.cta_count,
                matmul_element_bits,
            )
        ]

    args.print_args()

    # Generate all valid configurations
    print(
        "isUTCHMMA",
        isUTCHMMA,
        "isUTCIMMA",
        isUTCIMMA,
        "is8BitTransposeB",
        is8BitTransposeB,
    )
    print("m", m, "n", n, "k", k)
    cc = CompilerWithKernelCacheSingleton().get_compute_capability()
    configList = nv_tensor_ir.generateAllValidConfigurations_by_problem_size(
        nv_tensor_ir.MmaShape(m, n, k),
        matmul_element_bits,
        cc,
        isSparse=False,
        isWSMode=False,
        is8BitTransposeB=is8BitTransposeB,
        kPhase=4,
        isUTCHMMA=isUTCHMMA,
        isUTCIMMA=isUTCIMMA,
        blockScaling=isBlockScaled,
    )

    def should_keep_config(config):
        """Filter function to determine if a config should be kept."""
        cga_tile_size, mma_shape, cluster_shape, cta_count = _extract_config_properties(config)

        # Apply all filters
        if args.tile_size is not None and cga_tile_size != args.tile_size:
            return False
        if args.mma_shape is not None and mma_shape != args.mma_shape:
            return False
        if args.cta_count is not None and cta_count != args.cta_count:
            return False

        # For CudaTile, apply two pre-filters to avoid hard failures:
        if compiler_backend == "CudaTile":
            tile_m, tile_n = cga_tile_size[0], cga_tile_size[1]

            # 1. CudaTile requires all tile dimensions to be powers of two.
            #    Configs that violate this trigger an assertion abort in the
            #    CudaTile type verifier (not catchable in Python).
            def _is_pow2(v):
                return v > 0 and (v & (v - 1)) == 0

            if not _is_pow2(tile_m) or not _is_pow2(tile_n):
                return False

            # 2. Reject configs whose C-tile shared memory footprint exceeds
            #    the target SM limit.  The dominant smem term is the output
            #    (C) tile: tile_M * tile_N * bytes_per_element.  Configs that
            #    exceed the limit fail with ptxas "uses too much shared data".
            bytes_per_element = max(1, matmul_element_bits // 8)
            smem_estimate = tile_m * tile_n * bytes_per_element
            max_smem = _SM_MAX_SMEM_BYTES.get(args.cubin_chip, _SM_MAX_SMEM_BYTES_DEFAULT)
            if smem_estimate > max_smem:
                print(
                    f"#### Skipping tile_size={cga_tile_size[:2]} for CudaTile: "
                    f"estimated smem {smem_estimate} bytes exceeds {args.cubin_chip} limit {max_smem} bytes"
                )
                return False

        return True

    # Apply filter to configList
    filtered_configs = filter(should_keep_config, configList)

    configs = []
    seen = set()

    for config in filtered_configs:
        cga_tile_size, mma_shape, cluster_shape, cta_count = _extract_config_properties(config)

        # Create config and check for duplicates
        config_tuple = (
            tuple(cga_tile_size),
            tuple(mma_shape),
            tuple(cluster_shape),
            cta_count,
            args.stream_k,
            args.cubin_chip,
            matmul_element_bits,
        )
        if config_tuple not in seen:
            seen.add(config_tuple)
            configs.append(
                args.create_config(
                    cga_tile_size,
                    mma_shape,
                    cluster_shape,
                    cta_count,
                    matmul_element_bits,
                )
            )

    # Randomly shuffle the configs
    if tensorir_args.random_sweep_tile_configs and tensorir_args.random_sweep_tile_configs > 0:
        import time

        random.seed(int(time.time()))
        random.shuffle(configs)
        configs = configs[: tensorir_args.random_sweep_tile_configs]

    print(f"Generated {len(configs)} configs")

    return configs


def get_tensorir_compilation_config(m, n, k, matmul_element_bits, tensorir_args):
    """Get a single tensor IR compilation configuration.

    Args:
        m, n, k: Matrix dimensions
        matmul_element_bits: Element size in bits
        tensorir_args: User-provided arguments to override defaults

    Returns:
        List of configuration parameters
    """
    default_mma_k = 256 // matmul_element_bits

    kphase = 4

    # Initialize default configuration values
    defaults = {
        "tile_size": [128, 128, kphase * default_mma_k],
        "mma_shape": [128, 128, default_mma_k],
        "cluster_shape": [1, 1, 1],
        "cta_count": 1,
        "stream_k": False,
        "cubin_chip": get_default_cubin_chip(),
    }

    # Override defaults with user-provided arguments
    # Example: -tile_size 128,128,64 -mma_shape 64,128,16 -cluster_shape 1,2,1 -cta_count 2 -stream_k false -cubin_chip sm_100a
    config_values = {
        "tile_size": get_optional_arg(tensorir_args, "tile_size", parse_int_list, defaults["tile_size"]),
        "mma_shape": get_optional_arg(tensorir_args, "mma_shape", parse_int_list, defaults["mma_shape"]),
        "cluster_shape": get_optional_arg(tensorir_args, "cluster_shape", parse_int_list, defaults["cluster_shape"]),
        "cta_count": get_optional_arg(tensorir_args, "cta_count", int, defaults["cta_count"]),
        "stream_k": get_optional_arg(tensorir_args, "stream_k", bool, defaults["stream_k"]),
        "cubin_chip": get_optional_arg(tensorir_args, "cubin_chip", default=defaults["cubin_chip"]),
    }

    # Return configuration as list for backward compatibility
    return [
        config_values["tile_size"],
        config_values["mma_shape"],
        config_values["cluster_shape"],
        config_values["cta_count"],
        config_values["stream_k"],
        config_values["cubin_chip"],
        matmul_element_bits,
    ]


def find_mismatches(tensor_a, tensor_b, atol, rtol):
    # Use isclose with equal_nan=True: this will consider NaNs as equal.
    close_mask = torch.isclose(tensor_a, tensor_b, atol=atol, rtol=rtol, equal_nan=True)
    # Get indices where the tensors do not match
    diff_indices = torch.nonzero(~close_mask)

    mismatches = []
    for idx in diff_indices:
        index_tuple = tuple(idx.tolist())
        a_val = tensor_a[index_tuple]
        b_val = tensor_b[index_tuple]

        # Compute the absolute difference. Note: if one value is NaN (and the other is not),
        # the result will be NaN, so we handle that by assigning it an infinite diff.
        diff_val = torch.abs(a_val - b_val)
        if torch.isnan(diff_val):
            diff_val = float("inf")
        else:
            diff_val = diff_val.item()  # Convert tensor to a Python number for sorting

        # Also convert a_val and b_val to Python scalars if possible
        a_num = a_val.item() if hasattr(a_val, "item") else a_val
        b_num = b_val.item() if hasattr(b_val, "item") else b_val
        mismatches.append((index_tuple, a_num, b_num, diff_val))

    # Sort mismatches by the absolute difference in descending order.
    mismatches.sort(key=lambda x: x[3], reverse=True)

    # Print the largest 10 mismatches to reduce output length.
    for idx, a_num, b_num, diff_val in mismatches[:10]:
        print(f"Index: {idx}, Tensor A Value: {a_num}, Tensor B Value: {b_num}, diff = {diff_val}")


def splat_if_scalar(lsh, rsh):
    if isinstance(lsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(rsh.type, (ir.IntegerType, ir.FloatType)):
        raise ValueError(f"One of lsh, rsh must be tensor")

    if isinstance(lsh.type, (ir.IntegerType, ir.FloatType)):
        lsh = nv_tensor_ir.splat(nv_tensor_ir.TensorType.get_from_tensor_type(rsh.type, lsh.type), lsh)

    if isinstance(rsh.type, (ir.IntegerType, ir.FloatType)):
        rsh = nv_tensor_ir.splat(nv_tensor_ir.TensorType.get_from_tensor_type(lsh.type, rsh.type), rsh)
    return lsh, rsh


@dataclass
class TensorInfo:
    tensor_type: nv_tensor_ir.TensorType
    dtype: ir.Type
    shape: list[int]
    stride: list[int]
    stride_div: list[int]
    alignment: Optional[int] = None  # alignment in bytes


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def find_leading_dimension(stride, shape):
    """Find the leading dimension index using a priority-based approach.

    Priority order:
    1. stride == 1 and shape != 1 (meaningful leading dimensions)
    2. Last dimension satisfying stride == 1 (fallback)
    3. None (invalid)

    Args:
        stride_list: List of stride values to search through
        shape_list: List of shape values for priority-based selection

    Returns:
        int or None: Index of the leading dimension, or None if not found
    """
    if len(shape) != len(stride):
        raise ValueError("stride and shape must have the same length")

    # First priority: stride == 1 and shape != 1
    for i in range(len(stride) - 1, -1, -1):
        if stride[i] == 1 and shape[i] != 1:
            return i

    # Second priority: last dimension satisfying stride == 1
    for i in range(len(stride) - 1, -1, -1):
        if stride[i] == 1:
            return i

    # Invalid: no dimension with stride == 1
    return None


def calculate_stride_div_from_alignment(
    stride_dynamic,
    stride,
    dtype_width,
    shape=None,
):
    """Calculate stride divisibility from stride layout.

    Args:
        stride_dynamic: List of stride placeholders (0 for broadcast, 1 for leading, etc.)
        stride: List of actual stride values
        dtype_width: Data type width in bits
        shape: Optional list of shape values; dims with shape==1 are treated as
            broadcast-like (same effect as stride==0), so the TMA alignment div
            still lands on the leading dim under the new
            shape=1+dyn-stride convention.

    Returns:
        List of stride divisibility values
    """

    # Helper function for stride divisibility calculation (optimized with bit manipulation)
    def calculate_actual_divisibility(value):
        if value <= 0:
            return 1
        else:
            temp_value = abs(value)
            # Find highest power of 2 that divides the value using bit manipulation
            # x & (-x) gives the lowest set bit, which is the highest power of 2 divisor
            divisibility = temp_value & (-temp_value)
            # Cap at 32 for practical limits
            return min(max(1, divisibility), 32)

    stride_div = []
    tma_requirement_in_bits = 128
    stride_div_flag = True
    has_broadcast_dim = False

    for idx, s in enumerate(stride_dynamic):
        is_broadcast_dim = (s == 0) or (shape is not None and shape[idx] == 1)
        if is_broadcast_dim:
            stride_div.append(1)
            has_broadcast_dim = True
        else:
            if s != 1:
                if stride_div_flag and has_broadcast_dim:
                    actual_stride_div = calculate_actual_divisibility(stride[idx])
                    final_stride_div = gcd(actual_stride_div * dtype_width, tma_requirement_in_bits) // dtype_width
                    stride_div.append(final_stride_div)
                    stride_div_flag = False
                else:
                    stride_div.append(1)
            else:
                stride_div.append(1)

    if has_broadcast_dim:
        stride_div[0] = gcd(stride[0] * dtype_width, tma_requirement_in_bits) // dtype_width

    return stride_div


class TensorIRNode(ABC):
    """Base class for all Tensor IR operation nodes."""

    def __init__(
        self,
        node,
        node_map,
        ip,
        tensor_ir_test,
    ):
        self.node = node
        self.node_map = node_map
        self.ip = ip
        self.tensor_ir_test = tensor_ir_test
        self.children = [node_map[child] for child in node.producer_nodes]
        # Get output tensor info with compute data type from node kwargs or output data type
        self.output_tensor_info = self.tensor_ir_test.determine_tensor_ir_inout_tensor_type(
            node,
            (node.kwargs["compute_data_type"] if "compute_data_type" in node.kwargs else node.output[0].data_type),
        )

    @abstractmethod
    def run(self):
        """Run the operation and store the result in node_map"""
        pass

    def convert_and_splat_for_binary_pointwise(self, lsh, rsh, tensor_info):
        """Helper method for binary operations to handle input type conversions"""
        return self.tensor_ir_test.convert_and_splat_for_binary_pointwise(lsh, rsh, tensor_info)

    def convert_scalar_tensor(self, scalar_tensor, target_type):
        """Helper method to convert scalar tensors"""
        return self.tensor_ir_test.convert_scalar_tensor(scalar_tensor, target_type)

    def is_float(self, value):
        if isinstance(nv_tensor_ir.get_tensor_datatype(value.type), ir.FloatType):
            return True
        else:
            return False

    def is_integer(self, value):
        if isinstance(nv_tensor_ir.get_tensor_datatype(value.type), ir.IntegerType):
            return True
        else:
            return False


class ReductionNode(TensorIRNode):
    """Node for reduction operations"""

    def run(self):
        with self.ip:
            accumulator_type = ir.TypeAttr.get(eval(convert_datatype(self.tensor_ir_test.test_graph.compute_data_type, "tensorir")))

            reduction_mode = None
            if "reduction_mode.ADD" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.add
            elif "reduction_mode.AMAX" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.amax
            elif "reduction_mode.MIN" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.min
            elif "reduction_mode.MAX" in self.node.kwargs["mode"]:
                reduction_mode = nv_tensor_ir.ReductionMode.max

            input_datatype = nv_tensor_ir.get_tensor_datatype(self.children[0].type)
            output_datatype = nv_tensor_ir.get_tensor_datatype(self.output_tensor_info.tensor_type)

            if input_datatype != output_datatype:
                convert_value = nv_tensor_ir.convert(
                    nv_tensor_ir.TensorType.get(shape=self.output_tensor_info.shape, datatype=output_datatype),
                    self.children[0],
                )
            else:
                convert_value = self.children[0]

            # Detect reduction dimensions by comparing input and output shapes.
            # stride==0 is unreliable for outputs with non-broadcast physical layout
            # (e.g. shape=(B,1,N) with stride=(N,N,1) — no zero stride despite being reduced).
            input_shape = self.node.producer_nodes[0].output[0].dim
            output_shape = self.node.output[0].dim
            reduction_dimensions = [i for i, (in_d, out_d) in enumerate(zip(input_shape, output_shape)) if in_d != out_d]

            mlir_value = nv_tensor_ir.reduce(
                nv_tensor_ir.TensorType.get(shape=self.output_tensor_info.shape, datatype=output_datatype),
                convert_value,
                reduction_dimensions,
                reduction_mode,
            )

            self.node_map[self.node] = mlir_value


class MatmulNode(TensorIRNode):
    """Node for matmul operations"""

    def _convert_input_to_original_type(self, input_value, producer_node_idx):
        """Convert input to its original JSON-defined tensor type if needed."""
        producer_node = self.node.producer_nodes[producer_node_idx]
        original_type = producer_node.output[0].data_type

        tensor_info = self.tensor_ir_test.determine_tensor_ir_inout_tensor_type(producer_node, original_type)

        # Only convert if datatypes differ
        if nv_tensor_ir.get_tensor_datatype(input_value.type) != nv_tensor_ir.get_tensor_datatype(tensor_info.tensor_type):
            return nv_tensor_ir.convert(tensor_info.tensor_type, input_value)

        return input_value

    def run(self):
        with self.ip:
            # Convert inputs to their original JSON-defined tensor types
            lsh = self._convert_input_to_original_type(self.children[0], 0)
            rsh = self._convert_input_to_original_type(self.children[1], 1)

            # Get the appropriate output tensor type
            output_type = self.output_tensor_info.tensor_type

            # Perform matmul operation
            self.node_map[self.node] = nv_tensor_ir.matmul(output_type, lsh, rsh)


class ScaledMatmulNode(MatmulNode):
    """Node for scaled matmul operations, which inherits from `MatmulNode`"""

    def run(self):
        with self.ip:
            # Convert inputs to their original JSON-defined tensor types
            A = self._convert_input_to_original_type(self.children[0], 0)
            sfA = self._convert_input_to_original_type(self.children[1], 1)
            B = self._convert_input_to_original_type(self.children[2], 2)
            sfB = self._convert_input_to_original_type(self.children[3], 3)

            # Get the appropriate output tensor type
            output_type = self.output_tensor_info.tensor_type

            # Perform scaled matmul operation
            self.node_map[self.node] = nv_tensor_ir.scaled_matmul(output_type, A, sfA, B, sfB)


class BinaryOperationNode(TensorIRNode):
    """Node for binary operations"""

    # Map operation names to unified tensor operation functions
    BINARY_OP_MAP = {
        "add": nv_tensor_ir.add,
        "bias": nv_tensor_ir.add,
        "sub": nv_tensor_ir.sub,
        "mul": nv_tensor_ir.mul,
        "max": nv_tensor_ir.max,
        "min": nv_tensor_ir.min,
        "div": nv_tensor_ir.div,
        "mod": nv_tensor_ir.mod,
        "pow": nv_tensor_ir.pow,
        "add_square": nv_tensor_ir.add_square,
        "atan2": nv_tensor_ir.atan2,
        "relu_backward": nv_tensor_ir.relu_bwd,
        "tanh_backward": nv_tensor_ir.tanh_bwd,
        "sigmoid_backward": nv_tensor_ir.sigmoid_bwd,
        "gelu_backward": nv_tensor_ir.gelu_bwd,
        "gelu_approx_tanh_backward": nv_tensor_ir.gelu_approx_tanh_bwd,
        "logical_or": nv_tensor_ir.or_,
        "logical_and": nv_tensor_ir.and_,
    }

    def run(self):
        with self.ip:
            lsh, rsh = self.children[0], self.children[1]

            lsh, rsh = self.convert_and_splat_for_binary_pointwise(lsh, rsh, self.output_tensor_info)

            op_name = self.node.op_name
            if op_name not in self.BINARY_OP_MAP:
                print(f"Unimplemented binary Operation in Lowering to Tensor IR: {op_name}")
                return

            # Get the unified operation function
            op_func = self.BINARY_OP_MAP[op_name]

            # Handle operations with special calling conventions
            params = inspect.signature(op_func).parameters
            if not "output" in params.keys():
                self.node_map[self.node] = op_func(lsh, rsh)
            else:
                self.node_map[self.node] = op_func(self.output_tensor_info.tensor_type, lsh, rsh)


class UnaryOperationNode(TensorIRNode):
    """Node for unary operations"""

    # Map operation names to unified tensor operation functions
    UNARY_OP_MAP = {
        "abs": nv_tensor_ir.abs,
        "tanh": nv_tensor_ir.tanh_fwd,
        "ceil": nv_tensor_ir.ceil,
        "floor": nv_tensor_ir.floor,
        "cos": nv_tensor_ir.cos,
        "sin": nv_tensor_ir.sin,
        "tan": nv_tensor_ir.tan,
        "exp": nv_tensor_ir.exp,
        "log": nv_tensor_ir.log,
        "neg": nv_tensor_ir.neg,
        "rsqrt": nv_tensor_ir.rsqrt,
        "sqrt": nv_tensor_ir.sqrt,
        "erf": nv_tensor_ir.erf,
        "reciprocal": nv_tensor_ir.reciprocal,
        "relu": nv_tensor_ir.relu_fwd,
        "sigmoid": nv_tensor_ir.sigmoid_fwd,
        "gelu": nv_tensor_ir.gelu_fwd,
        "gelu_approx_tanh": nv_tensor_ir.gelu_approx_tanh_fwd,
        "logical_not": nv_tensor_ir.not_,
    }

    def run(self):
        with self.ip:
            if nv_tensor_ir.get_tensor_datatype(self.output_tensor_info.tensor_type) != nv_tensor_ir.get_tensor_datatype(self.children[0].type):
                convert_value = nv_tensor_ir.convert(self.output_tensor_info.tensor_type, self.children[0])
            else:
                convert_value = self.children[0]

            op_name = self.node.op_name
            if op_name not in self.UNARY_OP_MAP:
                print(f"Unimplemented unary Operation in Lowering to Tensor IR: {op_name}")
                return

            # Get the unified operation function
            op_func = self.UNARY_OP_MAP[op_name]

            # Execute the operation - all unary ops use the same calling convention
            self.node_map[self.node] = op_func(convert_value)


class ComparatorNode(TensorIRNode):
    """Node for comparison operations"""

    # Map operation names to comparator types
    COMPARATOR_MAP = {
        "float": {
            "cmp_lt": nv_tensor_ir.Comparator.olt,
            "cmp_ge": nv_tensor_ir.Comparator.oge,
            "cmp_gt": nv_tensor_ir.Comparator.ogt,
            "cmp_le": nv_tensor_ir.Comparator.ole,
            "cmp_eq": nv_tensor_ir.Comparator.oeq,
            "cmp_ne": nv_tensor_ir.Comparator.one,
        },
        "integer": {
            "cmp_lt": nv_tensor_ir.Comparator.lt,
            "cmp_ge": nv_tensor_ir.Comparator.ge,
            "cmp_gt": nv_tensor_ir.Comparator.gt,
            "cmp_le": nv_tensor_ir.Comparator.le,
            "cmp_eq": nv_tensor_ir.Comparator.eq,
            "cmp_ne": nv_tensor_ir.Comparator.neq,
        },
    }

    def run(self):
        with self.ip:
            op_name = self.node.op_name
            if op_name not in self.COMPARATOR_MAP["float"] and op_name not in self.COMPARATOR_MAP["integer"]:
                print(f"Unimplemented comparator operation: {op_name}")
                return

            lsh, rsh = splat_if_scalar(self.children[0], self.children[1])

            # The input tensors should be both integer or both float
            if self.is_integer(lsh) and self.is_integer(rsh):
                self.node_map[self.node] = nv_tensor_ir.cmp(
                    self.COMPARATOR_MAP["integer"][op_name],
                    lsh,
                    rsh,
                )
            elif self.is_float(lsh) and self.is_float(rsh):
                self.node_map[self.node] = nv_tensor_ir.cmp(
                    self.COMPARATOR_MAP["float"][op_name],
                    lsh,
                    rsh,
                )
            else:
                raise ValueError(f"Comparator {op_name} requires float or integer tensors")


class IdentityNode(TensorIRNode):
    """Node for identity operations"""

    def run(self):
        with self.ip:
            child = self.children[0]

            if isinstance(child.type, (ir.IntegerType, ir.FloatType)) and isinstance(self.output_tensor_info.tensor_type, nv_tensor_ir.TensorType):
                child = nv_tensor_ir.splat(
                    nv_tensor_ir.TensorType.get_from_tensor_type(self.output_tensor_info.tensor_type, child.type),
                    child,
                )

            if nv_tensor_ir.get_tensor_datatype(self.output_tensor_info.tensor_type) != nv_tensor_ir.get_tensor_datatype(child.type):
                self.node_map[self.node] = nv_tensor_ir.convert(
                    self.output_tensor_info.tensor_type,
                    child,
                )
            else:
                self.node_map[self.node] = child


class ConvolutionNode(TensorIRNode):
    """Base node for convolution operations"""

    def _prepare_common_params(self):
        accumulator_type_attr = ir.TypeAttr.get(eval(convert_datatype(self.tensor_ir_test.test_graph.compute_data_type, "tensorir")))
        pre_padding = self.node.kwargs["padding"]
        post_padding = self.node.kwargs["padding"]  # FIXME: what should this be?
        stride = self.node.kwargs["stride"]
        dilation = self.node.kwargs["dilation"]

        return accumulator_type_attr, pre_padding, post_padding, stride, dilation


class ConvFpropNode(ConvolutionNode):
    """Node for forward convolution"""

    def run(self):
        with self.ip:
            x = self.children[0]
            w = self.children[1]

            accumulator_type_attr, pre_padding, post_padding, stride, dilation = self._prepare_common_params()

            self.node_map[self.node] = nv_tensor_ir.conv_fprop(
                self.output_tensor_info.tensor_type,
                x,
                w,
                accumulator_type_attr,
                pre_padding,
                post_padding,
                stride,
                dilation,
            )


class ConvDgradNode(ConvolutionNode):
    """Node for gradient w.r.t. data convolution"""

    def run(self):
        with self.ip:
            dy = self.children[0]
            w = self.children[1]

            accumulator_type_attr, pre_padding, post_padding, stride, dilation = self._prepare_common_params()

            self.node_map[self.node] = nv_tensor_ir.conv_dgrad(
                self.output_tensor_info.tensor_type,
                dy,
                w,
                accumulator_type_attr,
                pre_padding,
                post_padding,
                stride,
                dilation,
            )


class ConvWgradNode(ConvolutionNode):
    """Node for gradient w.r.t. weight convolution"""

    def run(self):
        with self.ip:
            dy = self.children[0]
            w = self.children[1]

            accumulator_type_attr, pre_padding, post_padding, stride, dilation = self._prepare_common_params()

            self.node_map[self.node] = nv_tensor_ir.conv_dgrad(
                self.output_tensor_info.tensor_type,
                dy,
                w,
                accumulator_type_attr,
                pre_padding,
                post_padding,
                stride,
                dilation,
            )


class ActivationForwardNode(TensorIRNode):
    """Node for activation forward operations that take beta parameter"""

    # Map operation names to activation functions
    ACTIVATION_MAP = {
        "swish": nv_tensor_ir.swish_fwd,
        "softplus": nv_tensor_ir.softplus_fwd,
        "elu": nv_tensor_ir.elu_fwd,
    }

    def run(self):
        with self.ip:
            beta = self.tensor_ir_test.get_beta_attr()
            if beta is None:
                raise ValueError(f"Beta attribute not found for {self.node.op_name}")

            converted_x, _ = self.convert_and_splat_for_binary_pointwise(self.children[0], None, self.output_tensor_info)

            # Ensure we have float tensors for activations
            if not self.is_float(converted_x):
                raise ValueError(f"Activation operation {self.node.op_name} requires float tensors")

            op_name = self.node.op_name
            if op_name in self.ACTIVATION_MAP:
                self.node_map[self.node] = self.ACTIVATION_MAP[op_name](converted_x, beta)
            else:
                print(f"Unimplemented activation forward operation: {op_name}")


class ActivationBackwardNode(TensorIRNode):
    """Node for activation backward operations that take beta parameter"""

    # Map operation names to activation backward functions
    ACTIVATION_BWD_MAP = {
        "swish_backward": nv_tensor_ir.swish_bwd,
        "softplus_backward": nv_tensor_ir.softplus_bwd,
        "elu_backward": nv_tensor_ir.elu_bwd,
    }

    def run(self):
        with self.ip:
            beta = self.tensor_ir_test.get_beta_attr()
            if beta is None:
                raise ValueError(f"Beta attribute not found for {self.node.op_name}")

            converted_x, converted_grad = self.convert_and_splat_for_binary_pointwise(self.children[0], self.children[1], self.output_tensor_info)

            # Ensure we have float tensors for activation gradients
            if not self.is_float(converted_x) or not self.is_float(converted_grad):
                raise ValueError(f"Activation gradient operation {self.node.op_name} requires float tensors")

            op_name = self.node.op_name
            if op_name in self.ACTIVATION_BWD_MAP:
                self.node_map[self.node] = self.ACTIVATION_BWD_MAP[op_name](converted_x, converted_grad, beta)
            else:
                print(f"Unimplemented activation backward operation: {op_name}")


class BinarySelectNode(TensorIRNode):
    """Node for binary select operations"""

    def run(self):
        with self.ip:
            condition = self.children[0]
            true_value = self.children[1]
            false_value = self.children[2]
            true_value, false_value = splat_if_scalar(true_value, false_value)
            # Type checking: condition must be integer/boolean, true/false values must be float
            if not self.is_integer(condition):
                raise ValueError(f"binary_select condition (first operand) must be an integer tensor, got {condition.type}")

            if not self.is_float(true_value) or not self.is_float(false_value):
                raise ValueError(f"binary_select true and false values (second and third operands) must be float tensors")

            self.node_map[self.node] = nv_tensor_ir.binary_select(condition, true_value, false_value)


class test_tensor_ir:
    # Group operations by their handler class for cleaner lookup
    OPERATION_GROUPS = {
        ReductionNode: ["reduction"],
        MatmulNode: ["matmul"],
        ScaledMatmulNode: ["scaled_matmul"],
        BinaryOperationNode: [
            "add",
            "bias",
            "sub",
            "mul",
            "max",
            "min",
            "div",
            "mod",
            "atan2",
            "pow",
            "add_square",
            "logical_or",
            "logical_and",
            "relu_backward",
            "tanh_backward",
            "sigmoid_backward",
            "gelu_backward",
            "gelu_approx_tanh_backward",
        ],
        UnaryOperationNode: [
            "tanh",
            "abs",
            "ceil",
            "floor",
            "cos",
            "sin",
            "tan",
            "exp",
            "log",
            "neg",
            "rsqrt",
            "sqrt",
            "erf",
            "logical_not",
            "reciprocal",
            "relu",
            "sigmoid",
            "gelu",
            "gelu_approx_tanh",
        ],
        ComparatorNode: ["cmp_lt", "cmp_ge", "cmp_gt", "cmp_le", "cmp_eq", "cmp_ne"],
        BinarySelectNode: ["binary_select"],
        ActivationForwardNode: ["swish", "softplus", "elu"],
        ActivationBackwardNode: ["swish_backward", "softplus_backward", "elu_backward"],
        IdentityNode: ["identity"],
        ConvWgradNode: ["conv_wgrad"],
        ConvDgradNode: ["conv_dgrad"],
        ConvFpropNode: ["conv_fprop"],
    }
    # Compute capability to cubin chip mapping
    SUPPORTED_CUBIN_CHIP = _SUPPORTED_CUBIN_CHIP

    # Map of operation names to node classes - built from the operation groups
    NODE_CLASS_MAP = {}

    # Build the NODE_CLASS_MAP from the operation groups
    @classmethod
    def _init_node_class_map(cls):
        for node_class, op_names in cls.OPERATION_GROUPS.items():
            for op_name in op_names:
                cls.NODE_CLASS_MAP[op_name] = node_class

    def __init__(self, test_graph, static_shapes_only, compiler_backend="Tile"):
        self.test_graph = test_graph
        self.outputs = []
        self.ref_outputs = None
        self.static_shapes_only = static_shapes_only
        self.compiler_backend = compiler_backend
        self.node_overwrite_stride_func_map = {}

        # Initialize the node class map if it's empty
        if not self.NODE_CLASS_MAP:
            self._init_node_class_map()

        self.compiler_with_kernel_cache = CompilerWithKernelCacheSingleton()

    def determine_tensor_ir_inout_tensor_type(self, node, dtype=None):
        assert len(node.output) == 1
        test_tensor = node.output[0]
        if dtype is None:
            dtype = eval(convert_datatype(test_tensor.data_type, "tensorir"))
        else:
            dtype = eval(convert_datatype(dtype, "tensorir"))

        if hasattr(node, "is_by_value") and node.is_by_value:
            return TensorInfo(
                tensor_type=dtype,
                dtype=dtype,
                shape=[1],
                stride=[1],
                stride_div=[],
            )

        shape = []
        stride = []

        ori_stride = test_tensor.stride
        ori_shape = test_tensor.dim

        # Check if tensor is a scalar tensor (all strides 1 and all dims 1 in json definition)
        isScalarTensor = all(s == 1 for s in ori_stride) and all(d == 1 for d in ori_shape)
        # Calculate alignment first using the corrected approach
        dtype_width = dtype.width if hasattr(dtype, "width") else 16  # fallback to 16 for compatibility

        # Get TMA alignment
        def get_tma_alignment(dtype_width):
            if dtype_width >= 8:
                return 16  # bytes
            else:
                return 32  # bytes

        if isScalarTensor:
            # Unified broadcast encoding for both CudaTile and Collective backends:
            # static shape=1 + dynamic stride. The frontend emits an explicit
            # nv_tensor_ir.broadcast op when the consumer expects a wider shape.
            shape = [1] * len(ori_shape)
            stride = [1 if self.static_shapes_only else -1] * len(ori_stride)
            stride_div = [1] * len(ori_stride)
            return TensorInfo(
                tensor_type=nv_tensor_ir.TensorType.get(shape=shape, datatype=dtype),
                dtype=dtype,
                shape=shape,
                stride=stride,
                stride_div=stride_div,
                alignment=get_tma_alignment(dtype_width),
            )

        stride_div = []

        for s, d in zip(ori_stride, ori_shape):
            if d == 1:
                # Unified broadcast encoding for both backends: static shape=1
                # + dynamic stride. Triggers an explicit nv_tensor_ir.broadcast
                # op in the graph when the consumer expects a wider shape.
                shape.append(1)
                stride.append(1 if self.static_shapes_only else -1)
            else:
                if s != 1:
                    stride.append(s if self.static_shapes_only else -1)
                else:
                    stride.append(1)
                shape.append(d if self.static_shapes_only else -1)

        # Find leading dimension and calculate alignment
        leading_dim_idx = find_leading_dimension(stride, ori_shape)

        if leading_dim_idx is not None:
            leading_dim_size = ori_shape[leading_dim_idx]
            tma_requirement_bytes = get_tma_alignment(dtype_width)
            tma_requirement_bits = tma_requirement_bytes * 8
            alignment_bits = gcd(leading_dim_size * dtype_width, tma_requirement_bits)
            alignment = alignment_bits // 8  # convert to bytes
        else:
            alignment = get_tma_alignment(dtype_width)

        # Calculate stride divisibility from alignment
        stride_div = calculate_stride_div_from_alignment(
            stride,
            ori_stride,
            dtype_width,
            shape=shape,
        )

        reorder_mode = self.get_reorder_mode(node)

        ty = nv_tensor_ir.TensorType.get(
            shape=shape,
            datatype=dtype,
            reorder_mode=reorder_mode,
        )
        return TensorInfo(
            tensor_type=ty,
            dtype=dtype,
            shape=shape,
            stride=stride,
            stride_div=stride_div,
            alignment=alignment,
        )

    def calc_ref(self):
        self.ref_outputs = self.test_graph.calc_reference()

    def cleanup(self):
        """
        Clean up GPU memory by explicitly deleting tensor references and forcing garbage collection.

        This method should be called after test execution to free GPU memory.
        It clears:
        - Output tensors (self.outputs)
        - Reference outputs (self.ref_outputs)
        - Any cached data in the test_graph
        """
        import torch
        import gc

        # Clear output tensors
        if hasattr(self, "outputs") and self.outputs:
            for tensor in self.outputs:
                if torch.is_tensor(tensor) and tensor.is_cuda:
                    del tensor
            self.outputs.clear()
            self.outputs = []

        # Clear reference outputs
        if hasattr(self, "ref_outputs") and self.ref_outputs:
            if isinstance(self.ref_outputs, list):
                for tensor in self.ref_outputs:
                    if torch.is_tensor(tensor):
                        del tensor
                self.ref_outputs.clear()
            self.ref_outputs = None

        # Clear any cached nodes data
        if hasattr(self, "test_graph") and hasattr(self.test_graph, "nodes"):
            for node in self.test_graph.nodes:
                if hasattr(node, "output"):
                    for output_tensor in node.output:
                        output_tensor.cleanTensorData()

        # Clear entrance nodes cached values and GPU tensors
        if hasattr(self, "test_graph") and hasattr(self.test_graph, "entrance_nodes"):
            for node in self.test_graph.entrance_nodes:
                if hasattr(node, "output"):
                    for output_tensor in node.output:
                        output_tensor.cleanTensorData()

        # Final garbage collection pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def tensorir_compare_to_reference(self, atol=1e-2, rtol=1e-2):
        assert len(self.ref_outputs) == len(self.outputs)

        # Use the base class method for comparison
        return tg.test_graph.compare_to_reference(self.ref_outputs, self.outputs, atol=atol, rtol=rtol)

    def get_beta_attr(self):
        for node in self.test_graph.entrance_nodes:
            if node.name == "beta":
                beta_tensor = node.get_value()
                # Handle beta tensor of any dimension by accessing the first element
                return beta_tensor.item() if beta_tensor.numel() == 1 else beta_tensor.flatten()[0].item()
        return None

    def _get_reduction_output_identity(self, node, torch_dtype):
        if not (isinstance(node, tg.operation) and node.op_name == "reduction"):
            return None

        mode = str(node.kwargs.get("mode", ""))
        if "reduction_mode.ADD" in mode or "reduction_mode.AMAX" in mode:
            return 0.0
        if "reduction_mode.MUL" in mode:
            return 1.0
        if "reduction_mode.MAX" in mode:
            if torch_dtype.is_floating_point:
                return torch.finfo(torch_dtype).min
            if torch_dtype == torch.bool:
                return False
            return torch.iinfo(torch_dtype).min
        if "reduction_mode.MIN" in mode:
            if torch_dtype.is_floating_point:
                return torch.finfo(torch_dtype).max
            if torch_dtype == torch.bool:
                return True
            return torch.iinfo(torch_dtype).max
        return None

    def run_tensor_ir_module(
        self,
        module,
        compiler_backend,
        kernel_configs,
        dump_ir_path,
        load_ir_path,
        mlir_timing,
        timing_loop=1,
        atol=1e-2,
        rtol=1e-2,
    ):
        device = torch.device("cuda")

        # Prepare inputs - scalars stay on CPU, tensors go to GPU
        desc_inputs = []
        inputs_gpu = []

        node_overwrite_stride_tensor_map = {}

        for node in self.test_graph.entrance_nodes:
            torch_mem = node.get_value()
            if not node.output[0].is_by_value:
                # Tensor operand - move to GPU
                gpu_mem = torch.as_strided(
                    torch_mem.to(device=device),
                    size=torch_mem.shape,
                    stride=torch_mem.stride(),
                )
                inputs_gpu.append(gpu_mem)  # need to save gpu_mem in case of releasing
                desc_inputs.append(nv_tensor_ir.TensorIRTensorDescriptor(gpu_mem))
                if node in self.node_overwrite_stride_func_map:
                    node_overwrite_stride_tensor_map[node] = desc_inputs[-1]
            else:
                # Scalar operand - keep on CPU for automatic detection
                cpu_mem = torch_mem.to("cpu")
                desc_inputs.append(nv_tensor_ir.TensorIRTensorDescriptor(cpu_mem))

        if not self.ref_outputs:
            self.calc_ref()

        # Create output tensors on GPU.
        output_nodes = [node for node in self.test_graph.nodes if node.is_output_node()]
        outputs_gpu = []
        for node in output_nodes:
            output_dtype = eval(convert_datatype(node.output[0].data_type, "torch"))
            output_tensor = torch.empty_strided(
                size=tuple(node.output[0].dim),
                stride=tuple(node.output[0].stride),
                dtype=output_dtype,
                device=device,
            )
            reduction_identity = self._get_reduction_output_identity(node, output_dtype)
            if reduction_identity is not None:
                output_tensor.fill_(reduction_identity)
            outputs_gpu.append(output_tensor)

        # Add output tensors to DLPack inputs
        desc_outputs = [nv_tensor_ir.TensorIRTensorDescriptor(gpu_tensor) for gpu_tensor in outputs_gpu]
        all_desc = desc_inputs + desc_outputs

        with ir.Context() as ctx, ir.Location.unknown():
            nv_tensor_ir.register_dialect()

            best_perf = float("inf")
            best_config = dict(tile_size=[], mma_shape=[], cluster_shape=[], cta_count=[])
            finished_cnt = 0

            if compiler_backend == "Collective":
                for (
                    tile_size,
                    mma_shape,
                    cluster_shape,
                    cta_count,
                    stream_k,
                    cubin_chip,
                    matmul_element_bits,
                ) in kernel_configs:
                    cc = self.compiler_with_kernel_cache.get_compute_capability()
                    if not nv_tensor_ir.SM10xCompilationConfig.isValidConfiguration(
                        nv_tensor_ir.MmaShape(mma_shape[0], mma_shape[1], mma_shape[2]),
                        nv_tensor_ir.MmaShape(tile_size[0], tile_size[1], tile_size[2]),
                        cta_count,
                        matmul_element_bits,
                        nv_tensor_ir.MmaShape(cluster_shape[0], cluster_shape[1], cluster_shape[2]),
                        cc,
                    ):
                        print(f"#### Invalid configuration: {tile_size}, {mma_shape}, {cluster_shape}, {cta_count}")
                        print(f"#### Skip this config")
                        continue
                    print(f"\n#### Valid configuration: {tile_size}, {mma_shape}, {cluster_shape}, {cta_count}")

                    # TODO: Add enum to support more cubin_chip
                    expected_cubin_chip = _get_cubin_chip_for_compute_capability(cc)
                    if expected_cubin_chip != cubin_chip:
                        print(f"#### cubin_chip={cubin_chip} is not supported for cc={cc}; expected {expected_cubin_chip}")
                        print(f"#### Skip this config")
                        continue
                    cc = nv_tensor_ir.ComputeCapability(cc)
                    compile_options = nv_tensor_ir.TensorIRCompilationOptions(
                        cc,
                        nv_tensor_ir.CompilerBackend.Collective,
                        nv_tensor_ir.TensorConversionOptions(
                            tile_size,
                            mma_shape,
                            cluster_shape,
                            cta_count,
                            (nv_tensor_ir.TileSchedulerType.StreamK if stream_k else nv_tensor_ir.TileSchedulerType.Default),
                            cubin_chip,
                        ),
                        nv_tensor_ir.DebugOptions(dump_ir_path, load_ir_path, mlir_timing),
                    )

                    print(
                        f"#### Running tile_size={tile_size}, mma_shape={mma_shape}, cluster_shape={cluster_shape}, cta_count={cta_count}, stream_k={stream_k}, cubin_chip={cubin_chip}"
                    )

                    if self.compiler_with_kernel_cache.can_compile(module, compile_options):
                        print(f"#### Can compile")
                    else:
                        print(f"#### Cannot compile")
                        continue

                    cloned_module = ir.Module.parse(str(module))
                    shader = self.compiler_with_kernel_cache.compile(cloned_module, compile_options)

                    for node, tensor_desc in node_overwrite_stride_tensor_map.items():
                        overwrite_stride_func = self.node_overwrite_stride_func_map[node]
                        overwrite_stride = overwrite_stride_func(tile_size)
                        tensor_desc.overwrite_strides(overwrite_stride)

                    try:
                        execution_plan = nv_tensor_ir.ExecutionPlan(shader, *all_desc)
                    except Exception as e:
                        print(f"#### Failed to create execution plan: {e}")
                        continue
                    device_workspace_size = execution_plan.query_max_device_workspace_size()
                    print(f"#### Device workspace size: {device_workspace_size}")
                    device_workspace_mem_cpu = torch.zeros(device_workspace_size, dtype=torch.int8, device="cpu")
                    device_workspace_mem_gpu = device_workspace_mem_cpu.clone().detach().to("cuda")
                    device_workspace = nv_tensor_ir.DeviceWorkspace(device_workspace_mem_gpu.data_ptr(), device_workspace_size)
                    # TODO: Do we need to dump the launch config for debugging?
                    # execution_plan.dump_launch_config()
                    if timing_loop == 0:
                        execution_plan.launch(device_workspace)
                        self.outputs = outputs_gpu
                        if not self.ref_outputs:
                            self.calc_ref()
                        torch.cuda.synchronize()
                        passed = self.tensorir_compare_to_reference(atol, rtol)
                        assert passed, "Mismatch between TensorIR and reference"
                    elif timing_loop == 1:
                        execution_plan.launch(device_workspace)
                        torch.cuda.synchronize()
                    else:
                        # warm the caches
                        execution_plan.launch(device_workspace)
                        import utils

                        # TODO: Shall we use median instead of average?
                        avg_rt = utils.measure_gpu_runtime_with_events(lambda: execution_plan.launch(device_workspace), timing_loop)
                        if avg_rt < best_perf:
                            best_perf = avg_rt
                            best_config = {
                                "tile_size": tile_size,
                                "mma_shape": mma_shape,
                                "cluster_shape": cluster_shape,
                                "cta_count": cta_count,
                            }
                        torch.cuda.synchronize()
                    finished_cnt += 1
            else:
                _BACKEND_ENUM_MAP = {
                    "Tile": nv_tensor_ir.CompilerBackend.Tile,
                    "CudaTile": nv_tensor_ir.CompilerBackend.CudaTile,
                }
                backend_enum = _BACKEND_ENUM_MAP.get(compiler_backend, nv_tensor_ir.CompilerBackend.Tile)

                # Waive CudaTile tests that use ops with no CudaTile lowering pattern.
                if compiler_backend == "CudaTile":
                    unsupported = [
                        node.op_name for node in self.test_graph.nodes if isinstance(node, tg.operation) and node.op_name in _CUDA_TILE_UNSUPPORTED_OPS
                    ]
                    # erf and gelu_fwd only support f32/f64 operands in CudaTile;
                    # waive when the operation's output type is f16 or bf16.
                    if not unsupported:
                        unsupported = [
                            node.op_name
                            for node in self.test_graph.nodes
                            if isinstance(node, tg.operation)
                            and node.op_name in _CUDA_TILE_F32_ONLY_OPS
                            and node.output[0].data_type in (DataType.HALF, DataType.BFLOAT16)
                        ]
                    if unsupported:
                        print(f"#### CudaTile does not support op(s) {unsupported}; waiving test")
                        return StatusCode.WAIVED

                tile_cc = self.compiler_with_kernel_cache.get_compute_capability()
                tile_cubin_chip = _get_cubin_chip_for_compute_capability(tile_cc)
                for config in kernel_configs:
                    tile_size = config[0]  # Extract first value from the config list
                    cloned_module = ir.Module.parse(str(module))
                    conversion_options = nv_tensor_ir.TensorConversionOptions()
                    conversion_options.tileSize = tile_size
                    if compiler_backend == "CudaTile":
                        conversion_options.computeCapability = tile_cubin_chip

                    enable_timing = False
                    compile_options = nv_tensor_ir.TensorIRCompilationOptions(
                        tile_cc,
                        backend_enum,
                        conversion_options,
                        nv_tensor_ir.DebugOptions(dump_ir_path, load_ir_path, enable_timing),
                    )
                    print(f"#### Running tile_size={conversion_options.tileSize}, compiler_backend={compiler_backend}")
                    shader = self.compiler_with_kernel_cache.compile(cloned_module, compile_options)
                    execution_plan = nv_tensor_ir.ExecutionPlan(shader, *all_desc)
                    device_workspace_size = execution_plan.query_max_device_workspace_size()
                    print(f"#### Device workspace size: {device_workspace_size}")
                    device_workspace_mem_cpu = torch.zeros(device_workspace_size, dtype=torch.int8, device="cpu")
                    device_workspace_mem_gpu = device_workspace_mem_cpu.clone().detach().to("cuda")
                    device_workspace = nv_tensor_ir.DeviceWorkspace(device_workspace_mem_gpu.data_ptr(), device_workspace_size)
                    if timing_loop == 0:
                        execution_plan.launch(device_workspace)
                        self.outputs = outputs_gpu
                        if not self.ref_outputs:
                            self.calc_ref()
                        torch.cuda.synchronize()
                        passed = self.tensorir_compare_to_reference(atol, rtol)
                        assert passed, "Mismatch between TensorIR and reference"
                    elif timing_loop == 1:
                        execution_plan.launch(device_workspace)
                        torch.cuda.synchronize()
                    else:
                        # warm the caches
                        execution_plan.launch(device_workspace)
                        import utils

                        avg_rt = utils.measure_gpu_runtime_with_events(lambda: execution_plan.launch(device_workspace), timing_loop)
                        if avg_rt < best_perf:
                            best_perf = avg_rt
                            best_config = {
                                "tile_size": tile_size,
                            }
                        torch.cuda.synchronize()
                    finished_cnt += 1
            print(f"@@@@ Best perf achieved is {best_perf / 1000} msec with kernel config: {best_config}")
        print(f"@@@@ Run {finished_cnt} kernels successfully from {len(kernel_configs)} configs")
        if finished_cnt > 0:
            return StatusCode.PASSED
        return StatusCode.WAIVED

    def build_tensor_ir_module(self, json_test_name="graph"):
        input_tensors = []
        for node in self.test_graph.entrance_nodes:
            input_tensors.append(node)
        output_tensors = [node for node in self.test_graph.nodes if node.is_output_node()]

        align_name = "nv_tensor_ir.alignment"
        stride_name = "nv_tensor_ir.stride"
        with ir.Context() as ctx, ir.Location.unknown() as loc:
            ctx.enable_multithreading(False)
            nv_tensor_ir.register_dialect()

            module = ir.Module.create(loc)

            # Convert a stride list to string format, replacing -1 with '?'.
            # E.g. stride =[1, -1, 1] and stride_div = [1, 8, 1] -> "(1,?{div=8},1)"
            def stride_list_to_string(stride, stride_div):
                stride_elements = ["?{div=" + str(div) + "}" if s == -1 else str(s) for s, div in zip(stride, stride_div)]
                return f"({','.join(stride_elements)})"

            input_tensor_infos = list(
                map(
                    lambda x: self.determine_tensor_ir_inout_tensor_type(x),
                    input_tensors,
                )
            )

            arg_attrs = []
            input_types = []
            for tensor_info in input_tensor_infos:
                input_types.append(tensor_info.tensor_type)
                # Add stride and alignment attributes if it's a tensor not a scalar passed by value
                if tensor_info.tensor_type != tensor_info.dtype:
                    stride_str = ir.StringAttr.get(stride_list_to_string(tensor_info.stride, tensor_info.stride_div))
                    assert tensor_info.alignment is not None, "Alignment is required for tensor inputs"
                    align_attr = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(64),
                        tensor_info.alignment,  # use stored alignment
                    )
                    arg_attrs.append(ir.DictAttr.get({stride_name: stride_str, align_name: align_attr}))
                else:
                    arg_attrs.append(ir.DictAttr.get({}))

            output_tensor_infos = list(
                map(
                    lambda x: self.determine_tensor_ir_inout_tensor_type(x),
                    output_tensors,
                )
            )

            res_attrs = []
            output_types = []
            for tensor_info in output_tensor_infos:
                output_types.append(tensor_info.tensor_type)
                # Add stride and alignment attributes if it's a tensor not a scalar passed by value
                if tensor_info.tensor_type != tensor_info.dtype:
                    stride_str = ir.StringAttr.get(stride_list_to_string(tensor_info.stride, tensor_info.stride_div))
                    align_attr = ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(64),
                        tensor_info.alignment,  # use stored alignment
                    )
                    res_attrs.append(ir.DictAttr.get({stride_name: stride_str, align_name: align_attr}))
                else:
                    res_attrs.append(ir.DictAttr.get({}))

            # Create a mlir function signature for the kernel.
            function_type = ir.TypeAttr.get(T.function(inputs=input_types, results=output_types))
            function_name = json_test_name
            # Add a random string to the function name to avoid module collision
            function_name += "_" + str("".join(random.choice(string.ascii_letters + string.digits) for _ in range(20)))

            # Create tensor ir graph
            graph = nv_tensor_ir.graph(
                function_name,
                function_type=function_type,
                arg_attrs=arg_attrs,
                res_attrs=res_attrs,
                ip=ir.InsertionPoint(module.body),
            )

            node_map = {}  # nodes -> tensor_ir values (Do python semantics mess with this?)

            # Generate arguments
            graph.regions[0].blocks.append(*input_types)

            # Assign inputs to arguments
            for i, node in enumerate(input_tensors):
                node_map[node] = graph.regions[0].blocks[0].arguments[i]

            with ir.InsertionPoint(graph.regions[0].blocks[0]) as ip:
                for node, tensor_info in zip(output_tensors, output_tensor_infos):
                    # Recursively lower ops
                    self.build_tensor_ir_recursive(
                        node,
                        node_map,
                        ip,
                        tensor_info,
                    )  # This is passed the insertion point, which provides a mutable reference to the module. # Consider passing other things, as order in graph module is said to not matter.

                # Generate return op
                result = []
                for node in output_tensors:
                    if "compute_data_type" in node.kwargs and node.output[0]._data_type != node.kwargs["compute_data_type"]:
                        tensor_info = self.determine_tensor_ir_inout_tensor_type(node, node.output[0]._data_type)
                        node_map[node] = nv_tensor_ir.convert(
                            tensor_info.tensor_type,
                            node_map[node],
                        )
                    result.append(node_map[node])

                nv_tensor_ir.results_(result)
        module.operation.verify()

        return module

    def convert_and_splat_for_binary_pointwise(self, lsh, rsh, tensor_info):
        if lsh is not None and rsh is not None:
            if isinstance(lsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(rsh.type, nv_tensor_ir.TensorType):
                lsh = nv_tensor_ir.splat(
                    nv_tensor_ir.TensorType.get_from_tensor_type(rsh.type, lsh.type),
                    lsh,
                )
            elif isinstance(rsh.type, (ir.IntegerType, ir.FloatType)) and isinstance(lsh.type, nv_tensor_ir.TensorType):
                rsh = nv_tensor_ir.splat(
                    nv_tensor_ir.TensorType.get_from_tensor_type(lsh.type, rsh.type),
                    rsh,
                )
        out_type_datatype = nv_tensor_ir.get_tensor_datatype(tensor_info.tensor_type)
        if lsh is not None and out_type_datatype != nv_tensor_ir.get_tensor_datatype(lsh.type):
            convert_value0 = nv_tensor_ir.convert(
                nv_tensor_ir.TensorType.get(shape=tensor_info.shape, datatype=out_type_datatype),
                lsh,
            )
        else:
            convert_value0 = lsh
        if rsh is not None and out_type_datatype != nv_tensor_ir.get_tensor_datatype(rsh.type):
            convert_value1 = nv_tensor_ir.convert(
                nv_tensor_ir.TensorType.get(shape=tensor_info.shape, datatype=out_type_datatype),
                rsh,
            )
        else:
            convert_value1 = rsh
        return convert_value0, convert_value1

    def convert_scalar_tensor(self, scalar_tensor, target_type):
        if nv_tensor_ir.get_tensor_datatype(target_type) != nv_tensor_ir.get_tensor_datatype(scalar_tensor.type):
            return nv_tensor_ir.convert(
                nv_tensor_ir.TensorType.get_from_tensor_type(scalar_tensor.type, nv_tensor_ir.get_tensor_datatype(target_type)),
                scalar_tensor,
            )
        else:
            return scalar_tensor

    def build_tensor_ir_recursive(self, node, node_map, ip, output_tensor_info):
        """Build tensor IR representation recursively for the given node."""

        # for node in input_tensors:
        if (not self.static_shapes_only) and (len(node.producer_nodes) == 0):
            input_tensor = node_map[node]
            # Skip scalars / by-value arguments
            if not isinstance(input_tensor.type, nv_tensor_ir.TensorType):
                return

            # Broadcast input tensor to output shape when types differ.
            # Use input dtype so the broadcast doesn't change the element type
            # (e.g. scalar (1,1,1) bias vs output (1,M,N)).  The subsequent
            # convert op handles any dtype difference.
            # TODO:CL-19939 Unify broadcast-dim encoding between CudaTile and Collective backends.
            # This part will be revisited when we unify the broadcast encoding between the CudaTile and Collective backends.
            input_dtype = nv_tensor_ir.get_tensor_datatype(input_tensor.type)
            broadcast_tensor_type = nv_tensor_ir.TensorType.get(
                shape=output_tensor_info.shape,
                datatype=input_dtype,
            )

            if broadcast_tensor_type != input_tensor.type:
                # Only emit when input is actually broadcastable UP to target.
                # Mirror isShapeBroadcastable() in tensor_ir/lib/Utils/Utils.cpp:
                # dim is broadcastable if both dynamic, equal, or fromDim==1.
                # Skips spurious broadcast injection when the final graph
                # output is smaller than leaf inputs (e.g. graphs ending in a
                # reduction, like dBias / MatmulEpilog*Bias). The proper
                # per-edge target-shape fix is tracked by CL-19939.
                def _is_broadcastable_to(from_shape, to_shape):
                    if len(from_shape) != len(to_shape):
                        return False
                    for fd, td in zip(from_shape, to_shape):
                        if fd == -1 and td == -1:
                            continue
                        if fd == -1:
                            return False  # dyn from, static to: can't prove
                        if fd == td or fd == 1:
                            continue
                        return False
                    return True

                # Skip if the broadcast wouldn't actually change shape (e.g.
                # leaf has same shape as target but a different layout attr).
                # The dialect's BroadcastOp::verify rejects no-op broadcasts;
                # the type-mismatch we observed above was on dtype/layout, not
                # shape. Layout-only changes flow naturally without a
                # broadcast op — the consumer handles them.
                def _shapes_equal(from_shape, to_shape):
                    if len(from_shape) != len(to_shape):
                        return False
                    return all(fd == td for fd, td in zip(from_shape, to_shape))

                input_info = self.determine_tensor_ir_inout_tensor_type(node)
                if _is_broadcastable_to(input_info.shape, output_tensor_info.shape) and not _shapes_equal(input_info.shape, output_tensor_info.shape):
                    broadcast_tensor = nv_tensor_ir.broadcast(
                        broadcast_tensor_type,
                        input_tensor,
                    )
                    node_map[node] = broadcast_tensor

        # Skip if node is already processed
        if node in node_map.keys():
            return

        # Process all children first
        for child in node.producer_nodes:
            self.build_tensor_ir_recursive(
                child,
                node_map,
                ip,
                output_tensor_info,
            )

        # Only process operation nodes
        if not isinstance(node, tg.operation):
            print("not an op")
            return

        op_name = node.op_name

        # Get the node class for this operation
        node_class = self.NODE_CLASS_MAP.get(op_name)

        if node_class is None:
            print(f"Unimplemented Operation in Lowering to Tensor IR: {op_name}")
            return

        # For scaled matmul, we need to rewrite the tensor descriptor stride for the scaling factors to represent the
        # 128x4 interleave layout that meets the tensor-core instruction requirement.
        # Temporary limitation: this rewrite path only supports FP8 tensors with block_size == 32.
        # TODO: https://jirasw.nvidia.com/browse/CL-20137
        # Support the fp4 cases for pycudnnTest.py
        if op_name == "scaled_matmul":
            sfA_node, sfB_node = node.producer_nodes[1], node.producer_nodes[3]
            A_node = node.producer_nodes[0]
            B_node = node.producer_nodes[2]

            sfA_shape, sfA_stride = sfA_node.output[0].dim, sfA_node.output[0].stride
            sfB_shape, sfB_stride = sfB_node.output[0].dim, sfB_node.output[0].stride
            A_shape = A_node.output[0].dim
            B_shape = B_node.output[0].dim

            fp8_types = (DataType.FP8_E4M3, DataType.FP8_E5M2, DataType.FP8_E8M0)
            A_dtype = A_node.output[0].data_type
            B_dtype = B_node.output[0].data_type
            sfA_dtype = sfA_node.output[0].data_type
            sfB_dtype = sfB_node.output[0].data_type

            # Guard unsupported dtype combinations until non-FP8 lowering is added.
            if A_dtype not in fp8_types or B_dtype not in fp8_types or sfA_dtype not in fp8_types or sfB_dtype not in fp8_types:
                raise ValueError("scaled_matmul SF stride rewrite currently supports FP8-only tensors.")

            # Validate logical K-to-scale_K mapping from both A and B paths.
            if A_shape[2] % sfA_shape[2] != 0 or B_shape[1] % sfB_shape[1] != 0:
                raise ValueError("scaled_matmul requires K dimensions to be divisible by scale_K.")

            block_size = A_shape[2] // sfA_shape[2]
            block_size_b = B_shape[1] // sfB_shape[1]
            if block_size != block_size_b:
                raise ValueError(f"scaled_matmul expects consistent block_size from A/B, got {block_size} and {block_size_b}.")
            if block_size != 32:
                raise ValueError(f"scaled_matmul SF stride rewrite currently supports block_size == 32, got {block_size}.")

            # Rewrite the M/N stride of SF tensor descriptors to:
            #   (elements per 128x4 interleave block) * (number of K-blocks)
            # where:
            #   shape_K = scale_dim * block_size (each scale value covers block_size elements in K)
            #   number of K-blocks = ceil(shape_K / 128)
            # This stride depends on SF logical shape, not tile_size_K.
            self.node_overwrite_stride_func_map[sfA_node] = (
                lambda _tile_size: sfA_stride[:1] + [128 * 4 * math.ceil((sfA_shape[2] * block_size) / 128)] + sfA_stride[2:]
            )
            self.node_overwrite_stride_func_map[sfB_node] = lambda _tile_size: sfB_stride[:2] + [128 * 4 * math.ceil((sfB_shape[1] * block_size) / 128)]

        # Create and run the node
        ir_node = node_class(node, node_map, ip, self)
        ir_node.run()

    def get_reorder_mode(self, node):
        if hasattr(node, "get_reordering") and callable(node.get_reordering):
            if node.get_reordering() == "CUDNN_TENSOR_REORDERING_F8_128x4":
                return nv_tensor_ir.ReorderMode.f8_128x4
        return nv_tensor_ir.ReorderMode.none
