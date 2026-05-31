from enum import Enum


class DataType(Enum):
    FLOAT = "float32"
    DOUBLE = "float64"
    INT8 = "int8"
    INT32 = "int32"
    INT64 = "int64"
    HALF = "float16"
    BFLOAT16 = "bfloat16"
    BOOL = "boolean"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP8_E8M0 = "fp8_e8m0"


pytorch_mapping = {
    DataType.FLOAT: "torch.float32",
    DataType.DOUBLE: "torch.double",
    DataType.INT8: "torch.int8",
    DataType.INT32: "torch.int32",
    DataType.INT64: "torch.int64",
    DataType.HALF: "torch.float16",
    DataType.BFLOAT16: "torch.bfloat16",
    DataType.BOOL: "torch.bool",
    DataType.FP8_E4M3: "torch.float8_e4m3fn",
    DataType.FP8_E5M2: "torch.float8_e5m2",
    DataType.FP8_E8M0: "torch.uint8",
}

cudnn_mapping = {
    DataType.FLOAT: "cudnn.data_type.FLOAT",
    DataType.DOUBLE: "cudnn_data_type.DOUBLE",
    DataType.INT8: "cudnn.data_type.INT8",
    DataType.INT32: "cudnn.data_type.INT32",
    DataType.INT64: "cudnn.data_type.INT64",
    DataType.HALF: "cudnn.data_type.HALF",
    DataType.BFLOAT16: "cudnn.data_type.BFLOAT16",
    DataType.BOOL: "cudnn.data_type.BOOLEAN",
    DataType.FP8_E4M3: "cudnn.data_type.FP8_E4M3",
    DataType.FP8_E5M2: "cudnn.data_type.FP8_E5M2",
    DataType.FP8_E8M0: "cudnn.data_type.FP8_E8M0",
}

tensorir_mapping = {
    DataType.FLOAT: "T.f32()",
    DataType.DOUBLE: "T.f64()",
    DataType.INT8: "T.si8()",
    DataType.INT32: "T.si32()",
    DataType.INT64: "T.si64()",
    DataType.HALF: "T.f16()",
    DataType.BFLOAT16: "T.bf16()",
    DataType.BOOL: "T.bool()",
    DataType.FP8_E4M3: "T.f8E4M3FN()",
    DataType.FP8_E5M2: "T.f8E5M2()",
    DataType.FP8_E8M0: "T.f8E8M0FNU()",
}

cask_mapping = {
    DataType.FLOAT: "nv_tensor_ir.NumericTypeID.kF32",
    DataType.DOUBLE: "nv_tensor_ir.NumericTypeID.kF64",
    DataType.INT8: "nv_tensor_ir.NumericTypeID.kS8",
    DataType.INT32: "nv_tensor_ir.NumericTypeID.kS32",
    DataType.INT64: "nv_tensor_ir.NumericTypeID.kS64",
    DataType.HALF: "nv_tensor_ir.NumericTypeID.kF16",
    DataType.BFLOAT16: "nv_tensor_ir.NumericTypeID.kBF16",
    DataType.BOOL: "nv_tensor_ir.NumericTypeID.kB1",
    DataType.FP8_E4M3: "nv_tensor_ir.NumericTypeID.kE4M3",
    DataType.FP8_E5M2: "nv_tensor_ir.NumericTypeID.kE5M2",
    DataType.FP8_E8M0: "nv_tensor_ir.NumericTypeID.kUE8M0",
}


type_mapping = {
    "torch": pytorch_mapping,
    "cudnn": cudnn_mapping,
    "tensorir": tensorir_mapping,
    "cask": cask_mapping,
}


def convert_datatype(data_type, lib):
    choices = list(type_mapping.keys())
    if lib not in choices:
        raise KeyError(f"Invalid library specified: {lib}. Choose from {choices}.")
    try:
        return type_mapping[lib][data_type]
    except KeyError:
        raise KeyError(f"Data type {data_type} not found. Choose from {type_mapping[lib].keys()}.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")


def convert_to_torch_type(cudnn_type):
    import cudnn

    cudnn_to_torch_mapping = {
        cudnn.data_type.FLOAT: "torch.float32",
        cudnn.data_type.INT8: "torch.int8",
        cudnn.data_type.INT32: "torch.int32",
        cudnn.data_type.HALF: "torch.float16",
        cudnn.data_type.BFLOAT16: "torch.bfloat16",
        cudnn.data_type.BOOLEAN: "torch.bool",
    }
    try:
        return cudnn_to_torch_mapping[cudnn_type]
    except KeyError:
        raise ValueError("Unsupported tensor data type.", cudnn_type)


def convert_to_torch_type_wrapper(dtype):
    if isinstance(dtype, DataType):
        return convert_datatype(dtype, "torch")
    else:
        return convert_to_torch_type(dtype)
