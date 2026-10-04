"""meshoptimizer's encoder and decoder, shaped like its JavaScript module: the same functions in the same order,
counts and sizes read from the arrays where they can be. Codecs give bytes; filters and decoders give arrays."""

from typing import Literal

import numpy as np

from . import _core

__all__ = [
    "reorder_mesh",
    "reorder_points",
    "encode_vertex_buffer",
    "encode_vertex_buffer_level",
    "encode_index_buffer",
    "encode_index_sequence",
    "encode_gltf_buffer",
    "encode_filter_oct",
    "encode_filter_quat",
    "encode_filter_exp",
    "encode_filter_color",
    "decode_vertex_buffer",
    "decode_index_buffer",
    "decode_index_sequence",
    "decode_gltf_buffer",
]

Mode = Literal["ATTRIBUTES", "TRIANGLES", "INDICES"]
Filter = Literal["NONE", "OCTAHEDRAL", "QUATERNION", "EXPONENTIAL", "COLOR"]
ExpMode = Literal["Separate", "SharedVector", "SharedComponent", "Clamped"]
EXP_MODES = {"Separate": 0, "SharedVector": 1, "SharedComponent": 2, "Clamped": 3}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rows(data: np.ndarray) -> tuple[np.ndarray, int, int]:
    """An array as its bytes, with a row each element: the bytes, the count and each row's size."""
    data = np.ascontiguousarray(data)
    check(data.ndim >= 1 and len(data) > 0, "data must have a row for each element")
    return data.view(np.uint8).reshape(-1), len(data), data.nbytes // len(data)


def index32(indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices)
    check(indices.dtype in (np.uint16, np.uint32, np.int32), "indices must be 16 or 32 bit integers")
    return np.ascontiguousarray(indices.reshape(-1), dtype=np.uint32)


def reorder_mesh(indices: np.ndarray, triangles: bool, optsize: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Indices renumbered so vertices come in the order they are first used, the triangles first put in the order a
    GPU's vertex cache likes, or a strip's when optsize. The new indices, each vertex's new place, and how many are used."""
    check(np.asarray(indices).dtype in (np.uint32, np.int32), "indices must be 32 bit integers")
    indices = index32(indices)
    check(not triangles or len(indices) % 3 == 0, "triangles need indices in threes")
    if triangles:
        indices = _core.optimize_vertex_cache(indices, optsize)
    remap, unique = _core.optimize_vertex_fetch_remap(indices)
    return remap[indices], remap, unique


def reorder_points(positions: np.ndarray) -> np.ndarray:
    """The order to put points in so near ones are near in the list: each place's point."""
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    check(positions.ndim == 2 and positions.shape[1] >= 3, "positions must be n by 3 or more")
    return _core.spatial_sort_remap(positions.reshape(-1), *positions.shape)


def encode_vertex_buffer(data: np.ndarray) -> bytes:
    return encode_vertex_buffer_level(data, 2)


def encode_vertex_buffer_level(data: np.ndarray, level: int, version: int | None = None) -> bytes:
    """Rows of any type, each a multiple of four bytes, compressed harder from level 0 to 3."""
    data, count, size = rows(data)
    check(0 < size <= 256 and size % 4 == 0, "a row must be 4 to 256 bytes, in fours")
    check(0 <= level <= 3, "level must be 0 to 3")
    check(version in (None, 0, 1), "version must be 0 or 1")
    return _core.encode_vertex_buffer(data, count, size, level, -1 if version is None else version)


def encode_index_buffer(indices: np.ndarray) -> bytes:
    indices = index32(indices)
    check(len(indices) % 3 == 0, "triangles need indices in threes")
    return _core.encode_index_buffer(indices)


def encode_index_sequence(indices: np.ndarray) -> bytes:
    return _core.encode_index_sequence(index32(indices))


def encode_gltf_buffer(data: np.ndarray, mode: Mode, version: int = 0) -> bytes:
    """Data as EXT_meshopt_compression stores it, for a bufferView of the mode."""
    check(mode in ("ATTRIBUTES", "TRIANGLES", "INDICES"), "mode must be ATTRIBUTES, TRIANGLES or INDICES")
    if mode == "ATTRIBUTES":
        return encode_vertex_buffer_level(data, 2, version)
    return encode_index_buffer(data) if mode == "TRIANGLES" else encode_index_sequence(data)


def encode_filter(filter: str, values: np.ndarray, stride: int, bits: int, mode: int = 0, width: int = 4) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    check(values.ndim == 2 and values.shape[1] == width, f"values must be n by {width}")
    return _core.encode_filter(filter, values.reshape(-1), len(values), stride, bits, mode).reshape(-1, stride)


def encode_filter_oct(values: np.ndarray, stride: int, bits: int) -> np.ndarray:
    """Unit vectors, n by 4 with w kept as a sign, in octahedral form: n rows of stride bytes."""
    check(stride in (4, 8), "stride must be 4 or 8")
    check(2 <= bits <= 16, "bits must be 2 to 16")
    return encode_filter("OCTAHEDRAL", values, stride, bits)


def encode_filter_quat(values: np.ndarray, stride: int, bits: int) -> np.ndarray:
    """Unit quaternions, n by 4, in their smallest three components: n rows of 8 bytes."""
    check(stride == 8, "stride must be 8")
    check(4 <= bits <= 16, "bits must be 4 to 16")
    return encode_filter("QUATERNION", values, stride, bits)


def encode_filter_exp(values: np.ndarray, bits: int, mode: ExpMode = "SharedVector") -> np.ndarray:
    """Floats, n by k, as a mantissa of the bits and an exponent each, shared as the mode says: n rows of 4k bytes."""
    check(mode in EXP_MODES, "mode must be Separate, SharedVector, SharedComponent or Clamped")
    check(1 <= bits <= 24, "bits must be 1 to 24")
    values = np.asarray(values)
    check(values.ndim == 2 and values.shape[1] > 0, "values must be n by k")
    return encode_filter("EXPONENTIAL", values, 4 * values.shape[1], bits, EXP_MODES[mode], values.shape[1])


def encode_filter_color(values: np.ndarray, stride: int, bits: int) -> np.ndarray:
    """Colours, n by 4 from 0 to 1, as YCoCg with alpha: n rows of stride bytes."""
    check(stride in (4, 8), "stride must be 4 or 8")
    check(2 <= bits <= 16, "bits must be 2 to 16")
    return encode_filter("COLOR", values, stride, bits)


def source(data: bytes | np.ndarray) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else np.ascontiguousarray(data, dtype=np.uint8).reshape(-1)


def decode_vertex_buffer(data: bytes, count: int, size: int, filter: Filter | None = None) -> np.ndarray:
    """Encoded rows back as count rows of size bytes, the filter undone if one is named."""
    check(0 < size <= 256 and size % 4 == 0, "a row must be 4 to 256 bytes, in fours")
    check(filter in (None, "NONE", "OCTAHEDRAL", "QUATERNION", "EXPONENTIAL", "COLOR"), "filter must be NONE, OCTAHEDRAL, QUATERNION, EXPONENTIAL or COLOR")
    check(filter not in ("OCTAHEDRAL", "COLOR") or size in (4, 8), "this filter's rows are 4 or 8 bytes")
    check(filter != "QUATERNION" or size == 8, "quaternion rows are 8 bytes")
    return _core.decode_vertex_buffer(source(data), count, size, filter).reshape(count, size)


def decode_index_buffer(data: bytes, count: int, size: int) -> np.ndarray:
    """Encoded triangles back as count indices of size bytes, 2 or 4."""
    check(size in (2, 4), "size must be 2 or 4")
    check(count % 3 == 0, "triangles need indices in threes")
    return _core.decode_index_buffer(source(data), count, size).view(np.uint16 if size == 2 else np.uint32)


def decode_index_sequence(data: bytes, count: int, size: int) -> np.ndarray:
    check(size in (2, 4), "size must be 2 or 4")
    return _core.decode_index_sequence(source(data), count, size).view(np.uint16 if size == 2 else np.uint32)


def decode_gltf_buffer(data: bytes, count: int, size: int, mode: Mode, filter: Filter | None = None) -> np.ndarray:
    """An EXT_meshopt_compression bufferView back as plain data: rows of bytes, or indices."""
    check(mode in ("ATTRIBUTES", "TRIANGLES", "INDICES"), "mode must be ATTRIBUTES, TRIANGLES or INDICES")
    if mode == "ATTRIBUTES":
        return decode_vertex_buffer(data, count, size, filter)
    return decode_index_buffer(data, count, size) if mode == "TRIANGLES" else decode_index_sequence(data, count, size)
