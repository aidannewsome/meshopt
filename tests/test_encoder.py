"""js/meshopt_encoder.test.js from meshoptimizer 1.3, test for test, in the same order and with the same numbers."""

import numpy as np

import meshopt as decoder
import meshopt as encoder


def as_bytes(data: np.ndarray) -> np.ndarray:
    return data.view(np.uint8).reshape(-1)


def test_reorder_mesh():
    indices = np.array([4, 2, 5, 3, 1, 4, 0, 1, 3, 1, 2, 4], np.uint32)

    expected = np.array([0, 1, 2, 3, 1, 0, 4, 3, 0, 5, 3, 4], np.uint32)

    remap = np.array([5, 3, 1, 4, 0, 2], np.uint32)

    reordered, res_remap, unique = encoder.reorder_mesh(indices, triangles=True, optsize=True)
    assert np.array_equal(reordered, expected)
    assert np.array_equal(res_remap, remap)
    assert unique == 6


def test_reorder_points():
    points = np.array([1, 1, 1, 11, 11, 11, 2, 2, 2, 12, 12, 12], np.float32).reshape(-1, 3)

    expected = np.array([0, 2, 1, 3], np.uint32)

    remap = encoder.reorder_points(points)
    assert np.array_equal(remap, expected)


def groups() -> np.ndarray:
    data = np.zeros((16, 4), np.uint8)

    # this tests 0/2/4/8 bit groups in one stream
    for i in range(16):
        data[i] = [0, i * 1, i * 2, i * 8]
    return data


def test_roundtrip_vertex_buffer():
    data = groups()

    encoded = encoder.encode_vertex_buffer(data)
    assert encoded[0] == 0xA1

    decoded = decoder.decode_vertex_buffer(encoded, 16, 4)

    assert np.array_equal(decoded, data)


def test_roundtrip_vertex_buffer_v1():
    data = groups()

    encoded = encoder.encode_vertex_buffer_level(data, 3, version=1)
    assert encoded[0] == 0xA1

    decoded = decoder.decode_vertex_buffer(encoded, 16, 4)

    assert np.array_equal(decoded, data)


def test_roundtrip_index_buffer():
    data = np.array([0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9], np.uint32)

    encoded = encoder.encode_index_buffer(data)
    decoded = decoder.decode_index_buffer(encoded, len(data), 4)

    assert np.array_equal(decoded, data)


def test_roundtrip_index_buffer16():
    data = np.array([0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9], np.uint16)

    encoded = encoder.encode_index_buffer(data)
    decoded = decoder.decode_index_buffer(encoded, len(data), 2)

    assert np.array_equal(decoded, data)


def test_roundtrip_index_sequence():
    data = np.array([0, 1, 51, 2, 49, 1000], np.uint32)

    encoded = encoder.encode_index_sequence(data)
    decoded = decoder.decode_index_sequence(encoded, len(data), 4)

    assert np.array_equal(decoded, data)


def test_roundtrip_index_sequence16():
    data = np.array([0, 1, 51, 2, 49, 1000], np.uint16)

    encoded = encoder.encode_index_sequence(data)
    decoded = decoder.decode_index_sequence(encoded, len(data), 2)

    assert np.array_equal(decoded, data)


VECTORS = np.array([1, 0, 0, 0, 0, -1, 0, 0, 0.7071068, 0, 0.707168, 1, -0.7071068, 0, -0.707168, 1], np.float32).reshape(-1, 4)


def test_encode_filter_oct8():
    expected = np.array([0x7F, 0, 0x7F, 0, 0, 0x81, 0x7F, 0, 0x3F, 0, 0x7F, 0x7F, 0x81, 0x40, 0x7F, 0x7F], np.uint8)

    # 4 vectors, encode each vector into 4 bytes with 8 bits of precision/component
    encoded = encoder.encode_filter_oct(VECTORS, 4, 8)
    assert np.array_equal(as_bytes(encoded), expected)


def test_encode_filter_oct12():
    expected = np.array([0x7FF, 0, 0x7FF, 0, 0x0, 0xF801, 0x7FF, 0, 0x3FF, 0, 0x7FF, 0x7FFF, 0xF801, 0x400, 0x7FF, 0x7FFF], np.uint16)

    # 4 vectors, encode each vector into 8 bytes with 12 bits of precision/component
    encoded = encoder.encode_filter_oct(VECTORS, 8, 12)
    assert np.array_equal(as_bytes(encoded), as_bytes(expected))


def test_encode_filter_quat12():
    data = np.array([1, 0, 0, 0, 0, -1, 0, 0, 0.7071068, 0, 0, 0.707168, -0.7071068, 0, 0, -0.707168], np.float32).reshape(-1, 4)

    expected = np.array([0, 0, 0, 0x7FC, 0, 0, 0, 0x7FD, 0x7FF, 0, 0, 0x7FF, 0x7FF, 0, 0, 0x7FF], np.uint16)

    # 4 quaternions, encode each quaternion into 8 bytes with 12 bits of precision/component
    encoded = encoder.encode_filter_quat(data, 8, 12)
    assert np.array_equal(as_bytes(encoded), as_bytes(expected))


def test_encode_filter_exp():
    data = np.array([[1, -23.4, -0.1]], np.float32)

    expected = np.array([0xF7000200, 0xF7FFD133, 0xF7FFFFCD], np.uint32)

    # 1 vector with 3 components (12 bytes), encode each vector into 12 bytes with 15 bits of precision/component
    encoded = encoder.encode_filter_exp(data, 15)
    assert np.array_equal(as_bytes(encoded), as_bytes(expected))


def test_encode_filter_exp_mode():
    data = np.array([[1, -23.4], [-0.1, 11.0]], np.float32)

    expected = np.array([0xF3002000, 0xF7FFD133, 0xF3FFFCCD, 0xF7001600], np.uint32)

    # 2 vectors with 2 components (8 bytes), encode each vector into 8 bytes with 15 bits of precision/component
    encoded = encoder.encode_filter_exp(data, 15, "SharedComponent")
    assert np.array_equal(as_bytes(encoded), as_bytes(expected))


def test_encode_filter_exp_clamp():
    data = np.array([[1, -23.4, -0.1]], np.float32)

    expected = np.array([0xF3002000, 0xF7FFD133, 0xF2FFF99A], np.uint32)

    # 1 vector with 3 components (12 bytes), encode each vector into 12 bytes with 15 bits of precision/component
    # exponents are separate but clamped to 0
    encoded = encoder.encode_filter_exp(data, 15, "Clamped")
    assert np.array_equal(as_bytes(encoded), as_bytes(expected))


COLOURS = np.array([1, 0, 0, 1, 0, 1, 0, 0.5, 0, 0, 1, 0.25, 0.4, 0.4, 0.4, 0.75], np.float32).reshape(-1, 4)


def test_encode_filter_color8():
    expected = np.array([0x40, 0x7F, 0xC1, 0xFF, 0x7F, 0x00, 0x7F, 0xC0, 0x40, 0x81, 0xC0, 0xA0, 0x66, 0x00, 0x00, 0xDF], np.uint8)

    # 4 vectors, encode each vector into 4 bytes with 8 bits of precision/component
    encoded = encoder.encode_filter_color(COLOURS, 4, 8)
    assert np.array_equal(as_bytes(encoded), expected)


def test_encode_filter_color12():
    expected = np.array(
        [0x0400, 0x07FF, 0xFC01, 0x0FFF, 0x07FF, 0x0000, 0x07FF, 0x0C00, 0x0400, 0xF801, 0xFC00, 0x0A00, 0x0666, 0x0000, 0x0000, 0x0DFF],
        np.uint16,
    )

    # 4 vectors, encode each vector into 8 bytes with 12 bits of precision/component
    encoded = encoder.encode_filter_color(COLOURS, 8, 12)
    assert np.array_equal(as_bytes(encoded), as_bytes(expected))


TRIANGLES = np.array([0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9], np.uint32)


def test_encode_gltf_buffer():
    encoded = encoder.encode_gltf_buffer(TRIANGLES, "TRIANGLES")
    decoded = decoder.decode_gltf_buffer(encoded, len(TRIANGLES), 4, "TRIANGLES")

    assert encoded[0] == 0xE1
    assert np.array_equal(decoded, TRIANGLES)


def test_encode_gltf_buffer_attribute():
    encoded = encoder.encode_gltf_buffer(TRIANGLES[:, None], "ATTRIBUTES")
    decoded = decoder.decode_gltf_buffer(encoded, len(TRIANGLES), 4, "ATTRIBUTES")

    assert encoded[0] == 0xA0
    assert np.array_equal(decoded.view(np.uint32).ravel(), TRIANGLES)


def test_encode_gltf_buffer_attribute_v1():
    encoded = encoder.encode_gltf_buffer(TRIANGLES[:, None], "ATTRIBUTES", 1)
    decoded = decoder.decode_gltf_buffer(encoded, len(TRIANGLES), 4, "ATTRIBUTES")

    assert encoded[0] == 0xA1
    assert np.array_equal(decoded.view(np.uint32).ravel(), TRIANGLES)
