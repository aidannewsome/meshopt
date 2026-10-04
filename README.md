# meshopt

An independent Python binding for [meshoptimizer](https://github.com/zeux/meshoptimizer)'s encoder and decoder, modelled
on its JavaScript module. Not affiliated with the meshoptimizer project.

## Use

```python
import meshopt

indices, remap, unique = meshopt.reorder_mesh(indices, triangles=True, optsize=False)
vertices = vertices[np.argsort(remap)[:unique]]                  # each vertex moved to its new place

data = meshopt.encode_gltf_buffer(vertices, "ATTRIBUTES")        # bytes for an EXT_meshopt_compression bufferView
rows = meshopt.decode_gltf_buffer(data, len(vertices), vertices.itemsize * vertices.shape[1], "ATTRIBUTES")
```

The functions are the JavaScript module's (`MeshoptEncoder` and `MeshoptDecoder`), in snake case, with counts and sizes
taken from the arrays where they can be. Encoders give bytes; filters give rows of bytes as an array; decoders give rows
of bytes, or indices for an index buffer. The tests are the JavaScript module's tests, one for one.

Every call releases Python's lock, and the module runs on free-threaded Python with the lock off.

## Build

A short C++ binding made with nanobind over meshoptimizer's own sources, vendored from release 1.3 in `vendor/meshoptimizer`.
`uv sync` builds it and `uv run pytest` tests it. Releases are a wheel for each system, so installing needs no compiler.

## Licence

MIT, as meshoptimizer is. meshoptimizer is Arseny Kapoulkine's; its licence is in `vendor/meshoptimizer/LICENSE.md`.
