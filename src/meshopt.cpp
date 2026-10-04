// meshoptimizer's C functions, each called once on whole arrays with Python's lock let go. The checks that keep a call
// in bounds are here; what makes a call sensible, as the JavaScript module asserts it, is in python/meshopt.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "../vendor/meshoptimizer/meshoptimizer.h"

namespace nb = nanobind;

template <typename T> using In = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <typename T> using Out = nb::ndarray<nb::numpy, T, nb::ndim<1>>;

template <typename T> Out<T> out(std::vector<T>&& values)
{
	auto* held = new std::vector<T>(std::move(values));
	nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
	return Out<T>(held->data(), {held->size()}, owner);
}

static void sized(const char* name, size_t length, size_t count, size_t size)
{
	if (size != 0 && count > SIZE_MAX / size)
		throw nb::value_error("too many values");
	if (count * size != length)
		throw nb::value_error((std::string(name) + " holds " + std::to_string(length) + " values, not " + std::to_string(count) + " by " + std::to_string(size)).c_str());
}

static size_t vertices(const unsigned int* indices, size_t count)
{
	return count ? size_t(*std::max_element(indices, indices + count)) + 1 : 0;
}

// Triangles reordered for a GPU's vertex cache, as a strip likes them when strip is set.
static Out<uint32_t> optimize_vertex_cache(In<uint32_t> indices, bool strip)
{
	std::vector<uint32_t> result(indices.size());
	{
		nb::gil_scoped_release release;
		size_t count = vertices(indices.data(), indices.size());
		if (strip)
			meshopt_optimizeVertexCacheStrip(result.data(), indices.data(), indices.size(), count);
		else
			meshopt_optimizeVertexCache(result.data(), indices.data(), indices.size(), count);
	}
	return out(std::move(result));
}

// Each vertex's new place, in the order the indices first use them, and how many are used.
static std::pair<Out<uint32_t>, size_t> optimize_vertex_fetch_remap(In<uint32_t> indices)
{
	std::vector<uint32_t> remap;
	size_t unique;
	{
		nb::gil_scoped_release release;
		remap.resize(vertices(indices.data(), indices.size()));
		unique = meshopt_optimizeVertexFetchRemap(remap.data(), indices.data(), indices.size(), remap.size());
	}
	return {out(std::move(remap)), unique};
}

// The points' order along a space-filling curve.
static Out<uint32_t> spatial_sort_remap(In<float> positions, size_t count, size_t stride)
{
	sized("positions", positions.size(), count, stride);
	if (stride < 3)
		throw nb::value_error("a point has three coordinates at least");
	std::vector<uint32_t> remap(count);
	{
		nb::gil_scoped_release release;
		meshopt_spatialSortRemap(remap.data(), positions.data(), count, stride * sizeof(float));
	}
	return out(std::move(remap));
}

static nb::bytes encode_vertex_buffer(In<uint8_t> data, size_t count, size_t size, int level, int version)
{
	sized("data", data.size(), count, size);
	std::vector<unsigned char> buffer;
	{
		nb::gil_scoped_release release;
		buffer.resize(meshopt_encodeVertexBufferBound(count, size));
		buffer.resize(meshopt_encodeVertexBufferLevel(buffer.data(), buffer.size(), data.data(), count, size, level, version));
	}
	return nb::bytes(buffer.data(), buffer.size());
}

template <size_t (*Bound)(size_t, size_t), size_t (*Encode)(unsigned char*, size_t, const unsigned int*, size_t)>
static nb::bytes encode_indices(In<uint32_t> indices)
{
	std::vector<unsigned char> buffer;
	{
		nb::gil_scoped_release release;
		buffer.resize(Bound(indices.size(), vertices(indices.data(), indices.size())));
		buffer.resize(Encode(buffer.data(), buffer.size(), indices.data(), indices.size()));
	}
	return nb::bytes(buffer.data(), buffer.size());
}

// Floats in a filter's form: count elements of stride bytes, each read from four floats, or stride / 4 for exponents.
static Out<uint8_t> encode_filter(const std::string& filter, In<float> data, size_t count, size_t stride, int bits, int mode)
{
	sized("data", data.size(), count, filter == "EXPONENTIAL" ? stride / 4 : 4);
	std::vector<uint8_t> result(count * stride);
	{
		nb::gil_scoped_release release;
		if (filter == "OCTAHEDRAL")
			meshopt_encodeFilterOct(result.data(), count, stride, bits, data.data());
		else if (filter == "QUATERNION")
			meshopt_encodeFilterQuat(result.data(), count, stride, bits, data.data());
		else if (filter == "EXPONENTIAL")
			meshopt_encodeFilterExp(result.data(), count, stride, bits, data.data(), meshopt_EncodeExpMode(mode));
		else
			meshopt_encodeFilterColor(result.data(), count, stride, bits, data.data());
	}
	return out(std::move(result));
}

template <int (*Decode)(void*, size_t, size_t, const unsigned char*, size_t)>
static Out<uint8_t> decode(In<uint8_t> source, size_t count, size_t size, std::optional<std::string> filter)
{
	if (size != 0 && count > SIZE_MAX / size)
		throw nb::value_error("too many values");
	std::vector<uint8_t> result(count * size);
	int status;
	{
		nb::gil_scoped_release release;
		status = Decode(result.data(), count, size, source.data(), source.size());
		if (status == 0 && filter == "OCTAHEDRAL")
			meshopt_decodeFilterOct(result.data(), count, size);
		else if (status == 0 && filter == "QUATERNION")
			meshopt_decodeFilterQuat(result.data(), count, size);
		else if (status == 0 && filter == "EXPONENTIAL")
			meshopt_decodeFilterExp(result.data(), count, size);
		else if (status == 0 && filter == "COLOR")
			meshopt_decodeFilterColor(result.data(), count, size);
	}
	if (status != 0)
		throw nb::value_error(("Malformed buffer data: " + std::to_string(status)).c_str());
	return out(std::move(result));
}

NB_MODULE(_core, m)
{
	m.def("optimize_vertex_cache", &optimize_vertex_cache);
	m.def("optimize_vertex_fetch_remap", &optimize_vertex_fetch_remap);
	m.def("spatial_sort_remap", &spatial_sort_remap);
	m.def("encode_vertex_buffer", &encode_vertex_buffer);
	m.def("encode_index_buffer", &encode_indices<meshopt_encodeIndexBufferBound, meshopt_encodeIndexBuffer>);
	m.def("encode_index_sequence", &encode_indices<meshopt_encodeIndexSequenceBound, meshopt_encodeIndexSequence>);
	m.def("encode_filter", &encode_filter);
	m.def("decode_vertex_buffer", &decode<meshopt_decodeVertexBuffer>, nb::arg("source"), nb::arg("count"), nb::arg("size"), nb::arg("filter") = nb::none());
	m.def("decode_index_buffer", [](In<uint8_t> source, size_t count, size_t size) { return decode<meshopt_decodeIndexBuffer>(source, count, size, std::nullopt); });
	m.def("decode_index_sequence", [](In<uint8_t> source, size_t count, size_t size) { return decode<meshopt_decodeIndexSequence>(source, count, size, std::nullopt); });
}
