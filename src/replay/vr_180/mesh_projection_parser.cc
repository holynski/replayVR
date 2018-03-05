
#include <glog/logging.h>
#include <libavutil/spherical.h>
#include <math.h>
#include <replay/vr_180/mesh_projection_parser.h>
#include <replay/io/zlib_decompressor.h>
#include <replay/mesh/mesh.h>
#include <replay/io/byte_stream_reader.h>
#include <replay/io/stream_reader.h>
#include <algorithm>
#include <memory>
#include <unordered_map>

namespace replay {

namespace {

// TODO(holynski): This is a really hacky implementation. Refactor.
uint32_t DecodePackedInt(const uint8_t* stream, const size_t index,
                         const size_t num_bits_per_int) {
  size_t stream_index = (index * num_bits_per_int) / 8;
  int remaining_bits = num_bits_per_int;
  int start_offset = (num_bits_per_int * index) % 8;
  uint32_t output = 0;
  while (remaining_bits > 0) {
    int bits_to_read = std::min(8, remaining_bits);

    uint8_t bitmask = ~((1 << (8 - bits_to_read)) - 1);
    remaining_bits -= bits_to_read;

    if (start_offset > 0) {
      bitmask &= (1 << (8 - start_offset)) - 1;
      remaining_bits += start_offset;
      start_offset = 0;
    }
    const uint8_t byte_from_stream = stream[stream_index];
    output |=
        (static_cast<uint32_t>(byte_from_stream & bitmask) << remaining_bits) >>
        (8 - bits_to_read);
    stream_index++;
  }

  return output;
}

// TODO(holynski): Replace this with bit arithmetic.
int DecodeToSignedInt(const uint32_t input) {
  // int output1 = 0;
  int output2 = 0;
  // if (input == 13) {
  // LOG(INFO) << 13;
  //}
  // int x = -7;
  // output1 |= (input << 31);
  // output1 |= ((input + 1) >> 1);
  if (input % 2 == 0) {
    output2 = input / 2;
  } else {
    output2 = -((input + 1) / 2);
  }
  // LOG(INFO) << input;
  // CHECK_EQ(output1, output2);
  return output2;
}
}

// TODO(holynski): Clean up this function
Mesh MeshProjectionParser::ParseMesh() {
  Mesh mesh;
  uint32_t size = reader_->ReadUnsignedIntBE();  // size
  char* tag = reinterpret_cast<char*>(reader_->ReadData(4));
  CHECK_EQ(strncmp(tag, "mesh", 4), 0) << "Tag was not mesh!";
  uint32_t coordinate_count = reader_->ReadUnsignedIntBE();
  CHECK_GT(coordinate_count, 0);
  CHECK_EQ(coordinate_count & (1 << 31), 0);
  CHECK_LT(coordinate_count * sizeof(float), size);
  std::vector<float> coordinates(coordinate_count);

  for (int i = 0; i < coordinate_count; i++) {
    coordinates[i] = reader_->ReadFloatBE();
  }
  const int ccsb = ceil(log2(coordinate_count * 2));
  const uint32_t vertex_count = reader_->ReadUnsignedIntBE();
  CHECK_LT(vertex_count * 5 * (ceil(ccsb / 8.0)), size);
  CHECK_EQ((vertex_count) & (1 << 31), 0)
      << "First bit of vertex count should be reserved (0)";
  CHECK_GT(vertex_count, 0);
  uint8_t* xyzuv = reader_->ReadData(((ccsb * vertex_count * 5) / 8));
  CHECK_NOTNULL(xyzuv);
  uint32_t vertex_list_count = reader_->ReadUnsignedIntBE();
  CHECK_EQ((vertex_list_count) & (1 << 31), 0)
      << "First bit of coordinate count should be reserved (0)";
  int vcsb = ceil(log2(vertex_count * 2));

  std::vector<uint8_t> texture_ids(vertex_list_count); std::vector<uint8_t> index_types(vertex_list_count); std::vector<uint32_t> index_counts(vertex_list_count);
  std::vector<uint8_t*> index_deltas(vertex_list_count);

  for (int i = 0; i < vertex_list_count; i++) {
    CHECK_EQ(reader_->ReadByte(), 0);  // 0 means the texture is the video frame
    index_types[i] = reader_->ReadByte();
    CHECK_EQ(index_types[i], 1);  // We only support triangle strips for now.
    index_counts[i] = reader_->ReadUnsignedIntBE();
    CHECK_EQ((index_counts[i]) & (1 << 31), 0)
        << "First bit of index_count should be reserved (0)";
    index_deltas[i] = reader_->ReadData(((vcsb * index_counts[i]) / 8.0));
  }

  std::unordered_map<int, VertexId> mapping;
  int x_index = 0;
  int y_index = 0;
  int z_index = 0;
  int u_index = 0;
  int v_index = 0;
  for (int i = 0; i < vertex_count; i++) {
    uint32_t x_index_delta = DecodePackedInt(xyzuv, i * 5, ccsb);
    uint32_t y_index_delta = DecodePackedInt(xyzuv, i * 5 + 1, ccsb);
    uint32_t z_index_delta = DecodePackedInt(xyzuv, i * 5 + 2, ccsb);
    uint32_t u_index_delta = DecodePackedInt(xyzuv, i * 5 + 3, ccsb);
    uint32_t v_index_delta = DecodePackedInt(xyzuv, i * 5 + 4, ccsb);
    x_index += DecodeToSignedInt(x_index_delta);
    y_index += DecodeToSignedInt(y_index_delta);
    z_index += DecodeToSignedInt(z_index_delta);
    u_index += DecodeToSignedInt(u_index_delta);
    v_index += DecodeToSignedInt(v_index_delta);
    CHECK_GE(x_index, 0);
    CHECK_GE(y_index, 0);
    CHECK_GE(z_index, 0);
    CHECK_GE(u_index, 0);
    CHECK_GE(v_index, 0);
    CHECK_LT(x_index, coordinate_count);
    CHECK_LT(y_index, coordinate_count);
    CHECK_LT(z_index, coordinate_count);
    CHECK_LT(u_index, coordinate_count);
    CHECK_LT(v_index, coordinate_count);
    float x = coordinates[x_index];
    float y = coordinates[y_index];
    float z = coordinates[z_index];
    float u = coordinates[u_index];
    float v = coordinates[v_index];
    mapping[i] = mesh.AddVertex(Eigen::Vector3f(x, y, z));
    mesh.SetVertexUV(mapping[i], u, v);
  }

  for (int i = 0; i < vertex_list_count; i++) {
    const uint32_t index_count = index_counts[i];
    uint32_t index = 0;
    switch (index_types[i]) {
      case 0:
        LOG(FATAL) << "Mesh index type TRIANGLES not implemented.";
        break;
      case 1: {
        std::vector<uint32_t> indices;
        for (int v = 0; v < index_count; v++) {
          index += DecodeToSignedInt(DecodePackedInt(index_deltas[i], v, vcsb));
          indices.emplace_back(index);
          if (indices.size() == 3) {
            if (v % 2 == 0) {
              mesh.AddTriangleFace(mapping[indices[0]], mapping[indices[1]],
                                   mapping[indices[2]]);
            } else {
              mesh.AddTriangleFace(mapping[indices[0]], mapping[indices[2]],
                                   mapping[indices[1]]);
            }
            indices.erase(indices.begin());
          }
        }
      } break;
      case 2:
        LOG(FATAL) << "Mesh index type TRIANGLE FAN not implemented.";
        break;
    }
  }

  return mesh;
}

std::vector<Mesh> MeshProjectionParser::Parse(
    const AVSphericalMesh& metadata, const uint32_t encoding) {
  switch (encoding) {
    case 0x64666c38:
      {
        std::unique_ptr<ZlibDecompressor> decompressor = std::make_unique<ZlibDecompressor>(false);
        decompressor->Initialize(metadata.data, metadata.data_size);
        reader_ = std::unique_ptr<StreamReader>(std::move(decompressor));
      }
      break;
    case 0x72617720:
      {
      reader_ = std::make_unique<ByteStreamReader>(metadata.data, metadata.data_size);
      }
      break;
    default:
      LOG(FATAL) << "Unsupported encoding type " << encoding;
  }
  std::vector<Mesh> meshes;
  while (!reader_->EndOfStream()) {
  meshes.emplace_back(ParseMesh());
  }
  LOG(INFO) << "Found " << meshes.size() << " meshes!";
  return meshes;
}
}
