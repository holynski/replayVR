#ifndef REPLAY_TYPES_H_
#define REPLAY_TYPES_H_

#include <limits>

namespace replay {

typedef std::uint32_t VertexId;
typedef std::uint32_t TriangleId;
typedef std::uint32_t TriangleFaceId;
typedef std::uint32_t BicubicGridNodeId;
typedef std::uint32_t SegmentId;

static const VertexId kInvalidVertexId = std::numeric_limits<VertexId>::max();
static const TriangleFaceId kInvalidTriangleFaceId =
    std::numeric_limits<TriangleFaceId>::max();
static const TriangleId kInvalidTriangleId =
    std::numeric_limits<TriangleId>::max();
static const BicubicGridNodeId kInvalidBicubicGridNodeId =
    std::numeric_limits<BicubicGridNodeId>::max();
static const SegmentId kInvalidSegmentId =
    std::numeric_limits<SegmentId>::max();

}  // namespace replay

#endif  // REPLAY_TYPES_H_
