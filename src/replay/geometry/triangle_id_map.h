#ifndef REPLAY_TRIANGLE_ID_MAP_H_
#define REPLAY_TRIANGLE_ID_MAP_H_

#include <Eigen/Core>
#include "replay/util/types.h"
namespace replay {

// Convenience typedefs for a triangle id map, which is a RowArray of triangle
// ids.
typedef Eigen::Array<TriangleFaceId,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::RowMajor> TriangleIdMap;

}  // namespace replay

#endif  // REPLAY_TRIANGLE_ID_MAP_H_
