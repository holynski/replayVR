#ifndef REPLAY_ROW_ARRAY_H_
#define REPLAY_ROW_ARRAY_H_

#include <Eigen/Core>

namespace replay {

// Convenience typedefs for dynamic row array types.

typedef Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowArrayXXf;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowArrayXXd;
typedef Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowArrayXXb;
typedef Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    RowArrayXXi;

}  // namespace replay

#endif  // REPLAY_ROW_ARRAY_H_
