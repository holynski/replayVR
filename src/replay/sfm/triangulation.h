#pragma once 

namespace replay {

  // Triangulates a single point using two cameras.
  // In the TrackedPoint struct, the point depth for the two cameras will be asigned to be that of the triangulated point. In addition, 
  Eigen::Vector3d TriangulatePoint(const Camera& camera1, const Camera& camera2,  TrackedPoint* point);

  // Same as the function above, but uses any number of cameras (> 1).
  Eigen::Vector3d TriangulatePoint(const std::vector<const Camera>& cameras, TrackedPoint* point);

}
