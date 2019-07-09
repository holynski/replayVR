#pragma once

#include "replay/camera/camera.h"
#include "replay/sfm/tracked_point.h"

#include <glog/logging.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace replay {

struct VideoTrackerOptions {
  int max_points = 600;
  int min_points = 400;
  int window_size = 31;
  int distance_between_keypoints = 50;

  // If greater than zero, will limit the active points to being within a
  // certain number of degrees from the optical axis.
  double max_angle_from_optical_axis = -1;
  VideoTrackerOptions() = default;
};

cv::Mat3b VisualizeMatches(
    const cv::Mat3b& image1, const std::vector<Eigen::Vector2d>& points1,
    const cv::Mat3b& image2, const std::vector<Eigen::Vector2d>& points2,
    const int max_dimension = -1,
    const std::vector<cv::Scalar>& colors = std::vector<cv::Scalar>());

class VideoTracker {
 public:
  VideoTracker(const VideoTrackerOptions options = VideoTrackerOptions());

  // Tracks the previous frame's points into the next frame, and removed the
  // active points which aren't found. If the number of tracked points falls
  // below the 'min_points' parameter defined in the VideoTrackerOptions struct,
  // new points will be found and added.
  //
  // This function will replace the current frame with the camera/image
  // provided.
  const std::vector<TrackedPoint*>& TrackFrame(const cv::Mat3b& frame,
                                               const Camera* camera);

  // Finds all the active tracks in a reference frame, without changing the
  // internal state of the video tracker. This is effectively the same as
  // TrackFrame(), except:
  //   - It doesn't replace the current frame and current image with the new
  //   image and camera provided.
  //   - It doesn't add new points if we fall below the threshold
  //   - It doesn't delete points that weren't found in the passed frame.
  //
  // This call will however add the new camera to the observations of any points
  // that are successfully matched.
  //
  // This is useful if we're tracking a video, but want to find the same matches
  // in a non-adjacent frame or a separate image, like for loop closure. As far
  // as the tracker is concerned, the following two sets of calls will behave
  // identically:
  //
  // A)
  //    TrackFrame()
  //    TrackFrame()
  //    TrackFrame()
  // B)
  //    TrackFrame()
  //    MatchReference()
  //    MatchReference()
  //    TrackFrame()
  //    MatchReference()
  //    TrackFrame()
  //
  // A parameter is provided to specify a custom window_size, since non-adjacent
  // frames (larger baseline) may require a larger search window.
  const std::vector<TrackedPoint*>& MatchReference(const cv::Mat3b& frame,
                                                   const Camera* camera,
                                                   const int window_size = 31);

  // Returns all the tracked points in the lifetime of the video tracker. These
  // tracks may be totally new, with only a single observation.
  std::vector<TrackedPoint*> GetTracks() const;

  // Returns all the tracked points which were seen by the last tracked frame.
  // These tracks may be totally new, with only a single observation.
  std::vector<TrackedPoint*> GetActiveTracks() const;

  // Returns the 2D points for a particular frame. Mostly for visualization
  // purposes.
  std::vector<Eigen::Vector2d> GetKeypointsForFrame(const Camera* camera) const;

  // Returns the 2D points that are shared between a pair of frames.
  int GetMatchingKeypoints(const Camera* camera1, const Camera* camera2,
                           std::vector<Eigen::Vector2d>* points1,
                           std::vector<Eigen::Vector2d>* points2) const;

  // A static function to be used if you just want to track a set of points
  // between two frames and don't want to worry about TrackedPoints or Cameras
  // or anything. You pass in two images, and a set of points in the first
  // image, and it returns a vector of points in the second image. If a point
  // was not found, its value will be set to (-1,-1).
  static std::vector<Eigen::Vector2d> TrackPoints(
      const cv::Mat3b& source, const cv::Mat3b& dest,
      const std::vector<Eigen::Vector2d>& source_points,
      const int window_size = 31);

 private:
  const VideoTrackerOptions options_;
  std::vector<TrackedPoint*> active_points_;
  std::vector<TrackedPoint*> all_points_;
  const Camera* last_camera_;
  cv::Mat1b last_image_;
};
}  // namespace replay
