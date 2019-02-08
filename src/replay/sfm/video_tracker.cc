
#include "replay/sfm/video_tracker.h"

#include "replay/camera/camera.h"
#include "replay/sfm/tracked_point.h"

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace replay {

cv::Mat3b VisualizeMatches(const cv::Mat3b& image1,
                           const std::vector<Eigen::Vector2d>& points1,
                           const cv::Mat3b& image2,
                           const std::vector<Eigen::Vector2d>& points2,
                           const int max_dimension,
                           const std::vector<cv::Scalar>& colors) {
  CHECK_EQ(points1.size(), points2.size());
  cv::Mat3b canvas(std::max(image1.rows, image2.rows),
                   image1.cols + image2.cols);
  static const int point_radius = std::max(canvas.rows, canvas.cols) / 500;
  image1.copyTo(canvas(cv::Rect(0, 0, image1.cols, image1.rows)));
  image2.copyTo(canvas(cv::Rect(image1.cols, 0, image2.cols, image2.rows)));
  for (int i = 0; i < points1.size(); i++) {
    const cv::Scalar unique_color((i % 10) * 25, i % 100 + 155, i % 191 + 50);
    if (points1[i].x() >= 0 && points1[i].y() >= 0) {
      cv::circle(
          canvas, cv::Point2f(points1[i].x(), points1[i].y()), point_radius,
          (colors.size() == points1.size() ? colors[i] : unique_color), -1);
    }
    if (points2[i].x() >= 0 && points2[i].y() >= 0) {
      cv::circle(
          canvas, cv::Point2f(points2[i].x() + image1.cols, points2[i].y()),
          point_radius,
          (colors.size() == points1.size() ? colors[i] : unique_color), -1);
    }
  }
  if (max_dimension > 0) {
    float scale_factor =
        max_dimension / static_cast<float>(std::max(canvas.rows, canvas.cols));
    cv::resize(canvas, canvas, cv::Size(0, 0), scale_factor, scale_factor);
  }
  return canvas;
}

VideoTracker::VideoTracker(const VideoTrackerOptions& options)
    : options_(options) {}

const std::vector<TrackedPoint*>& VideoTracker::TrackFrame(
    const cv::Mat3b& frame, const Camera* camera) {
  cv::Mat1b frame_gray;
  cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

  if (last_camera_) {
    CHECK_EQ(frame_gray.size(), last_image_.size());
    CHECK_EQ(last_image_.depth(), frame_gray.depth());

    std::vector<cv::Point2f> points;
    for (const auto& point : active_points_) {
      points.emplace_back(point->GetObservation(last_camera_).x(),
                          point->GetObservation(last_camera_).y());
    }

    // Perform a forwards pass on existing features.
    std::vector<cv::Point2f> tracked_points;
    std::vector<unsigned char> tracking_status;
    std::vector<float> flow_error;
    cv::calcOpticalFlowPyrLK(
        last_image_, frame_gray, points, tracked_points, tracking_status,
        flow_error, cv::Size(options_.window_size, options_.window_size));

    // Perform a backwards pass for bi-directional consistency.
    std::vector<cv::Point2f> reverse_tracked_points;
    std::vector<unsigned char> reverse_tracking_status;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(
        frame_gray, last_image_, tracked_points, reverse_tracked_points,
        reverse_tracking_status, backward_error,
        cv::Size(options_.window_size, options_.window_size));

    std::vector<bool> mark_for_deletion(active_points_.size(), false);
    for (int i = 0; i < points.size(); i++) {
      const cv::Point2f cycle_error = points[i] - reverse_tracked_points[i];
      const double squared_cycle_error =
          cycle_error.x * cycle_error.x + cycle_error.y * cycle_error.y;

      // If tracking succeeded, add the track to the list of valid tracks.
      if (tracking_status[i] == 1 && reverse_tracking_status[i] == 1 &&
          tracked_points[i].x >= 0 && tracked_points[i].y >= 0 &&
          tracked_points[i].x < frame.cols &&
          tracked_points[i].y < frame.rows && squared_cycle_error <= 1.0 &&
          camera->AngleFromOpticalAxis(camera->UnprojectPoint(
              Eigen::Vector2d(tracked_points[i].x, tracked_points[i].y), 1)) <
              options_.max_angle_from_optical_axis) {
        active_points_[i]->SetObservation(camera, tracked_points[i]);
      } else {
        mark_for_deletion[i] = true;
      }
    }
    for (int i = active_points_.size() - 1; i >= 0; i--) {
      if (mark_for_deletion[i]) {
        active_points_.erase(active_points_.begin() + i);
      }
    }
  }
  if (active_points_.size() < options_.min_points) {
    LOG(INFO) << "Adding new points!";
    cv::Mat1b mask(frame.rows, frame.cols, uint8_t(255));
    for (int i = 0; i < active_points_.size(); i++) {
      cv::circle(mask,
                 cv::Point2f(active_points_[i]->GetObservation(camera).x(),
                             active_points_[i]->GetObservation(camera).y()),
                 options_.distance_between_keypoints, cv::Scalar(0), -1);
    }
    const int num_points_to_add = options_.max_points - active_points_.size();
    std::vector<cv::Point2f> new_points;
    cv::goodFeaturesToTrack(frame_gray, new_points, num_points_to_add, 0.01,
                            options_.distance_between_keypoints, mask);
    if (new_points.size() == 0) {
      last_image_ = frame_gray;
      return active_points_;
    }

    // Refine the locations of these features.
    cv::cornerSubPix(
        frame_gray, new_points, cv::Size(10, 10), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::Type::EPS + cv::TermCriteria::Type::COUNT, 40, 0.001));
    for (int i = 0; i < new_points.size(); i++) {
      Eigen::Vector2d point2d(new_points[i].x, new_points[i].y);
      Eigen::Vector3d point3d = camera->UnprojectPoint(point2d, 1);
      double angle = camera->AngleFromOpticalAxis(point3d);
      if (angle < options_.max_angle_from_optical_axis) {
        TrackedPoint* point = new TrackedPoint();
        active_points_.push_back(point);
        all_points_.push_back(point);
        point->SetObservation(camera, new_points[i]);
      }
    }
  }

  last_camera_ = camera;
  last_image_ = frame_gray;
  return active_points_;
}

const std::vector<TrackedPoint*>& VideoTracker::MatchReference(
    const cv::Mat3b& frame, const Camera* camera, const int window_size) {
  cv::Mat1b frame_gray;
  cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

  if (last_camera_) {
    CHECK_EQ(frame_gray.size(), last_image_.size());
    CHECK_EQ(last_image_.depth(), frame_gray.depth());

    std::vector<cv::Point2f> points;
    for (const auto& point : active_points_) {
      points.emplace_back(point->GetObservation(last_camera_).x(),
                          point->GetObservation(last_camera_).y());
    }

    // Perform a forwards pass on existing features.
    std::vector<cv::Point2f> tracked_points;
    std::vector<unsigned char> tracking_status;
    std::vector<float> flow_error;
    cv::calcOpticalFlowPyrLK(last_image_, frame_gray, points, tracked_points,
                             tracking_status, flow_error,
                             cv::Size(window_size, window_size));

    // Perform a backwards pass for bi-directional consistency.
    std::vector<cv::Point2f> reverse_tracked_points;
    std::vector<unsigned char> reverse_tracking_status;
    std::vector<float> backward_error;
    cv::calcOpticalFlowPyrLK(frame_gray, last_image_, tracked_points,
                             reverse_tracked_points, reverse_tracking_status,
                             backward_error,
                             cv::Size(window_size, window_size));

    for (int i = 0; i < points.size(); i++) {
      const cv::Point2f cycle_error = points[i] - reverse_tracked_points[i];
      const double squared_cycle_error =
          cycle_error.x * cycle_error.x + cycle_error.y * cycle_error.y;

      // If tracking succeeded, add the track to the list of valid tracks.
      if (tracking_status[i] == 1 && reverse_tracking_status[i] == 1 &&
          tracked_points[i].x >= 0 && tracked_points[i].y >= 0 &&
          tracked_points[i].x < frame.cols &&
          tracked_points[i].y < frame.rows && squared_cycle_error <= 1.0 &&
          camera->AngleFromOpticalAxis(camera->UnprojectPoint(
              Eigen::Vector2d(tracked_points[i].x, tracked_points[i].y), 1)) <
              options_.max_angle_from_optical_axis) {
        active_points_[i]->SetObservation(camera, tracked_points[i]);
      }
    }
  }

  return active_points_;
}

std::vector<TrackedPoint*> VideoTracker::GetTracks() const {
  return all_points_;
}
std::vector<TrackedPoint*> VideoTracker::GetActiveTracks() const {
  return active_points_;
}

std::vector<Eigen::Vector2d> VideoTracker::TrackPoints(
    const cv::Mat3b& source, const cv::Mat3b& dest,
    const std::vector<Eigen::Vector2d>& source_points, const int window_size) {
  CHECK_EQ(source.size(), dest.size());
  CHECK_EQ(source.depth(), dest.depth());
  CHECK_GT(source_points.size(), 0);

  cv::Mat1b source_gray;
  cv::Mat1b dest_gray;
  std::vector<Eigen::Vector2d> new_points(source_points.size(),
                                          Eigen::Vector2d(-1, -1));

  cv::cvtColor(source, source_gray, cv::COLOR_RGB2GRAY);
  cv::cvtColor(dest, dest_gray, cv::COLOR_RGB2GRAY);

  std::vector<cv::Point2f> points;
  for (const auto& point : source_points) {
    points.emplace_back(point.x(), point.y());
  }

  // Perform a forwards pass on existing features.
  std::vector<cv::Point2f> tracked_points;
  std::vector<unsigned char> tracking_status;
  std::vector<float> flow_error;
  cv::calcOpticalFlowPyrLK(source_gray, dest_gray, points, tracked_points,
                           tracking_status, flow_error,
                           cv::Size(window_size, window_size));

  // Perform a backwards pass for bi-directional consistency.
  std::vector<cv::Point2f> reverse_tracked_points;
  std::vector<unsigned char> reverse_tracking_status;
  std::vector<float> backward_error;
  cv::calcOpticalFlowPyrLK(dest_gray, source_gray, tracked_points,
                           reverse_tracked_points, reverse_tracking_status,
                           backward_error, cv::Size(window_size, window_size));

  for (int i = 0; i < points.size(); i++) {
    const cv::Point2f cycle_error = points[i] - reverse_tracked_points[i];
    const double squared_cycle_error =
        cycle_error.x * cycle_error.x + cycle_error.y * cycle_error.y;

    // If tracking succeeded, add the track to the list of valid tracks.
    if (tracking_status[i] == 1 && reverse_tracking_status[i] == 1 &&
        tracked_points[i].x >= 0 && tracked_points[i].y >= 0 &&
        tracked_points[i].x < source.cols &&
        tracked_points[i].y < source.rows && squared_cycle_error <= 1.0) {
      new_points[i] = Eigen::Vector2d(tracked_points[i].x, tracked_points[i].y);
    }
  }
  return new_points;
}

std::vector<Eigen::Vector2d> VideoTracker::GetKeypointsForFrame(
    const Camera* camera) const {
  std::vector<Eigen::Vector2d> points_eigen;
  for (int i = 0; i < all_points_.size(); i++) {
    if (all_points_[i]->HasObservation(camera)) {
      points_eigen.emplace_back(
          all_points_[i]->GetObservation(camera).head<2>());
    }
  }
  return points_eigen;
}

}  // namespace replay
