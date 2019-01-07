#include <glog/logging.h>
#include <replay/sfm/calibrate_from_mesh.h>
#include <replay/util/filesystem.h>
#include <replay/util/strings.h>
#include <replay/vr_180/vr_180_video_reader.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "replay/camera/fisheye_camera.h"
#include "replay/mesh/mesh.h"
#include "replay/sfm/reconstruction.h"
#include "replay/sfm/tracked_point.h"
#include "replay/sfm/video_tracker.h"
#include "replay/util/matrix_utils.h"

DEFINE_string(video_file, "", "Spherical video file to parse");
DEFINE_string(
    tracked_reconstruction, "",
    "Optional: tracked reconstruction. If provided, and existing file, it will "
    "be loaded. Otherwise, it will be used to save the intermediate tracked "
    "reconstruction (video tracking and calibration, pre BA)");
DEFINE_string(output_reconstruction, "", "Output reconstruction file");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_video_file.length() > 0) << "Filename is empty.";
  replay::VR180VideoReader reader;
  reader.Open(FLAGS_video_file);
  replay::Reconstruction scene;

  if (FLAGS_tracked_reconstruction.length() > 0 &&
      replay::FileExists(FLAGS_tracked_reconstruction)) {
    CHECK(scene.Load(FLAGS_tracked_reconstruction));
  } else {
    std::vector<replay::Mesh> meshes = reader.GetMeshes();
    replay::FisheyeCamera left_camera, right_camera;

    const int width = reader.GetWidth() / 2;
    const int height = reader.GetHeight();
    left_camera.SetImageSize(Eigen::Vector2i(width, height));
    left_camera.SetFocalLength(
        Eigen::Vector2d(576.835880496, height * 576.835880496 / 1920.0));
    left_camera.SetPrincipalPoint(Eigen::Vector2d(
        967.01102733744608, 960.09077375080085 * height / 1920.0));
    left_camera.SetSkew(0);
    left_camera.SetDistortionCoeffs(
        {0.12481379377850065, -0.053443040456748615, 0.0047589052125656906,
         -0.0011817204127081628, -0.00061737746607320326});
    left_camera.SetPosition(Eigen::Vector3d(-0.032102036000307241,
                                            5.8096701608655296e-05,
                                            -0.00017414086709508259));

    replay::CalibrateFromMesh(meshes[0], &left_camera);

    right_camera = left_camera;

    replay::CalibrateFromMesh(meshes[1], &right_camera);

    reader.SeekToFrame(200);

    double time;
    cv::Mat3b frame;
    Eigen::Matrix3f rotation;
    replay::VideoTrackerOptions tracker_options;
    tracker_options.max_points = 7000;
    tracker_options.min_points = 6000;
    tracker_options.distance_between_keypoints = 10;
    tracker_options.window_size = 31;

    Eigen::Vector2d camera_fov = left_camera.GetFOV();
    double min_fov = std::min(camera_fov[0], camera_fov[1]);
    tracker_options.max_angle_from_optical_axis = (min_fov / 2) - 5;

    replay::VideoTracker tracker(tracker_options);

    cv::Mat1b last_frame;

    // 6.35cm is the IPD of the camera
    // 12cm is the the approximate radius of the circle, when holding camera to
    // forehead
    //static const double ipd = 6.35;
    //static const double circle_radius = 12.0;
    static const double min_triangulation_angle = 0.3;

    static const Eigen::Vector3d left_offset(-3.2102036, 5.80967e-5,
                                             -0.00017414);
    static const Eigen::Vector3d right_offset(3.2102036, -5.80967e-5,
                                              0.00017414);
    int frame_id = 0;

    while (reader.GetOrientedFrame(frame, rotation, &time)) {
      if (frame_id > 0) {
        break;
      }
      cv::Mat3b left_image = frame(cv::Rect(0, 0, width, height));
      cv::Mat3b right_image = frame(cv::Rect(width, 0, width, height));
      replay::FisheyeCamera* left_frame_camera =
          new replay::FisheyeCamera(left_camera);
      replay::FisheyeCamera* right_frame_camera =
          new replay::FisheyeCamera(right_camera);

      left_frame_camera->SetPosition(
            left_offset);
      right_frame_camera->SetPosition(
          right_offset);

      // Change these to a camera->Rotate() call. So it preserves everything else. Or maybe figure out how to set the position so that it's consistent.
      left_frame_camera->SetRotation(rotation.cast<double>());
      right_frame_camera->SetRotation(rotation.cast<double>());

      tracker.TrackFrame(left_image, left_frame_camera);

      // Find the left points in the right image
      tracker.MatchReference(right_image, right_frame_camera, 31);

      LOG(ERROR) << "Baseline: "
                 << left_frame_camera->GetRightVector().normalized().dot(
                        (right_frame_camera->GetPosition() -
                         left_frame_camera->GetPosition()));
      // Go through the points and triangulate them
      const std::vector<replay::TrackedPoint*> active_tracks =
          tracker.GetActiveTracks();
      for (int i = 0; i < active_tracks.size(); i++) {
        if (!active_tracks[i]->HasObservation(right_frame_camera)) {
          continue;
        }
        const Eigen::Vector3d point_to_left_camera =
            left_frame_camera
                ->PixelToWorldRay(
                    active_tracks[i]->GetObservation2D(left_frame_camera))
                .normalized();
        const Eigen::Vector3d point_to_right_camera =
            right_frame_camera
                ->PixelToWorldRay(
                    active_tracks[i]->GetObservation2D(right_frame_camera))
                .normalized();
        double triangulation_angle =
            acos(point_to_left_camera.dot(point_to_right_camera)) * 180.0 /
            M_PI;
        // Only include points that are near the horizontal middle of the frame.
        // The edges have less parallax and are unreliable for triangulation
        if (triangulation_angle > min_triangulation_angle) {
          active_tracks[i]->Triangulate(
              {left_frame_camera, right_frame_camera});
        }
      }

      std::vector<Eigen::Vector2d> left_points(active_tracks.size()),
          right_points(active_tracks.size());
      std::vector<cv::Scalar> colors(active_tracks.size(), cv::Scalar(0, 0, 0));
      double max_depth = 1000;
      double min_depth = 0;
      for (int i = 0; i < active_tracks.size(); i++) {
        if (!active_tracks[i]->HasObservation(right_frame_camera)) {
          continue;
        }
        left_points[i] = active_tracks[i]->GetObservation2D(left_frame_camera);
        right_points[i] =
            active_tracks[i]->GetObservation2D(right_frame_camera);
        double depth = active_tracks[i]->GetObservation(right_frame_camera).z();
        depth = (depth - min_depth) / (max_depth - min_depth);
        colors[i] = cv::Scalar((1 - depth) * 255, 0, depth * 255);
        if (depth <= 0) {
          colors[i] = cv::Scalar(0, 0, 0);
        }
        if (depth > 1) {
          colors[i] = cv::Scalar(0, 0, 255);
        }
      }
      cv::imshow("tracker",
                 replay::VisualizeMatches(left_image, left_points, right_image,
                                          right_points, 1000, colors));
      cv::waitKey(1);

      scene.FindSimilarViewpoints(left_frame_camera);
      tracker.MatchReference(right_image, right_frame_camera, 31);

      scene.AddCamera(left_frame_camera);
      scene.AddCamera(right_frame_camera);
      LOG(INFO) << "Tracked frame " << frame_id;
      frame_id++;
    }

    // Go through the tracked points and remove any observations that don't have
    // depth
    std::vector<replay::TrackedPoint*> tracked_points = tracker.GetTracks();
    std::vector<replay::TrackedPoint*> filtered_points;
    for (int i = 0; i < tracked_points.size(); i++) {
      // A point we want to keep for bundle-adjustment should have at least six
      // observations. A left/right pair for three views. It also must have a 3D
      // position, otherwise it indicates that the triangulation has failed.
      if (tracked_points[i]->NumObservations() >= 4 &&
          tracked_points[i]->GetPoint().norm() > 0.0) {
        filtered_points.push_back(tracked_points[i]);
      }
    }

    scene.SetPoints(tracked_points);
    CHECK(scene.Save(FLAGS_tracked_reconstruction));
  }
  scene.SaveMesh("/Users/holynski/beforeBA.ply");

  return 1;
}
