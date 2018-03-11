#include <glog/logging.h>
#include <replay/vr_180/vr_180_video_reader.h>
#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include "replay/rendering/opengl_renderer.h"
#include "replay/rendering/stereo_video_angular_renderer.h"
#include "openvr.h"

DEFINE_string(video_file, "", "Spherical video file to parse");

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(FLAGS_video_file.length() > 0) << "Filename is empty.";
  replay::VR180VideoReader reader;

  std::shared_ptr<replay::OpenGLRenderer> renderer = std::make_shared<replay::OpenGLRenderer>();
  renderer->Initialize();

  replay::StereoVideoAngularRenderer stereo_renderer(renderer);
  int file_number = 1;
  CHECK(stereo_renderer.Initialize(FLAGS_video_file));

  vr::EVRInitError error = vr::VRInitError_None;
  vr::IVRSystem *hmd = vr::VR_Init(&error, vr::VRApplication_Scene);

  if (error != vr::VRInitError_None)
  {
	  hmd = NULL;
	  char buf[1024];
	  sprintf_s(buf, sizeof(buf), "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(error));
	  LOG(FATAL) << buf;
	  return false;
  }
  vr::IVRRenderModels *renderModels = (vr::IVRRenderModels *)vr::VR_GetGenericInterface(vr::IVRRenderModels_Version, &error);

  if (!renderModels)
  {
	  hmd = NULL;
	  vr::VR_Shutdown();

	  char buf[1024];
	  sprintf_s(buf, sizeof(buf), "Unable to get render model interface: %s", vr::VR_GetVRInitErrorAsEnglishDescription(error));
	  LOG(FATAL) << buf;
	  return false;
  }

  LOG(INFO) << "Successfully initialized HMD!";
  uint32_t width, height;
  hmd->GetRecommendedRenderTargetSize(&width, &height);
  vr::HmdMatrix44_t mat = hmd->GetProjectionMatrix(vr::Eye_Left, 0.01, 2);

  //camera.SetFocalLength(mat.m[0][0] * width / 2);
  //camera.SetPrincipalPoint(width/2, height/2);
  vr::TrackedDevicePose_t tracked_device_pose[vr::k_unMaxTrackedDeviceCount];
  Eigen::Matrix4f projection_left;
  projection_left << mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3],
	  mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3],
	  mat.m[2][0], mat.m[2][1], mat.m[2][2], mat.m[2][3],
	  mat.m[3][0], mat.m[3][1], mat.m[3][2], mat.m[3][3];
  mat = hmd->GetProjectionMatrix(vr::Eye_Right, 0.01, 2);
  Eigen::Matrix4f projection_right;
  projection_right << mat.m[0][0], mat.m[0][1], mat.m[0][2], mat.m[0][3],
	  mat.m[1][0], mat.m[1][1], mat.m[1][2], mat.m[1][3],
	  mat.m[2][0], mat.m[2][1], mat.m[2][2], mat.m[2][3],
	  mat.m[3][0], mat.m[3][1], mat.m[3][2], mat.m[3][3];
  int hmd_index = -1;
	  for (int nDevice = 0; nDevice < vr::k_unMaxTrackedDeviceCount; ++nDevice)
	  {
		  if (hmd->GetTrackedDeviceClass(nDevice) == vr::TrackedDeviceClass_HMD)
		  {
			  hmd_index = nDevice;
			  break;
		  }

	  }
  
  while (true) {
	  vr::VRCompositor()->WaitGetPoses(tracked_device_pose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

	  vr::HmdMatrix34_t abs_tracking = tracked_device_pose[hmd_index].mDeviceToAbsoluteTracking;
	  Eigen::Matrix3f rotation;
	  rotation << abs_tracking.m[0][0], abs_tracking.m[0][1], abs_tracking.m[0][2], abs_tracking.m[1][0], abs_tracking.m[1][1], abs_tracking.m[1][2], abs_tracking.m[2][0], abs_tracking.m[2][1], abs_tracking.m[2][2];
	 // Eigen::Vector3f tracked(abs_tracking.m[2][0], abs_tracking.m[2][1], abs_tracking.m[2][2]);
	
	  stereo_renderer.RenderEye(projection_left, width, height, 0, rotation);
	  stereo_renderer.RenderEye(projection_right, width, height, 1, rotation);
    int keyvalue = cv::waitKey(1);

	if (keyvalue == ' ') {
		break;
	}
  }
  return 0;
}
