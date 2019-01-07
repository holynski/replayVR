# Add headers
file(GLOB REPLAY_HDRS *.h)

set(THIRD_PARTY_SRC
  third_party/theia/sfm/camera/camera.cc
  third_party/theia/sfm/camera/camera_intrinsics_model.cc
  third_party/theia/sfm/camera/division_undistortion_camera_model.cc
  third_party/theia/sfm/camera/fisheye_camera_model.cc
  third_party/theia/sfm/camera/fov_camera_model.cc
  third_party/theia/sfm/camera/pinhole_camera_model.cc
  third_party/theia/sfm/camera/pinhole_radial_tangential_camera_model.cc
  third_party/theia/sfm/camera/projection_matrix_utils.cc
  )

set(REPLAY_SRC
  camera/camera.cc
  camera/fisheye_camera.cc
  camera/pinhole_camera.cc
  depth_map/depth_map.cc
  io/byte_stream_reader.cc
  io/read_ply_file.cc
  io/video_reader.cc
  io/write_ply_file.cc
  io/zlib_decompressor.cc
  mesh/mesh.cc
  rendering/opengl_context.cc
  rendering/stereo_video_angular_renderer.cc
  rendering/vr180_renderer.cc
  rendering/vr180_undistorter.cc
  rendering/vr_context.cc
  sfm/calibrate_from_mesh.cc
  sfm/reconstruction.cc
  sfm/tracked_point.cc
  sfm/video_tracker.cc
  util/filesystem.cc
  util/lru_cache.cc
  util/matrix_utils.cc
  util/strings.cc
  util/timer.cc
  vr_180/mesh_projection_parser.cc
  vr_180/vr_180_video_reader.cc
  )


set(REPLAY_LIBRARY_DEPENDENCIES
  #${THEIA_LIBRARIES}
  ${CERES_LIBRARIES}
  ${CGAL_3RD_PARTY_LIBRARIES}
  ${CGAL_3RD_PARTY_LIBRARIES}
  ${CGAL_LIBRARIES}
  ${CGAL_LIBRARIES}
  ${FFMPEG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GL_LIBRARIES}
  ${OPENVR_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${ZLIB_LIBRARIES}
  maxflow
  slic
  tinyply
  triangle
)

set(REPLAY_LIBRARY_SOURCE
  ${REPLAY_SRC}
  ${THIRD_PARTY_SRC}
  ${REPLAY_HDRS})

add_library(replay ${REPLAY_LIBRARY_SOURCE})
set_target_properties(replay PROPERTIES
  VERSION ${REPLAY_VERSION}
  SOVERSION ${REPLAY_VERSION_MAJOR}
  )
target_link_libraries(replay ${REPLAY_LIBRARY_DEPENDENCIES})

add_custom_command(TARGET replay POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/replay/shaders
        $<TARGET_FILE_DIR:replay>/shaders)

install(TARGETS replay
  EXPORT  REPLAYExport
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib${LIB_SUFFIX}
  ARCHIVE DESTINATION lib${LIB_SUFFIX})

if (BUILD_TESTING)
  macro (GTEST FILENAME)
    string(REPLACE "/" ";" TEST_NAME_LIST ${FILENAME})
    list(REVERSE TEST_NAME_LIST)
    list(GET TEST_NAME_LIST 0 TEST_NAME)
    add_executable(${TEST_NAME}_test test/test_main.cc ${FILENAME}_test.cc)
    target_link_libraries(${TEST_NAME}_test
      gtest
      replay
      ${REPLAY_LIBRARY_DEPENDENCIES})
    add_test(NAME ${TEST_NAME}_test
      COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TEST_NAME}_test)
  endmacro (GTEST)

  #gtest(experimental/mask_plane_fit/ransac_plane_fit)
endif (BUILD_TESTING)
