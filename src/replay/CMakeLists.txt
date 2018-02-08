# Add headers
file(GLOB REPLAY_HDRS *.h)

set(REPLAY_SRC
  io/spherical_video_reader.cc
  io/video_reader.cc
  io/read_ply_file.cc
  io/write_ply_file.cc
  io/zlib_decompressor.cc
  mesh.cc
  )

set(REPLAY_LIBRARY_DEPENDENCIES
  #${THEIA_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GL_LIBRARIES}
  ${CGAL_LIBRARIES}
  ${CGAL_3RD_PARTY_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${FFMPEG_LIBRARIES}
  ${ZLIB_LIBRARIES}
  ${CGAL_LIBRARIES}
  ${CGAL_3RD_PARTY_LIBRARIES}
  maxflow
  tinyply
  triangle
  slic
)

set(REPLAY_LIBRARY_SOURCE
  ${REPLAY_SRC}
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
