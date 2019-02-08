cmake_minimum_required(VERSION 2.8.3)
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif (COMMAND cmake_policy)

# Disable warnings about rpath on MacOS X for now.
# TODO(cmsweeney): Figure out the proper way to handle this!
if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # CMake 3.0
endif()

project(REPLAY C CXX)
include_directories(${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

SET(MACOSX_DEPLOYMENT_TARGET ${DARWIN_MAJOR_SDK_VERSION}.${DARWIN_MINOR_SDK_VERSION})
SET(CMAKE_CXX_FLAGS "--sysroot ${CMAKE_OSX_SYSROOT} ${CMAKE_CXX_FLAGS}")
MESSAGE("Setting MACOSX_DEPLOYMENT_TARGET to '${MACOSX_DEPLOYMENT_TARGET}'.")



# ==============================================================================
# Additional cmake find modules
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(BUILD_TESTING "Enable testing" ON)
# Add a path to the shader directory
add_definitions(-DREPLAY_SRC_DIR="${CMAKE_SOURCE_DIR}/src/replay/shaders")


enable_testing()
if (NOT MSVC)
  add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1)
endif (NOT MSVC)
set(GTEST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/libraries/gtest")
add_definitions(-DGTEST_TESTING_OUTPUT_DIRECTORY="${CMAKE_BINARY_DIR}/testing_output")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_DEBUG_POSTFIX "-debug")

# Replay data directory
add_definitions(-DREPLAY_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

# Eigen
set(MIN_EIGEN_VERSION 3.2.0)
find_package(Eigen ${MIN_EIGEN_VERSION} REQUIRED)
if (EIGEN_FOUND)
  message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")
endif (EIGEN_FOUND)

# Use a larger inlining threshold for Clang, since it hobbles Eigen,
# resulting in an unreasonably slow version of the blas routines. The
# -Qunused-arguments is needed because CMake passes the inline
# threshold to the linker and clang complains about it and dies.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Qunused-arguments -mllvm -inline-threshold=600")
  # Older versions of Clang (<= 2.9) do not support the 'return-type-c-linkage'
  # option, so check for its presence before adding it to the default flags set.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-Wno-return-type-c-linkage"
    HAVE_RETURN_TYPE_C_LINKAGE)
  if (HAVE_RETURN_TYPE_C_LINKAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type-c-linkage")
  endif(HAVE_RETURN_TYPE_C_LINKAGE)
endif ()

# Theia.
#find_package(Theia REQUIRED)
#if (THEIA_FOUND)
  #message("-- Found Theia SfM: ${THEIA_INCLUDE_DIRS}")
#else (THEIA_FOUND)
  #message(FATAL_ERROR "Can't find Theia SfM library.")
#endif (THEIA_FOUND)

# GFlags. The namespace patch is borrow from Ceres Solver (see license in
# FindGflags.cmake)
find_package(Gflags REQUIRED)
if (GFLAGS_FOUND)
  message("-- Found Google Flags: ${GFLAGS_INCLUDE_DIRS} in namespace: ${GFLAGS_NAMESPACE}")
  add_definitions(-DREPLAY_GFLAGS_NAMESPACE=${GFLAGS_NAMESPACE})
else (GFLAGS_FOUND)
  message(FATAL_ERROR "Can't find Google FLAGS.")
endif (GFLAGS_FOUND)

find_package(FreeImage REQUIRED)

# Google Logging
message("-- Check for Google Log")
find_package(Glog REQUIRED)
if (GLOG_FOUND)
  message("-- Found Google Logging: ${GLOG_INCLUDE_DIRS}")
else (GLOG_FOUND)
  message(FATAL_ERROR "Can't find Google Logging. Please set GLOG_INCLUDE_DIR & "
    "GLOG_LIBRARY")
endif (GLOG_FOUND)

message("-- Check for CGAL")
find_package(CGAL REQUIRED)
if (CGAL_FOUND)
  message("-- Found CGAL: ${CGAL_INCLUDE_DIRS}")
  include(${CGAL_USE_FILE})
else (CGAL_FOUND)
  message(FATAL_ERROR "Can't find CGAL.")
endif (CGAL_FOUND)

# CUDA for BiCGSTAB
message("-- Check for CUDA")
find_package(CUDA)
if (CUDA_FOUND)
  add_definitions(-DREPLAY_CUDA_ENABLED)
  # The settings for CUDA were borrowed from the COLMAP library:
  # https://github.com/colmap/colmap/blob/master/src/mvs/CMakeLists.txt
  #set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
  if (MSVC)
    # Workaround for Visual Studio compiler error,
    # due problems with compiling too complex code.
    string(REPLACE "/Zo" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else ()
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
  endif (MSVC)
  message("-- Found Cuda: ${CUDA_INCLUDE_DIRS}")
else (CUDA_FOUND)
  message("-- Can't find Cuda. Disabling mesh refinement.")
endif (CUDA_FOUND)

# Ceres
message("-- Check for Ceres")
find_package(Ceres REQUIRED)
if (CERES_FOUND)
  message("-- Found Ceres: ${CERES_INCLUDE_DIRS}")
else (CERES_FOUND)
  message(FATAL_ERROR "Can't find Ceres. Please set CERES_INCLUDE_DIR & "
    "CERES_LIBRARY")
endif (CERES_FOUND)

find_package(OpenGL REQUIRED)
if (OPENGL_FOUND)
  message("-- Found OpenGL: ${OPENGL_INCLUDE_DIR}")
  find_package(GLFW3 3.2 REQUIRED)
  if (GLFW_FOUND)
    message ("-- Found GLFW: ${GLFW_INCLUDE_DIR} ")
    find_package(GLEW REQUIRED)
    if (GLEW_FOUND)
      message ("-- Found GLEW: ${GLEW_INCLUDE_DIR}")
      list(APPEND GL_INCLUDE_DIRS
	${GLFW_INCLUDE_DIR}
	${GLFW_INCLUDE_DIRS}
	${OPENGL_INCLUDE_DIR}
	${GLEW_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIR})
      list(APPEND GL_LIBRARIES
	${GLFW_LIBRARY}
	${OPENGL_LIBRARIES}
	${OPENGL_gl_LIBRARY}
	${GLEW_LIBRARIES})
    else (GLEW FOUND)
      message (FATAL_ERROR "-- Can't find GLEW. Please set GLEW_INCLUDE_DIR & GLEW_LIBRARY")
    endif (GLEW_FOUND)
  else (GLFW_FOUND)
    message (FATAL_ERROR "-- Can't find GLFW")
  endif (GLFW_FOUND)
else (OPENGL_FOUND)
  message (FATAL_ERROR "-- Can't find OpenGL!")
endif (OPENGL_FOUND)

find_package(OpenCV REQUIRED) 
if (OpenCV_FOUND) 
  message("-- Found OpenCV in: ${OpenCV_INCLUDE_DIRS}")
endif (OpenCV_FOUND)

find_package(FFmpeg REQUIRED) 
if (FFMPEG_FOUND) 
  message("-- Found FFMPEG in: ${FFMPEG_INCLUDE_DIR}")
endif (FFMPEG_FOUND)

find_package(ZLIB REQUIRED) 
if (ZLIB_FOUND) 
  message("-- Found ZLIB in: ${ZLIB_INCLUDE_DIRS}")
endif (ZLIB_FOUND)

find_package(OPENVR REQUIRED) 
if (OPENVR_FOUND) 
  message("-- Found OPENVR in: ${OPENVR_INCLUDE_DIRS}")
endif (OPENVR_FOUND)

include_directories(
  include
  src
  #${THEIA_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${GL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${FFMPEG_INCLUDE_DIR}
  ${ZLIB_INCLUDE_DIRS}
  ${CGAL_INCLUDE_DIRS}
  ${OPENVR_INCLUDE_DIRS}
  ${FREEIMAGE_INCLUDE_DIRS}
)

# NOTE: This fix came from Ceres solver with the following comment:
#
# Eigen SparseQR generates various compiler warnings related to unused and
# uninitialised local variables, which prevents Ceres compilation as we use
# -Werror.  To avoid having to individually suppress these warnings around
# the #include statments for Eigen headers across all GCC/Clang versions, we
# tell CMake to treat Eigen headers as system headers.  This results in all
# compiler warnings from them being suppressed.
#
# Note that this is *not* propagated to clients, ie CERES_INCLUDE_DIRS
# used by clients after find_package(Ceres) does not identify Eigen as
# as system headers.
include_directories(SYSTEM ${EIGEN_INCLUDE_DIRS})

# build Third party libraries included in distro.
add_subdirectory(libraries)


include_directories(
  libraries
  ${cereal_SOURCE_DIR}
  ${cereal_SOURCE_DIR}/include
  ${gtest_SOURCE_DIR}/include
  ${gtest_SOURCE_DIR}
  ${tinyply_SOURCE_DIR}
  ${triangle_SOURCE_DIR}
  ${maxflow_SOURCE_DIR}
)

if (CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS} libraries/cusp)
endif (CUDA_FOUND)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(OptimizeCompilerFlags)
optimizecompilerflags()

add_subdirectory(src/replay)
add_subdirectory(applications)
