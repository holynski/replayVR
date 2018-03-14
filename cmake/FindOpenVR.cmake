# - try to find the OpenVR SDK - currently designed for the version on GitHub.
#
# Cache Variables: (probably not for direct use in your scripts)
#  OPENVR_INCLUDE_DIR
#
# Non-cache variables you might use in your CMakeLists.txt:
#  OPENVR_FOUND
#  OPENVR_INCLUDE_DIRS
#  OPENVR_PLATFORM - something like Win32, Win64, etc.
#
# Requires these CMake modules:
#  FindPackageHandleStandardArgs (known included with CMake >=2.6.2)
#
# Original Author:
# 2015 Ryan A. Pavlik <ryan@sensics.com>
#
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

find_path(OPENVR_ROOT_DIR
	NAMES
	OpenVRSDK/headers/openvr.h
	OpenVRSDK/include/openvr.h
	PATHS
	"C:/Program Files/"
	${_root_dirs}
	PATH_SUFFIXES
	${OPENVR_DIR}
	"$ENV{PROGRAMFILES}"
  "${CMAKE_SOURCE_DIR}/libraries/openvr/"
)

set(OPENVR_ROOT_DIR "${OPENVR_ROOT_DIR}/OpenVRSDK/")

find_path(OPENVR_INCLUDE_DIR
	NAMES
	openvr_driver.h
	HINTS
	include/
	headers/
	PATHS
  ${CMAKE_SOURCE_DIR}/libraries/openvr/headers/
	${OPENVR_ROOT_DIR}
	PATH_SUFFIXES
	headers
	public/headers
	steam
	public/steam
  )

find_library( OPENVR_LIBRARY
            NAMES
                openvr_api64
				openvr_api32
				openvr_api
            HINTS
                "${OPENVR_ROOT_DIR}/lib"
                "${OPENVR_ROOT_DIR}/lib/x64"
                "$ENV{OPENVR_ROOT_DIR}/lib"
                "$ENV{OPENVR_ROOT_DIR}/lib/x64"
            PATHS
                "$ENV{PROGRAMFILES}/OpenVRSDK/lib"
				"$ENV{PROGRAMFILES}/openvr/lib"
        )

if (OPENVR_LIBRARY) 
if (OPENVR_INCLUDE_DIR)
	set(OPENVR_FOUND "YES")
endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVR
	DEFAULT_MSG
	OPENVR_INCLUDE_DIR)

if(OPENVR_FOUND)
	list(APPEND OPENVR_INCLUDE_DIRS ${OPENVR_INCLUDE_DIR})
	list(APPEND OPENVR_LIBRARIES ${OPENVR_LIBRARY})
	mark_as_advanced(OPENVR_ROOT_DIR)
endif()

mark_as_advanced(OPENVR_INCLUDE_DIR)
