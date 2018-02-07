# FindMVE.cmake - Find MVE library.
#
# This module defines the following variables:
#
# MVE_FOUND: TRUE iff mve is found.
# MVE_INCLUDE_DIRS: Include directories for mve.
# MVE_LIBRARIES: Libraries required to link mve.
#
# The following variables control the behaviour of this module:
#
# MVE_INCLUDE_DIR_HINTS: List of additional directories in which to
#                         search for mve includes, e.g: /timbuktu/include.
# MVE_LIBRARY_DIR_HINTS: List of additional directories in which to
#                         search for mve libraries, e.g: /timbuktu/lib.
#
# The following variables are also defined by this module, but in line with
# CMake recommended FindPackage() module style should NOT be referenced directly
# by callers (use the plural variables detailed above instead).  These variables
# do however affect the behaviour of the module via FIND_[PATH/LIBRARY]() which
# are NOT re-called (i.e. search for library is not repeated) if these variables
# are set with valid values _in the CMake cache_. This means that if these
# variables are set directly in the cache, either by the user in the CMake GUI,
# or by the user passing -DVAR=VALUE directives to CMake when called (which
# explicitly defines a cache variable), then they will be used verbatim,
# bypassing the HINTS variables and other hard-coded search locations.
#
# MVE_INCLUDE_DIR: Include directory for mve, not including the
#                   include directory of any dependencies.
# MVE_LIBRARY: mve library, not including the libraries of any
#               dependencies.

# Reset CALLERS_CMAKE_FIND_LIBRARY_PREFIXES to its value when
# FindMVE was invoked.
macro(MVE_RESET_FIND_LIBRARY_PREFIX)
  if (MSVC)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${CALLERS_CMAKE_FIND_LIBRARY_PREFIXES}")
  endif (MSVC)
endmacro(MVE_RESET_FIND_LIBRARY_PREFIX)

# Called if we failed to find mve or any of it's required dependencies,
# unsets all public (designed to be used externally) variables and reports
# error message at priority depending upon [REQUIRED/QUIET/<NONE>] argument.
macro(MVE_REPORT_NOT_FOUND REASON_MSG)
  unset(MVE_FOUND)
  unset(MVE_INCLUDE_DIRS)
  unset(MVE_LIBRARIES)
  # Make results of search visible in the CMake GUI if mve has not
  # been found so that user does not have to toggle to advanced view.
  mark_as_advanced(CLEAR MVE_INCLUDE_DIR
                         MVE_LIBRARY)

  mve_reset_find_library_prefix()

  # Note <package>_FIND_[REQUIRED/QUIETLY] variables defined by FindPackage()
  # use the camelcase library name, not uppercase.
  if (MVE_FIND_QUIETLY)
    message(STATUS "Failed to find mve - " ${REASON_MSG} ${ARGN})
  elseif (MVE_FIND_REQUIRED)
    message(FATAL_ERROR "Failed to find mve - " ${REASON_MSG} ${ARGN})
  else()
    # Neither QUIETLY nor REQUIRED, use no priority which emits a message
    # but continues configuration and allows generation.
    message("-- Failed to find mve - " ${REASON_MSG} ${ARGN})
  endif ()
endmacro(MVE_REPORT_NOT_FOUND)

# Handle possible presence of lib prefix for libraries on MSVC, see
# also MVE_RESET_FIND_LIBRARY_PREFIX().
if (MSVC)
  # Preserve the caller's original values for CMAKE_FIND_LIBRARY_PREFIXES
  # s/t we can set it back before returning.
  set(CALLERS_CMAKE_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
  # The empty string in this list is important, it represents the case when
  # the libraries have no prefix (shared libraries / DLLs).
  set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "" "${CMAKE_FIND_LIBRARY_PREFIXES}")
endif (MSVC)

# Search user-installed locations first, so that we prefer user installs
# to system installs where both exist.
list(APPEND MVE_CHECK_INCLUDE_DIRS
  /usr/local/include
  /usr/local/homebrew/include # Mac OS X
  /opt/local/var/macports/software # Mac OS X.
  /opt/local/include
  /usr/include)
# Windows (for C:/Program Files prefix).
list(APPEND MVE_CHECK_PATH_SUFFIXES
  mve/include
  mve/Include
  MVE/include
  MVE/Include
  )

list(APPEND MVE_CHECK_LIBRARY_DIRS
  /usr/local/lib
  /usr/local/homebrew/lib # Mac OS X.
  /opt/local/lib
  /usr/lib)
# Windows (for C:/Program Files prefix).
list(APPEND MVE_CHECK_LIBRARY_SUFFIXES
  mve/lib
  mve/Lib
  MVE/lib
  MVE/Lib
  )

# Search supplied hint directories first if supplied.
find_path(MVE_INCLUDE_DIR
  NAMES mve/camera.h
  PATHS ${MVE_INCLUDE_DIR_HINTS}
  ${MVE_CHECK_INCLUDE_DIRS}
  PATH_SUFFIXES ${MVE_CHECK_PATH_SUFFIXES})
if (NOT MVE_INCLUDE_DIR OR
    NOT EXISTS ${MVE_INCLUDE_DIR})
  mve_report_not_found(
    "Could not find mve include directory, set MVE_INCLUDE_DIR "
    "to directory containing mve/camera.h")
endif (NOT MVE_INCLUDE_DIR OR
       NOT EXISTS ${MVE_INCLUDE_DIR})

find_library(MVE_LIBRARY NAMES mve
  PATHS ${MVE_LIBRARY_DIR_HINTS}
  ${MVE_CHECK_LIBRARY_DIRS}
  PATH_SUFFIXES ${MVE_CHECK_LIBRARY_SUFFIXES})
if (NOT MVE_LIBRARY OR
    NOT EXISTS ${MVE_LIBRARY})
  mve_report_not_found(
    "Could not find mve library, set MVE_LIBRARY "
    "to full path to libmve.")
endif (NOT MVE_LIBRARY OR
       NOT EXISTS ${MVE_LIBRARY})

# Search for the MVE Util library
find_path(MVE_UTIL_INCLUDE_DIR
  NAMES util/file_system.h
  PATHS ${MVE_UTIL_INCLUDE_DIR_HINTS}
  ${MVE_CHECK_INCLUDE_DIRS}
  PATH_SUFFIXES ${MVE_CHECK_PATH_SUFFIXES})
if (NOT MVE_UTIL_INCLUDE_DIR OR
    NOT EXISTS ${MVE_UTIL_INCLUDE_DIR})
  mve_report_not_found(
    "Could not find mve util include directory, set MVE_UTIL_INCLUDE_DIR "
    "to directory containing mve/file_system.h")
endif (NOT MVE_UTIL_INCLUDE_DIR OR
       NOT EXISTS ${MVE_UTIL_INCLUDE_DIR})

find_library(MVE_UTIL_LIBRARY NAMES mve_util
  PATHS ${MVE_UTIL_LIBRARY_DIR_HINTS}
  ${MVE_CHECK_LIBRARY_DIRS}
  PATH_SUFFIXES ${MVE_CHECK_LIBRARY_SUFFIXES})
if (NOT MVE_UTIL_LIBRARY OR
    NOT EXISTS ${MVE_UTIL_LIBRARY})
  mve_report_not_found(
    "Could not find mve util library, set MVE_UTIL_LIBRARY "
    "to full path to libmve_util.")
endif (NOT MVE_UTIL_LIBRARY OR
       NOT EXISTS ${MVE_UTIL_LIBRARY})

#set (CMAKE_SHARED_LINKER_FLAGS "-lpng -ltiff -ljpeg -lpthread")

# Add libping, libtiff, and libjpeg
find_package(TIFF REQUIRED)
find_package(JPEG)
find_package(PNG)

# Mark internally as found, then verify. MVE_REPORT_NOT_FOUND() unsets
# if called.
set(MVE_FOUND TRUE)

# MVE does not seem to provide any record of the version in its
# source tree, thus cannot extract version.

# Catch case when caller has set MVE_INCLUDE_DIR in the cache / GUI and
# thus FIND_[PATH/LIBRARY] are not called, but specified locations are
# invalid, otherwise we would report the library as found.
if (NOT MVE_INCLUDE_DIR OR
    NOT EXISTS ${MVE_INCLUDE_DIR}/mve/camera.h)
  mve_report_not_found(
    "Caller defined MVE_INCLUDE_DIR:"
    " ${MVE_INCLUDE_DIR} does not contain mve/camera.h header.")
endif (NOT MVE_INCLUDE_DIR OR
       NOT EXISTS ${MVE_INCLUDE_DIR}/mve/camera.h)
# TODO: This regex for mve library is pretty primitive, we use lowercase
#       for comparison to handle Windows using CamelCase library names, could
#       this check be better?
string(TOLOWER "${MVE_LIBRARY}" LOWERCASE_MVE_LIBRARY)
if (NOT MVE_LIBRARY OR
    NOT "${LOWERCASE_MVE_LIBRARY}" MATCHES ".*mve[^/]*")
  mve_report_not_found(
    "Caller defined MVE_LIBRARY: "
    "${MVE_LIBRARY} does not match mve.")
endif (NOT MVE_LIBRARY OR
       NOT "${LOWERCASE_MVE_LIBRARY}" MATCHES ".*mve[^/]*")

# Set standard CMake FindPackage variables if found.
if (MVE_FOUND)
  list(APPEND MVE_INCLUDE_DIRS
    ${MVE_INCLUDE_DIR}
    ${MVE_UTIL_INCLUDE_DIR}
    ${TIFF_INCLUDE_DIR}
    ${JPEG_INCLUDE_DIR}
    ${PNG_INCLUDE_DIR})
  list(APPEND MVE_LIBRARIES
    ${MVE_LIBRARY}
    ${MVE_UTIL_LIBRARY}
    ${TIFF_LIBRARY}
    ${JPEG_LIBRARY}
    ${PNG_LIBRARY})
endif (MVE_FOUND)

mve_reset_find_library_prefix()

# Handle REQUIRED / QUIET optional arguments.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MVE DEFAULT_MSG
  MVE_INCLUDE_DIRS MVE_LIBRARIES)

# Only mark internal variables as advanced if we found mve, otherwise
# leave them visible in the standard GUI for the user to set manually.
if (MVE_FOUND)
  mark_as_advanced(FORCE MVE_INCLUDE_DIR
                         MVE_LIBRARY)
endif (MVE_FOUND)
