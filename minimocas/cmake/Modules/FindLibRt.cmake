# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindLibRt
# --------
#
# Find the native realtime includes and library.
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``LIBRT::LIBRT``, if
# LIBRT has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``LIBRT_FOUND``
#   True if RealTime library found
#
# ``LIBRT_INCLUDE_DIRS``
#   Location of time.h and other LibRT headers
#
# ``LIBRT_LIBRARIES``
#   List of libraries to link with when using LibRT
#
# Hints
# ^^^^^
#
#  ``LIBRT_ROOT``
#    Set this variable to a directory that contains a LibRT installation.

include(FindPackageHandleStandardArgs)

find_path(LIBRT_INCLUDE_DIRS
  NAMES time.h
  PATHS ${LIBRT_ROOT}/include/
)

find_library(LIBRT_LIBRARIES rt)

find_package_handle_standard_args(LibRt DEFAULT_MSG LIBRT_LIBRARIES LIBRT_INCLUDE_DIRS)

mark_as_advanced(LIBRT_INCLUDE_DIRS LIBRT_LIBRARIES)

if(LIBRT_FOUND)
    if(NOT TARGET LIBRT::LIBRT)
      add_library(LIBRT::LIBRT UNKNOWN IMPORTED)
      set_target_properties(LIBRT::LIBRT PROPERTIES
        IMPORTED_LOCATION "${LIBRT_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBRT_INCLUDE_DIRS}")
    endif()
endif()
