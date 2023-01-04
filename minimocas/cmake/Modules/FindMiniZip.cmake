# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindMiniZip
# ------------
#
# Find the MiniZip includes and library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``MINIZIP_FOUND``
#   True if MiniZip library found
#
# ``MINIZIP_INCLUDE_DIRS``
#   Location of MiniZip headers
#
# ``MINIZIP_LIBRARIES``
#   List of libraries to link with when using MiniZip
#

include(FindPackageHandleStandardArgs)

find_path(MINIZIP_INCLUDE_DIR NAMES minizip/unzip.h minizip/zip.h)
find_library(MINIZIP_LIBRARY NAMES minizip)

find_package_handle_standard_args(MINIZIP DEFAULT_MSG MINIZIP_LIBRARY MINIZIP_INCLUDE_DIR )

mark_as_advanced(MINIZIP_INCLUDE_DIR MINIZIP_LIBRARY)

if(MINIZIP_FOUND)
  set(MINIZIP_INCLUDE_DIRS ${MINIZIP_INCLUDE_DIR})
  set(MINIZIP_LIBRARIES    ${MINIZIP_LIBRARY})
else()
  set(MINIZIP_INCLUDE_DIRS)
  set(MINIZIP_LIBRARIES)
endif()
