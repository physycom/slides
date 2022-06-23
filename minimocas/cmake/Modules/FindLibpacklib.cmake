# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindLibpacklib
# ------------
#
# Find the Libpacklib library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``LIBPACKLIB_FOUND``
#   True if Libpacklib library found
#
# ``LIBPACKLIB_LIBRARIES``
#   List of libraries to link with when using Libpacklib
#

include(FindPackageHandleStandardArgs)

find_path(LIBPACKLIB_INCLUDE_DIR cspack.h )
find_library(LIBPACKLIB_LIBRARY NAMES packlib PATH_SUFFIXES lib64 libx32)
find_package_handle_standard_args(Libpacklib DEFAULT_MSG LIBPACKLIB_LIBRARY LIBPACKLIB_INCLUDE_DIR)
mark_as_advanced(LIBPACKLIB_INCLUDE_DIR LIBPACKLIB_LIBRARY )

set(LIBPACKLIB_INCLUDE_DIRS ${LIBPACKLIB_INCLUDE_DIR} )
set(LIBPACKLIB_LIBRARIES ${LIBPACKLIB_LIBRARY} )
