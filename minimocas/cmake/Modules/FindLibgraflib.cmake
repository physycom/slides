# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindLibgraflib
# ------------
#
# Find the Libgraflib library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``LIBGRAFLIB_FOUND``
#   True if Libgraflib library found
#
# ``LIBGRAFLIB_LIBRARIES``
#   List of libraries to link with when using Libgraflib
#

include(FindPackageHandleStandardArgs)

find_path(LIBGRAFLIB_INCLUDE_DIR graflib.h hplot.h)
find_library(LIBGRAFLIB_LIBRARY NAMES graflib)
find_package_handle_standard_args(Libgraflib DEFAULT_MSG LIBGRAFLIB_LIBRARY LIBGRAFLIB_INCLUDE_DIR)
mark_as_advanced(LIBGRAFLIB_INCLUDE_DIR LIBGRAFLIB_LIBRARY )

set(LIBGRAFLIB_INCLUDE_DIRS ${LIBGRAFLIB_INCLUDE_DIR} )
set(LIBGRAFLIB_LIBRARIES ${LIBGRAFLIB_LIBRARY} )
