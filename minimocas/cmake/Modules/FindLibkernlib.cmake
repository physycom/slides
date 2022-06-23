# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindLibkernlib
# ------------
#
# Find the Libkernlib library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``LIBKERNLIB_FOUND``
#   True if Libkernlib library found
#
# ``LIBKERNLIB_LIBRARIES``
#   List of libraries to link with when using Libkernlib
#

include(FindPackageHandleStandardArgs)

find_path(LIBKERNLIB_INCLUDE_DIR kernlib.h )
find_library(LIBKERNLIB_LIBRARY NAMES kernlib PATH_SUFFIXES lib64 libx32)
find_package_handle_standard_args(Libkernlib DEFAULT_MSG LIBKERNLIB_LIBRARY LIBKERNLIB_INCLUDE_DIR)
mark_as_advanced(LIBKERNLIB_INCLUDE_DIR LIBKERNLIB_LIBRARY )

set(LIBKERNLIB_INCLUDE_DIRS ${LIBKERNLIB_INCLUDE_DIR} )
set(LIBKERNLIB_LIBRARIES ${LIBKERNLIB_LIBRARY} )
