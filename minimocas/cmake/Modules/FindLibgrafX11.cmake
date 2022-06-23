# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindLibgrafX11
# ------------
#
# Find the LibgrafX11 library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``LIBGRAFX11_FOUND``
#   True if LibgrafX11 library found
#
# ``LIBGRAFX11_LIBRARIES``
#   List of libraries to link with when using LibgrafX11
#

include(FindPackageHandleStandardArgs)

find_path(LIBGRAFX11_INCLUDE_DIR higz.h )
find_library(LIBGRAFX11_LIBRARY NAMES grafX11)
find_package_handle_standard_args(LibgrafX11 DEFAULT_MSG LIBGRAFX11_LIBRARY LIBGRAFX11_INCLUDE_DIR)
mark_as_advanced(LIBGRAFX11_INCLUDE_DIR LIBGRAFX11_LIBRARY )

set(LIBGRAFX11_INCLUDE_DIRS ${LIBGRAFX11_INCLUDE_DIR} )
set(LIBGRAFX11_LIBRARIES ${LIBGRAFX11_LIBRARY} )
