# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindLibphtools
# ------------
#
# Find the Libphtools library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``LIBPHTOOLS_FOUND``
#   True if Libphtools library found
#
# ``LIBPHTOOLS_LIBRARIES``
#   List of libraries to link with when using Libphtools
#

include(FindPackageHandleStandardArgs)

find_library(LIBPHTOOLS_LIBRARY NAMES phtools PATH_SUFFIXES lib64 libx32 /usr/lib/x86_64-linux-gnu)
find_package_handle_standard_args(Libphtools DEFAULT_MSG LIBPHTOOLS_LIBRARY)
mark_as_advanced(LIBPHTOOLS_LIBRARY )

set(LIBPHTOOLS_LIBRARIES ${LIBPHTOOLS_LIBRARY} )
