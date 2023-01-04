# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindUriParser
# ------------
#
# Find the UriParser includes and library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``URIPARSER_FOUND``
#   True if UriParser library found
#
# ``URIPARSER_INCLUDE_DIRS``
#   Location of UriParser headers
#
# ``URIPARSER_LIBRARIES``
#   List of libraries to link with when using UriParser
#

include(FindPackageHandleStandardArgs)

find_path(URIPARSER_INCLUDE_DIR NAMES uriparser/Uri.h uriparser/UriBase.h )
find_library(URIPARSER_LIBRARY NAMES uriparser)

find_package_handle_standard_args(URIPARSER DEFAULT_MSG URIPARSER_LIBRARY URIPARSER_INCLUDE_DIR)

mark_as_advanced(URIPARSER_INCLUDE_DIR URIPARSER_LIBRARY)

if(URIPARSER_FOUND)
  set(URIPARSER_INCLUDE_DIRS ${URIPARSER_INCLUDE_DIR})
  set(URIPARSER_LIBRARIES    ${URIPARSER_LIBRARY})
else()
  set(URIPARSER_INCLUDE_DIRS)
  set(URIPARSER_LIBRARIES)
endif()
