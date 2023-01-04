# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindSQLite3
# ------------
#
# Find the SQLite3 includes and library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``SQLITE3_FOUND``
#   True if SQLite3 library found
#
# ``SQLITE3_INCLUDE_DIRS``
#   Location of SQLite3 headers
#
# ``SQLITE3_LIBRARIES``
#   List of libraries to link with when using SQLite3
#

include(FindPackageHandleStandardArgs)
include(SelectLibraryConfigurations)

find_path(SQLITE3_INCLUDE_DIR NAMES sqlite3.h)

if(NOT SQLITE3_LIBRARY)
  find_library(SQLITE3_LIBRARY_RELEASE NAMES sqlite3)
  find_library(SQLITE3_LIBRARY_DEBUG NAMES sqlite3d)
  select_library_configurations(SQLITE3)
endif()

find_package_handle_standard_args(SQLITE3 DEFAULT_MSG SQLITE3_LIBRARY SQLITE3_INCLUDE_DIR)
mark_as_advanced(SQLITE3_INCLUDE_DIR SQLITE3_LIBRARY)

if(SQLITE3_FOUND)
  set(SQLITE3_LIBRARIES ${SQLITE3_LIBRARY})
  set(SQLITE3_INCLUDE_DIRS ${SQLITE3_INCLUDE_DIR})
else()
  set(SQLITE3_LIBRARIES)
  set(SQLITE3_INCLUDE_DIRS)
endif()
