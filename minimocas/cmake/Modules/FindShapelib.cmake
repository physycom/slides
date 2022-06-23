# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindShapelib
# ------------
#
# Find the Shapelib library.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
# ``Shapelib_FOUND``
#   True if Shapelib library found
#
#  ``Shapelib_INCLUDE_DIRS``
#    Location of FFTW header files.
#
# ``Shapelib_LIBRARIES``
#   List of libraries to link with when using Shapelib
#

find_path(Shapelib_INCLUDE_DIR NAMES shapefil.h)

if(MSVC)
    set(Shapelib_LIBNAME shapelib)
else()
    set(Shapelib_LIBNAME shp)
endif()

find_library(Shapelib_LIBRARY NAMES ${Shapelib_LIBNAME})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Shapelib
    FOUND_VAR Shapelib_FOUND
    REQUIRED_VARS Shapelib_INCLUDE_DIR Shapelib_LIBRARY
)

set(Shapelib_INCLUDE_DIRS ${Shapelib_INCLUDE_DIR})
set(Shapelib_LIBRARIES ${Shapelib_LIBRARY})

mark_as_advanced(Shapelib_INCLUDE_DIR Shapelib_LIBRARY)
