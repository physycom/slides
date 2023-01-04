# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindKML
# --------
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project::
#
#  ``KML_FOUND``
#    True if LibKML found on the local system
#
#  ``KML_INCLUDE_DIRS``
#    Location of LibKML header files.
#
#  ``KML_LIBRARIES``
#    The KML libraries.
#
# Hints
# ^^^^^
#
#  ``KML_ROOT_DIR``
#    Set this variable to a directory that contains a LibKML installation.
#

include(FindPackageHandleStandardArgs)

find_path(KML_INCLUDE_DIR
  NAMES kml/dom.h
  HINTS ${KML_ROOT_DIR}/include ${CMAKE_BINARY_DIR}/lib
)

if(KML_INCLUDE_DIR)
  set(KML_INCLUDE_DIRS
    ${KML_INCLUDE_DIR}
    ${KML_INCLUDE_DIR}/kml/third_party/boost_1_34_1
    ${KML_INCLUDE_DIR}/kml/third_party/expat.src
  )
endif()

find_library(KML_BASE_LIBRARY
  NAMES kmlbase
  PATHS ${KML_ROOT_DIR}/lib ${CMAKE_BINARY_DIR}/lib
)

find_library(KML_CONVENIENCE_LIBRARY
  NAMES kmlconvenience
  PATHS ${KML_ROOT_DIR}/lib ${CMAKE_BINARY_DIR}/lib
)

find_library(KML_DOM_LIBRARY
  NAMES kmldom
  PATHS ${KML_ROOT_DIR}/lib ${CMAKE_BINARY_DIR}/lib
)

find_library(KML_ENGINE_LIBRARY
  NAMES kmlengine
  PATHS ${KML_ROOT_DIR}/lib ${CMAKE_BINARY_DIR}/lib
)

find_library(KML_REGIONATOR_LIBRARY
  NAMES kmlregionator
  PATHS ${KML_ROOT_DIR}/lib ${CMAKE_BINARY_DIR}/lib
)

find_library(KML_XSD_LIBRARY
  NAMES kmlxsd
  PATHS ${KML_ROOT_DIR}/lib ${CMAKE_BINARY_DIR}/lib
)

find_library(EXPAT_LIBRARY
  NAMES expat
  PATHS ${KML_ROOT_DIR}/lib ${CMAKE_BINARY_DIR}/lib
  NO_DEFAULT_PATH
)

# if not found try to use system expat
if(NOT EXPAT_LIBRARY)
  find_library(EXPAT_LIBRARY expat)
endif()

if(KML_BASE_LIBRARY AND KML_CONVENIENCE_LIBRARY AND KML_DOM_LIBRARY AND KML_ENGINE_LIBRARY AND KML_REGIONATOR_LIBRARY AND KML_XSD_LIBRARY AND EXPAT_LIBRARY)
  set(KML_LIBRARIES ${KML_BASE_LIBRARY} ${KML_CONVENIENCE_LIBRARY} ${KML_DOM_LIBRARY} ${KML_ENGINE_LIBRARY} ${KML_REGIONATOR_LIBRARY} ${KML_XSD_LIBRARY} ${EXPAT_LIBRARY})
endif()

find_package_handle_standard_args(KML
  FOUND_VAR
    KML_FOUND
  REQUIRED_VARS
    KML_INCLUDE_DIRS
    KML_LIBRARIES
)

mark_as_advanced(
  KML_INCLUDE_DIR
  KML_LIBRARY
  EXPAT_LIBRARY
)
