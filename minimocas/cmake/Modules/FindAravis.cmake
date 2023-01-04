# Distributed under the OSI-approved BSD 3-Clause License.
# Copyright Stefano Sinigardi

#.rst:
# FindAravis
# --------
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
#  ``Aravis_FOUND``
#    True if Aravis is found on the local system
#
#  ``Aravis_INCLUDE_DIRS``
#    Location of Aravis header files.
#
#  ``Aravis_LIBRARIES``
#    The Aravis libraries.
#

include(FindPackageHandleStandardArgs)

if(NOT Aravis_INCLUDE_DIR)
  find_path(Aravis_INCLUDE_DIR arv.h
    PATH_SUFFIXES aravis-0.8)
endif()

if(NOT Aravis_LIBRARY)
  find_library(Aravis_LIBRARY aravis-0.8
    PATH_SUFFIXES x86_64-linux-gnu)
endif()

set(Aravis_INCLUDE_DIRS ${Aravis_INCLUDE_DIR})
set(Aravis_LIBRARIES ${Aravis_LIBRARY})
mark_as_advanced(Aravis_LIBRARY Aravis_INCLUDE_DIR)

if(EXISTS "${Aravis_CORE_INCLUDE_DIR}/arvversion.h")
  file(READ ${Aravis_CORE_INCLUDE_DIR}/arvversion.h ARVVERSION_HEADER_CONTENTS)
    string(REGEX MATCH "define ARAVIS_MAJOR_VERSION * +([0-9]+)"
                 Aravis_VERSION_MAJOR "${ARVVERSION_HEADER_CONTENTS}")
    string(REGEX REPLACE "define ARAVIS_MAJOR_VERSION * +([0-9]+)" "\\1"
                 Aravis_VERSION_MAJOR "${Aravis_VERSION_MAJOR}")

    string(REGEX MATCH "define ARAVIS_MINOR_VERSION * +([0-9]+)"
                 Aravis_VERSION_MINOR "${ARVVERSION_HEADER_CONTENTS}")
    string(REGEX REPLACE "define ARAVIS_MINOR_VERSION * +([0-9]+)" "\\1"
                 Aravis_VERSION_MINOR "${Aravis_VERSION_MINOR}")

    string(REGEX MATCH "define ARAVIS_MICRO_VERSION * +([0-9]+)"
                 Aravis_VERSION_PATCH "${ARVVERSION_HEADER_CONTENTS}")
    string(REGEX REPLACE "define ARAVIS_MICRO_VERSION * +([0-9]+)" "\\1"
                 Aravis_VERSION_PATCH "${Aravis_VERSION_PATCH}")

  if(NOT Aravis_VERSION_MAJOR)
    set(Aravis_VERSION "?")
  else()
    set(Aravis_VERSION "${Aravis_VERSION_MAJOR}.${Aravis_VERSION_MINOR}.${Aravis_VERSION_PATCH}")
  endif()
endif()

find_package_handle_standard_args(Aravis
      REQUIRED_VARS  Aravis_INCLUDE_DIR Aravis_LIBRARY
      VERSION_VAR    Aravis_VERSION
)

if( Aravis_FOUND AND NOT TARGET Aravis::Aravis )
  add_library( Aravis::Aravis      UNKNOWN IMPORTED )
  set_target_properties( Aravis::Aravis PROPERTIES
    IMPORTED_LOCATION                 "${Aravis_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES     "${Aravis_INCLUDE_DIR}"
    IMPORTED_LINK_INTERFACE_LANGUAGES "C" )
endif()
