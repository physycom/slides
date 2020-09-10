cmake_minimum_required (VERSION 3.11)
project(slides)

if (EXISTS "$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
  message(STATUS "Enabling physycom settings")
  include("$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
else()
  message(STATUS "Unable to find physycom settings file")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(minimocas/CMakeLists.txt)
add_subdirectory($ENV{WORKSPACE}/pybind11 pybind11)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/lib
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/jsoncons/include
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/utils
)

set(WORKSPACE $ENV{WORKSPACE})
string(REPLACE "\\" "/" WORKSPACE ${WORKSPACE})

pybind11_add_module(pysim
  ${CMAKE_CURRENT_LIST_DIR}/src/slides_python_interface.cpp
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/lib/carto.cpp
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/lib/carto_bestpath.cpp
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/lib/carto_edit.cpp
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/lib/simulation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/lib/simulation_io.cpp
  ${CMAKE_CURRENT_LIST_DIR}/minimocas/src/lib/pawn.cpp
)
target_link_libraries(pysim PRIVATE
  Boost::system
  Boost::filesystem
)
install(TARGETS pysim DESTINATION ${CMAKE_CURRENT_LIST_DIR}/bin)

add_executable(slides_tools
  ${CMAKE_CURRENT_LIST_DIR}/src/slides_tools.cpp
)
target_link_libraries(slides_tools
  carto
  simulation_deps
  Threads::Threads
  ${common_libs}
)

# CONFIG
list(APPEND cities bari dubrovnik ferrara sybenik venezia)
foreach(city ${cities})
  configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/vars/templates/${city}_template.json.in
    ${CMAKE_CURRENT_LIST_DIR}/vars/templates/${city}_template.json
  )
  # replace \ with / in windows path
  if (MSVC)
    file(READ ${CMAKE_CURRENT_LIST_DIR}/vars/templates/${city}_template.json TEMPLATE_SIM)
    string(REPLACE "\\" "/" TEMPLATE_SIM_ESC ${TEMPLATE_SIM} )
    file(WRITE ${CMAKE_CURRENT_LIST_DIR}/vars/templates/${city}_template.json ${TEMPLATE_SIM_ESC})
  endif()
endforeach()
