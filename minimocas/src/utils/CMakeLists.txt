cmake_minimum_required (VERSION 3.1)
project(utils)

if (EXISTS "$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
   message(STATUS "Enabling physycom settings")
   include("$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
else()
   message(STATUS "Unable to find physycom settings file")
endif()

find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})

add_executable(test_gnssconv     ${CMAKE_SOURCE_DIR}/test/test_gnssconv.cpp)
add_executable(test_split        ${CMAKE_SOURCE_DIR}/test/test_split.cpp)
add_executable(test_combinatoric ${CMAKE_SOURCE_DIR}/test/test_combinatoric.cpp)
add_executable(test_histo        ${CMAKE_SOURCE_DIR}/test/test_histo.cpp)

install (TARGETS
    test_gnssconv
    test_split
    test_combinatoric
    test_histo
    DESTINATION ${CMAKE_SOURCE_DIR}/bin/
)
