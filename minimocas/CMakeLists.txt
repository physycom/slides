cmake_minimum_required (VERSION 3.11)
project(minimocas)

if (EXISTS "$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
  message(STATUS "Enabling physycom settings")
  include("$ENV{WORKSPACE}/sysconfig/cmake/physycom_config.cmake")
else()
  message(STATUS "Unable to find physycom settings file")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND FORCE_LLVM_CPP_STDLIB)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND ENABLE_PROFILING)
  message(STATUS "Enabled profiling options")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
  string(REPLACE "/INCREMENTAL" "" CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
endif()

if (UNIX AND NOT APPLE)
  find_package(LibRt REQUIRED)
endif()
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Boost 1.64.0 COMPONENTS system filesystem REQUIRED)
find_package(CURLpp)

# Search for OpenMP and use it
if ( (UNIX OR MSVC) AND NOT APPLE )
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

if(MSVC)
  add_definitions(-DNOMINMAX)
endif()

if(MSVC)
  #add_definitions(/WX)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
  set(BUILD_SHARED_LIBS TRUE)
else()
  add_definitions(-Wall -Wextra -Wpedantic)
  add_definitions(-Wno-reorder -Wno-implicit-fallthrough -Wno-unused-local-typedefs)
  add_definitions(-Wno-unused-parameter -Wno-unused-function)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9)
    add_definitions(-Wno-maybe-uninitialized)
  endif()
endif()

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src/
  ${CMAKE_CURRENT_LIST_DIR}/src/lib
  ${CMAKE_CURRENT_LIST_DIR}/src/app
  ${CMAKE_CURRENT_LIST_DIR}/src/utils
  ${CMAKE_CURRENT_LIST_DIR}/src/libtracpp
  ${CMAKE_CURRENT_LIST_DIR}/src/jsoncons/include
  ${Boost_INCLUDE_DIRS}
)

if(UNIX AND NOT APPLE)
  list(APPEND common_libs LIBRT::LIBRT)
endif()

if(UNIX)
  set(APPEND common_libs ${CMAKE_DL_LIBS})
endif()

add_library(carto SHARED
  ${CMAKE_CURRENT_LIST_DIR}/src/lib/carto.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/lib/carto_bestpath.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/lib/carto_edit.cpp
)

add_library(simulation_deps SHARED
  ${CMAKE_CURRENT_LIST_DIR}/src/lib/pawn.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/lib/simulation.cpp
  ${CMAKE_CURRENT_LIST_DIR}/src/lib/simulation_io.cpp
)
target_link_libraries(simulation_deps
  carto
  Boost::filesystem
)

if (CURLpp_FOUND)
  add_library(simulation_conn
    ${CMAKE_CURRENT_LIST_DIR}/src/lib/connection.cpp
  )

  target_link_libraries(simulation_conn
    ${CURLPP_LIBRARIES}
  )

  target_link_libraries(simulation_deps
    simulation_conn
  )
endif()

add_executable(sim_engine
  ${CMAKE_CURRENT_LIST_DIR}/src/app/sim_engine.cpp
)
target_link_libraries(sim_engine
  carto
  simulation_deps
  Threads::Threads
  ${common_libs}
)
