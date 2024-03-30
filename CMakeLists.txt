cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)

project(dedup_proj)

set(SRCDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OBJDIR ${CMAKE_CURRENT_BINARY_DIR})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "Build type: Release")
else()
  if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "CMake build type is set to ${CMAKE_BUILD_TYPE}! This might result in bad performance!")
  else()
    message(STATUS "Build type: Release")
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
if (UNIX)
  set(CMAKE_C_STANDARD 99)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
  find_package(Threads REQUIRED)
endif()

if (MSVC)
    # Disabling Incremental Linking might be useful for example to use SizeBench to analyze compiled binary size contributions from different parts of the code
    add_link_options($<$<CONFIG:RelWithDebInfo>:/INCREMENTAL:NO>)
    #add_link_options($<$<CONFIG:Debug>:/INCREMENTAL:NO>)

    if (WIN32 AND NOT MSVC_VERSION VERSION_LESS 142)
        # This should enable hot reload for VS2022
        add_link_options($<$<CONFIG:Debug>:/INCREMENTAL>)
        add_compile_options($<$<CONFIG:Debug>:/Zi>)
    endif()
endif()

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

include_directories(AFTER ${SRCDIR})

function(add_stem2file VAR pattern stem)
    set(TMPV ${${VAR}})
    foreach(IND ${stem})
        string(REGEX REPLACE "^[ ]*([0-9a-zA-Z_%]+)[ ]*$" \\1 IND ${IND})
        string(REPLACE "%STEM%" ${IND} NEXT ${pattern})
        set(TMPV "${TMPV};${NEXT}")
    endforeach()
    set(${VAR} "${TMPV}" PARENT_SCOPE)
endfunction()

add_definitions(-DBUILD_LIB)

add_definitions(-DHAVE_BOOL)
if (MSVC)
  include_directories(AFTER "msinttypes")
  add_definitions(-D_UNICODE -DUNICODE)
endif (MSVC)

# Not the cleanest thing ever, but we just make Mac compilation use unix/linux code, for now it works
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions(-D__unix)
    add_definitions(-D__linux)
endif()

if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
  add_definitions(-DBIT64)
endif ("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")

set(DEDUP_PROJ_SRC "${SRCDIR}/dedup_proj.cpp")

add_executable(dedup_proj ${DEDUP_PROJ_SRC} )

install(TARGETS dedup_proj DESTINATION bin)
