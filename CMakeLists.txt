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

set(XXHASH_SRC "${SRCDIR}/contrib/xxHash/xxhash.c")
set(SSDEEP_SRC "${SRCDIR}/contrib/ssdeep/fuzzy.c")

#add_subdirectory(contrib/embeddings.cpp)
#set(BERT_SRC "${SRCDIR}/contrib/embeddings.cpp/bert.cpp")
#set(GGML_INCLUDE_DIR "${SRCDIR}/contrib/embeddings.cpp/ggml/include/ggml")
#include_directories(AFTER ${GGML_INCLUDE_DIR})
add_subdirectory(contrib/llama.cpp-b2632)
set(LLAMA_EXTRA_SRC "${SRCDIR}/contrib/llama.cpp-b2632/common/common.cpp")

# netlib blas
#add_subdirectory(contrib/BLAS)

# compiled openblas
#add_library(blas STATIC IMPORTED) # or STATIC instead of SHARED
#set_target_properties(blas PROPERTIES
#  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/contrib/OpenBLAS-0.3.26-x64/lib/libopenblas.lib"
#  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/contrib/OpenBLAS-0.3.26-x64/include"
#)
#set(BLAS_LIBRARIES "blas")
#set(NOFORTRAN True)
#add_subdirectory(contrib/OpenBLAS-0.3.26)
set(FAISS_ENABLE_PYTHON OFF)
set(FAISS_ENABLE_GPU OFF)
set(BUILD_TESTING OFF)
add_subdirectory(contrib/faiss)

set(DEDUP_PROJ_SRC "${SRCDIR}/dedup_proj.cpp")

add_executable(dedup_proj ${DEDUP_PROJ_SRC} ${XXHASH_SRC} ${SSDEEP_SRC} ${LLAMA_EXTRA_SRC} )
#target_link_libraries(dedup_proj bert faiss)
target_link_libraries(dedup_proj llama build_info faiss)

install(TARGETS dedup_proj DESTINATION bin)
