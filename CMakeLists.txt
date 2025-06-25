cmake_minimum_required(VERSION 3.18)
project(WordCountCUDA LANGUAGES CXX CUDA)

# ----------------------------------------------------------------------------
# 1) Language standards
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD        17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS      OFF)
set(CMAKE_CUDA_STANDARD       17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS     OFF)

# ----------------------------------------------------------------------------
# 2) Output directory for the executable
# ----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ----------------------------------------------------------------------------
# 3) Build vendored libmarisa (static C++ library)
# ----------------------------------------------------------------------------
set(MARISA_ROOT "${CMAKE_SOURCE_DIR}/vendor/marisa")
if (NOT EXISTS "${MARISA_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR "libmarisa C++ subproject not found at ${MARISA_ROOT}")
endif()

set(BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
set(ENABLE_NATIVE_CODE ON  CACHE BOOL "" FORCE)
set(BUILD_TESTING      OFF CACHE BOOL "" FORCE)

add_subdirectory(
  ${MARISA_ROOT}
  ${CMAKE_BINARY_DIR}/vendor/marisa-build
  EXCLUDE_FROM_ALL
)

# ----------------------------------------------------------------------------
# 4) Find CUDA runtime
# ----------------------------------------------------------------------------
find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND AND TARGET CUDAToolkit::cudart)
  message(STATUS "Using CUDAToolkit::cudart")
  set(CUDA_RUNTIME_LIB CUDAToolkit::cudart)
else()
  message(STATUS "Falling back to FindCUDA and CUDA::cudart")
  find_package(CUDA REQUIRED)
  set(CUDA_RUNTIME_LIB CUDA::cudart)
endif()

# ----------------------------------------------------------------------------
# 5) Gather source files
# ----------------------------------------------------------------------------
file(GLOB_RECURSE HOST_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cu")

if (NOT HOST_SOURCES)
  message(FATAL_ERROR "No C++ sources found in src/")
endif()

# ----------------------------------------------------------------------------
# 6) Define executable
# ----------------------------------------------------------------------------
add_executable(WordCountCUDA ${HOST_SOURCES} ${CUDA_SOURCES})

# ----------------------------------------------------------------------------
# 7) Include directories and compile properties
# ----------------------------------------------------------------------------
target_include_directories(WordCountCUDA
  PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_SOURCE_DIR}/src"
    "${MARISA_ROOT}/include"
)

set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ----------------------------------------------------------------------------
# 8) Link libraries
# ----------------------------------------------------------------------------
target_link_libraries(WordCountCUDA
  PRIVATE
    marisa
    ${CUDA_RUNTIME_LIB}
)

# ----------------------------------------------------------------------------
# 9) Windows-specific: static CRT + exception handling
# ----------------------------------------------------------------------------
if (MSVC)
  # Use static CRT
  foreach(flag_var
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_DEBUG
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_DEBUG)
    string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
  endforeach()

  # Properly isolate /EHsc handling per language
  target_compile_options(WordCountCUDA PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc>
  )

  # Force CUDA architectures
  set_target_properties(WordCountCUDA PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "52"
  )
endif()

