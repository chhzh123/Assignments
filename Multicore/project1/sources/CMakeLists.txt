cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

file(GLOB SOURCES
    "src/*.h"
    "src/*.cpp"
    "src/*.cu"
)

cuda_add_executable(
    main
    ${SOURCES}
)