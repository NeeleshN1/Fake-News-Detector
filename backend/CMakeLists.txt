cmake_minimum_required(VERSION 3.16)
project(fake_news_detector LANGUAGES CXX)

# -------------------------
# Compiler settings
# -------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fix for Windows MSBuild PowerShell bug
set(CMAKE_VS_GLOBALS "UsePowerShell=false")

# -------------------------
# ONNX Runtime
#   Point ORT_ROOT at a prebuilt ONNX Runtime release dir containing include/ and lib/.
#   Default: backend/third_party/onnxruntime  (override with -DORT_ROOT=...).
# -------------------------
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root")

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ORT_ROOT}/include" "${ORT_ROOT}/include/onnxruntime"
          "${ORT_ROOT}/include/onnxruntime/core/session")
find_library(ORT_LIB onnxruntime HINTS "${ORT_ROOT}/lib")

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found under ORT_ROOT='${ORT_ROOT}'.\n"
        "Download a prebuilt release (https://github.com/microsoft/onnxruntime/releases) "
        "and set -DORT_ROOT=/path/to/onnxruntime-<os>-x64-<ver>.")
endif()
message(STATUS "ONNX Runtime include: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib:     ${ORT_LIB}")

include_directories(include ${ORT_INCLUDE_DIR})

# -------------------------
# Inference executable
# -------------------------
add_executable(fake_news
    src/main.cpp
    src/classifier.cpp
    src/tokenizer.cpp
    src/onnx_model.cpp
)

target_link_libraries(fake_news ${ORT_LIB})

# Copy the ONNX Runtime shared library next to the binary on Windows so it runs in place.
if(WIN32)
    file(GLOB ORT_DLLS "${ORT_ROOT}/lib/*.dll")
    foreach(_dll ${ORT_DLLS})
        add_custom_command(TARGET fake_news POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:fake_news>")
    endforeach()
endif()
