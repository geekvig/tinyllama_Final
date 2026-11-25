cmake_minimum_required(VERSION 3.16)
project(tinyllama CXX)

# ------------------------------------------------------------------
# Optional: CMake build. build.bat / build.sh do the same with one g++
# command per program, which is easier to read when learning.
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # helps VS Code IntelliSense

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3 -D_GLIBCXX_ASSERTIONS")

option(TINYLLAMA_ASAN "Build with AddressSanitizer + UBSan (Linux/WSL)" OFF)
option(TINYLLAMA_TSAN "Build with ThreadSanitizer (Linux/WSL)" OFF)

find_package(Threads REQUIRED)

# Header-only engine: everything lives in include/.
add_library(tinyllama_lib INTERFACE)
target_include_directories(tinyllama_lib INTERFACE include)
target_link_libraries(tinyllama_lib INTERFACE Threads::Threads)
target_compile_options(tinyllama_lib INTERFACE -Wall -Wextra)
if(TINYLLAMA_ASAN)
    target_compile_options(tinyllama_lib INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(tinyllama_lib INTERFACE -fsanitize=address,undefined)
endif()
if(TINYLLAMA_TSAN)
    target_compile_options(tinyllama_lib INTERFACE -fsanitize=thread)
    target_link_options(tinyllama_lib INTERFACE -fsanitize=thread)
endif()

add_executable(tinyllama src/main.cpp)
target_link_libraries(tinyllama PRIVATE tinyllama_lib)

add_executable(run_tests tests/run_tests.cpp)
target_link_libraries(run_tests PRIVATE tinyllama_lib)

enable_testing()
add_test(NAME all_tests COMMAND run_tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
