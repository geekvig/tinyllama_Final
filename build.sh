#!/usr/bin/env bash
# ==================================================================
#  build.sh -- build everything with g++ (Linux, WSL, macOS w/ g++,
#  or the MSYS2 UCRT64 shell on Windows).   Usage:  ./build.sh
#  Optional:  ./build.sh asan   -> also build AddressSanitizer tests
#             ./build.sh tsan   -> also build ThreadSanitizer tests
#  (sanitizers need Linux or WSL; MinGW g++ on Windows lacks them)
# ==================================================================
set -e
cd "$(dirname "$0")"
mkdir -p build
COMMON="-std=c++17 -Wall -Wextra -Iinclude -pthread"
REL="-O2 -march=native"
DBG="-O0 -g3 -D_GLIBCXX_ASSERTIONS"

echo "[1/4] tinyllama (release)";    g++ $COMMON $REL src/main.cpp       -o build/tinyllama.exe
echo "[2/4] tinyllama_dbg (debug)";  g++ $COMMON $DBG src/main.cpp       -o build/tinyllama_dbg.exe
echo "[3/4] run_tests (release)";    g++ $COMMON $REL tests/run_tests.cpp -o build/run_tests.exe
echo "[4/4] run_tests_dbg (debug)";  g++ $COMMON $DBG tests/run_tests.cpp -o build/run_tests_dbg.exe

if [ "$1" = "asan" ]; then
  echo "[+] run_tests_asan (AddressSanitizer + UBSan)"
  g++ $COMMON -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer tests/run_tests.cpp -o build/run_tests_asan.exe
fi
if [ "$1" = "tsan" ]; then
  echo "[+] run_tests_tsan (ThreadSanitizer)"
  g++ $COMMON -O1 -g -fsanitize=thread tests/run_tests.cpp -o build/run_tests_tsan.exe
fi
echo
echo "Build OK. Now run:   ./build/run_tests.exe"
