@echo off
REM ==================================================================
REM  build.bat -- build everything with MSYS2 UCRT64 g++ (Windows)
REM  Run from the project root, in cmd or PowerShell:   build.bat
REM  Produces build\tinyllama.exe        (fast, -O2)
REM           build\tinyllama_dbg.exe    (debuggable, -O0 -g3)
REM           build\run_tests.exe / build\run_tests_dbg.exe
REM ==================================================================
set PATH=C:\msys64\ucrt64\bin;%PATH%
if not exist build mkdir build

set COMMON=-std=c++17 -Wall -Wextra -Iinclude -pthread
set REL=-O2 -march=native
set DBG=-O0 -g3 -D_GLIBCXX_ASSERTIONS

echo [1/4] tinyllama.exe (release)
g++ %COMMON% %REL% src\main.cpp -o build\tinyllama.exe || goto :fail
echo [2/4] tinyllama_dbg.exe (debug)
g++ %COMMON% %DBG% src\main.cpp -o build\tinyllama_dbg.exe || goto :fail
echo [3/4] run_tests.exe (release)
g++ %COMMON% %REL% tests\run_tests.cpp -o build\run_tests.exe || goto :fail
echo [4/4] run_tests_dbg.exe (debug)
g++ %COMMON% %DBG% tests\run_tests.cpp -o build\run_tests_dbg.exe || goto :fail

echo.
echo Build OK. Now run:   build\run_tests.exe
exit /b 0
:fail
echo.
echo BUILD FAILED -- read the first error message above (scroll up).
exit /b 1
