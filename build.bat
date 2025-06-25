@echo off
setlocal

REM -------------------------------------------------------------------
REM build.bat — bootstrap VS env + run CMake in a plain cmd.exe
REM -------------------------------------------------------------------

REM 1) Find VS install path via VSINSTALLDIR or vswhere
if NOT DEFINED VSINSTALLDIR (
  for /f "usebackq tokens=*" %%i in (`
    "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" ^
     -latest -products * ^
     -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
     -property installationPath
  `) do set "VSINSTALLDIR=%%i"
)

if NOT EXIST "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" (
  echo ERROR: Cannot find vcvars64.bat under "%VSINSTALLDIR%"
  pause
  exit /b 1
)

REM 2) Initialize the MSVC + Windows SDK environment
call "%VSINSTALLDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul

REM 3) Create build folder
if not exist build mkdir build
pushd build

REM 4) Configure with CMake
cmake -S .. -B . ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_SEPARABLE_COMPILATION=ON ^
  -DCMAKE_C_FLAGS="/MT" ^
  -DCMAKE_CXX_FLAGS="/MT"

if errorlevel 1 (
  echo.
  echo [ERROR] CMake configure failed
  popd
  pause
  exit /b 1
)

REM 5) Build only the MyApp target in parallel
cmake --build . --config Release --target WordCountCUDA -- /m

if errorlevel 1 (
  echo.
  echo [ERROR] Build failed
  popd
  pause
  exit /b 1
)

echo.
echo [SUCCESS] WordCountCUDA built in "%CD%\bin\Release\WordCountCUDA.exe"
popd
pause
endlocal
