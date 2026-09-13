@echo off
REM Builds a native installer for UltimateTickTackToe on Windows.
REM The Linux/macOS equivalent is build-installer.sh.
REM
REM Usage: build-installer.bat [type]      (type defaults to exe; msi also works)
REM
REM Requires a JDK 22+ (jpackage lives in the JDK) and the WiX Toolset v3,
REM which jpackage shells out to for exe/msi. jpackage cannot cross-compile:
REM a Windows installer must be built on Windows.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "APPNAME=UltimateTickTackToe"
set "MAINJAR=UltimateTickTackToe-1.0-SNAPSHOT.jar"
set "MAINCLASS=com.hottes.caleb.ultimateticktacktoe.Launcher"
set "TYPE=%~1"
if "%TYPE%"=="" set "TYPE=exe"

REM --- locate a JDK 22+ ----------------------------------------------------
set "JPACKAGE="
if defined JAVA_HOME if exist "%JAVA_HOME%\bin\jpackage.exe" set "JPACKAGE=%JAVA_HOME%\bin\jpackage.exe"
if not defined JPACKAGE (
  for /f "delims=" %%J in ('where jpackage 2^>nul') do if not defined JPACKAGE set "JPACKAGE=%%J"
)
if not defined JPACKAGE (
  for /d %%D in ("%ProgramFiles%\Java\jdk-*" "%ProgramFiles%\Eclipse Adoptium\jdk-*" "%ProgramFiles%\Microsoft\jdk-*") do (
    if exist "%%D\bin\jpackage.exe" set "JPACKAGE=%%D\bin\jpackage.exe"
  )
)
if not defined JPACKAGE (
  echo error: jpackage not found. Install a JDK 22+ and set JAVA_HOME.
  exit /b 1
)
for /f "tokens=1 delims=." %%V in ('"%JPACKAGE%" --version 2^>nul') do set "JVER=%%V"
if defined JVER if %JVER% LSS 22 (
  echo error: the build targets Java 22 but "%JPACKAGE%" is version %JVER%.
  echo        Point JAVA_HOME at a JDK 22 or newer.
  exit /b 1
)
for %%P in ("%JPACKAGE%") do set "JBIN=%%~dpP"
set "JAVA_HOME=%JBIN:~0,-5%"

REM --- locate WiX v3 (candle/light), needed for exe and msi ----------------
where candle >nul 2>&1
if errorlevel 1 (
  if defined WIX if exist "%WIX%bin\candle.exe" set "PATH=%WIX%bin;%PATH%"
)
where candle >nul 2>&1
if errorlevel 1 (
  for /d %%D in ("%ProgramFiles(x86)%\WiX Toolset v3*" "%ProgramFiles%\WiX Toolset v3*") do (
    if exist "%%D\bin\candle.exe" set "PATH=%%D\bin;!PATH!"
  )
)
where candle >nul 2>&1
if errorlevel 1 (
  echo error: WiX Toolset v3 not found - jpackage needs candle.exe/light.exe for %TYPE%.
  echo        Install it with:  winget install WiXToolset.WiXToolset
  exit /b 1
)

echo === Gradle build (tests skipped) ===
call gradlew.bat installDist -x test --console=plain || exit /b 1

echo === jpackage --type %TYPE% ===
if exist "build\jpackage\%APPNAME%" rmdir /s /q "build\jpackage\%APPNAME%"
"%JPACKAGE%" ^
  --type %TYPE% ^
  --name "%APPNAME%" ^
  --app-version 1.0.0 ^
  --vendor "Caleb Hottes" ^
  --description "Ultimate Tic Tac Toe" ^
  --input "build\install\%APPNAME%\lib" ^
  --main-jar "%MAINJAR%" ^
  --main-class "%MAINCLASS%" ^
  --dest "build\jpackage" ^
  --win-dir-chooser ^
  --win-menu ^
  --win-shortcut ^
  --win-shortcut-prompt || exit /b 1

echo === Done. Output in build\jpackage ===
dir /b "build\jpackage"
endlocal
