@echo off
REM Builds the Windows installer for UltimateTickTackToe.
REM Requires a JDK 22+ (for jpackage) and the WiX Toolset v3 on PATH for --type exe/msi.

setlocal
set "JAVA_HOME=C:\Program Files\Java\jdk-22"
set "JPACKAGE=%JAVA_HOME%\bin\jpackage.exe"
REM jpackage needs WiX v3 (candle/light) on PATH to build exe/msi installers
set "PATH=C:\Program Files (x86)\WiX Toolset v3.14\bin;%PATH%"
set "APPNAME=UltimateTickTackToe"
set "MAINJAR=UltimateTickTackToe-1.0-SNAPSHOT.jar"
set "MAINCLASS=com.hottes.caleb.ultimateticktacktoe.Launcher"
set "TYPE=%~1"
if "%TYPE%"=="" set "TYPE=exe"

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
