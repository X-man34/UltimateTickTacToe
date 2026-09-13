#!/usr/bin/env bash
# Builds a native installer for UltimateTickTackToe on Linux or macOS.
# The Windows equivalent is build-installer.bat.
#
# Usage: ./build-installer.sh [type]
#   type defaults to deb/rpm on Linux and dmg on macOS.
#
# Requires a JDK 22+ (jpackage lives in the JDK). jpackage cannot cross-compile:
# a Linux installer must be built on Linux, a macOS one on macOS.

set -euo pipefail
cd "$(dirname "$0")"

APPNAME="UltimateTickTackToe"
MAINJAR="UltimateTickTackToe-1.0-SNAPSHOT.jar"
MAINCLASS="com.hottes.caleb.ultimateticktacktoe.Launcher"

# --- locate jpackage -----------------------------------------------------
if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/jpackage" ]; then
  JPACKAGE="$JAVA_HOME/bin/jpackage"
elif command -v jpackage >/dev/null 2>&1; then
  JPACKAGE="$(command -v jpackage)"
else
  echo "error: jpackage not found. Install a JDK 22+ and set JAVA_HOME." >&2
  exit 1
fi

JVER=$("$JPACKAGE" --version 2>/dev/null | cut -d. -f1)
if [ -n "$JVER" ] && [ "$JVER" -lt 22 ] 2>/dev/null; then
  echo "error: the build targets Java 22 but $JPACKAGE is version $JVER." >&2
  echo "       Point JAVA_HOME at a JDK 22 or newer." >&2
  exit 1
fi

# --- pick a package type -------------------------------------------------
OS="$(uname -s)"
TYPE="${1:-}"
EXTRA=()
case "$OS" in
  Linux)
    if [ -z "$TYPE" ]; then
      if command -v dpkg-deb >/dev/null 2>&1; then TYPE=deb; else TYPE=rpm; fi
    fi
    # deb needs dpkg-deb and fakeroot; rpm needs rpmbuild
    if [ "$TYPE" = deb ] && ! command -v dpkg-deb >/dev/null 2>&1; then
      echo "error: --type deb needs dpkg-deb (apt install dpkg fakeroot)." >&2; exit 1
    fi
    if [ "$TYPE" = rpm ] && ! command -v rpmbuild >/dev/null 2>&1; then
      echo "error: --type rpm needs rpmbuild (install rpm-build)." >&2; exit 1
    fi
    EXTRA=(--linux-shortcut)
    ;;
  Darwin)
    [ -z "$TYPE" ] && TYPE=dmg
    ;;
  *)
    echo "error: unsupported OS '$OS'. Use build-installer.bat on Windows." >&2
    exit 1
    ;;
esac

echo "=== Gradle build (tests skipped) ==="
./gradlew installDist -x test --console=plain

echo "=== jpackage --type $TYPE ==="
rm -rf "build/jpackage/$APPNAME"
"$JPACKAGE" \
  --type "$TYPE" \
  --name "$APPNAME" \
  --app-version 1.0.0 \
  --vendor "Caleb Hottes" \
  --description "Ultimate Tic Tac Toe" \
  --input "build/install/$APPNAME/lib" \
  --main-jar "$MAINJAR" \
  --main-class "$MAINCLASS" \
  --dest build/jpackage \
  "${EXTRA[@]}"

echo "=== Done. Output in build/jpackage ==="
ls -1 build/jpackage
