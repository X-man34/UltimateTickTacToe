# Ultimate Tic Tac Toe

A JavaFX desktop implementation of [Ultimate Tic Tac Toe](https://en.wikipedia.org/wiki/Ultimate_tic-tac-toe)
with a Monte Carlo Tree Search (MCTS) opponent that can be guided by trained
value and policy neural networks.

## The game

Ultimate Tic Tac Toe is played on a 3x3 grid of 3x3 boards. Winning a small
board claims that cell of the big board, and the cell you play in dictates
which small board your opponent must play in next. Win three small boards in a
row to win the game.

## Features

- **Human vs. human, human vs. computer, or computer vs. computer.** Each of
  the two player slots is configured independently.
- **MCTS engine** with multithreaded rollouts and a configurable time budget.
- **Neural-network guidance.** A value network and a policy network
  (`valueNetworkV1_1.zip` / `policyNetworkV1_0.zip`, bundled as resources) are
  trained with Deeplearning4j and used by the MEDIUM difficulty preset.
- **Three presets — EASY, MEDIUM, HARD** — plus a **Custom** mode exposing the
  raw engine knobs: UCB C value, compute time, thread count, and whether
  force-play is allowed. See [MCTS_ANALYSIS.md](MCTS_ANALYSIS.md) for how the
  search uses them.
- **Saveable engine configurations.** An `EvaluatorConfiguration` serialises to
  a zip containing a JSON descriptor plus any neural nets it references; see
  the prebuilt ones in `evaluators/`.

## Requirements

- **JDK 22 or newer** (the build targets Java 22, and `jpackage` ships in the JDK).
- Windows, Linux or macOS — see [Platform targeting](#platform-targeting).
- Extra tooling for building an installer, depending on the host:

  | Host    | Package types | Also needs                                                                         |
  | ------- | ------------- | ---------------------------------------------------------------------------------- |
  | Windows | `exe`, `msi`  | [WiX Toolset v3](https://wixtoolset.org/) (`winget install WiXToolset.WiXToolset`) |
  | Linux   | `deb`         | `dpkg-deb` and `fakeroot`                                                          |
  | Linux   | `rpm`         | `rpmbuild`                                                                         |
  | macOS   | `dmg`, `pkg`  | Xcode command line tools                                                           |

## Running from source

```
gradlew.bat run     # Windows
./gradlew run       # Linux / macOS
```

## Building an installer

```
build-installer.bat          REM Windows: .exe (default), or pass msi
./build-installer.sh         # Linux: .deb (default), or pass rpm
                             # macOS: .dmg (default), or pass pkg
```

Both scripts do the same thing: run `installDist -x test`, then `jpackage`,
writing the result to `build/jpackage/`. Each locates the JDK and any needed
packaging tools automatically and fails with an actionable message if
something is missing; set `JAVA_HOME` to pick a specific JDK.

The output is self-contained — it bundles a Java runtime, so end users do not
need a JDK — and weighs roughly 300-340 MB. The Windows installer offers an
install-directory chooser, a Start menu entry and an optional desktop
shortcut; the Linux packages install a desktop shortcut.

**`jpackage` cannot cross-compile.** A Windows installer must be built on
Windows, a `.deb` on Linux, a `.dmg` on macOS. There is no way to produce all
three from one machine short of a CI matrix or a VM.

To build just the unpacked application directory instead of an installer, use
`jpackage --type app-image`.

## Platform targeting

Left to their defaults, the `nd4j` and `javacpp` dependencies pull native
binaries for *every* OS and architecture — Linux, macOS, Android, iOS and
Windows — which added about 1.2 GB of jars. `build.gradle` therefore pins the
build to exactly one platform, in two places:

- `ext.javacppPlatform`, honoured by the `org.bytedeco.gradle-javacpp-platform`
  plugin, which rewrites every bytedeco `*-platform` dependency to that single
  platform.
- `nd4j-native`, declared directly with the same string as its classifier,
  because `nd4j-native-platform` is an `org.nd4j` artifact that the plugin
  above does not rewrite.

Both default to **the platform you are building on** (`windows-x86_64`,
`linux-x86_64`, `macosx-arm64`, ...), which is what you want, since `jpackage`
cannot cross-compile anyway. Override it if you need to:

```
./gradlew installDist -PjavacppPlatform=linux-x86_64
```

Without a matching native backend ND4J fails at startup, so only override this
if you know the target has one.

Not every bytedeco preset publishes a binary for every platform at these
(2022-era) versions, so unusual targets may fail to resolve. **Apple Silicon
is the most likely to hit this**: if a `macosx-arm64` artifact is missing,
build with `-PjavacppPlatform=macosx-x86_64` and run under Rosetta. The
Windows and Linux x86-64 paths are both known good.

Note that the javacpp plugin version is deliberately `1.5.10` rather than
`1.5.7` (the javacpp version the rest of the tree resolves to). On Gradle 8.8
the 1.5.7 plugin fails with `NoSuchMethodException: getOriginalMetadata` and
silently prunes nothing.

## Project layout

```
src/main/java/com/hottes/caleb/ultimateticktacktoe/
  UltimateTickTackToe.java   JavaFX Application entry point
  Launcher.java              main class used by the jar/installer
  Resources.java             shared constants, images, bundled networks
  BoardState / SubBoardState game rules and state
  ui/                        menu, game screen, player configuration
  gameindependant/           game-agnostic engine, incl. mcts/
  generictree/               tree structure backing MCTS
  machinelearning/           network training, data generation, benchmarking
evaluators/                  saved engine configurations (.zip)
data/                        training data, saved models, game logs
docs/                        generated javadoc
build-installer.bat          installer build script, Windows
build-installer.sh           installer build script, Linux / macOS
MCTS_ANALYSIS.md             deep dive on the search engine
```

## Building and testing

```
gradlew.bat build      REM full build including tests   (./gradlew build on Linux/macOS)
gradlew.bat test       REM tests only
gradlew.bat jar        REM runnable jar (needs JavaFX on the module path)
```

The installer script skips tests (`-x test`) so that packaging does not depend
on the test suite passing.



## License

See [LICENSE](LICENSE).
