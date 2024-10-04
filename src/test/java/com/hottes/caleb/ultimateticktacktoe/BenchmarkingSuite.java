package com.hottes.caleb.ultimateticktacktoe;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.platform.suite.api.*;

@Suite
@SuiteDisplayName("Benchmarks various MCTS funcionalities")
@SelectPackages({"com.hottes.caleb.ultimateticktacktoe", "com.hottes.caleb.ultimateticktacktoe.gameindependant"})
@IncludeTags("benchmark")
class BenchmarkingSuite {

}