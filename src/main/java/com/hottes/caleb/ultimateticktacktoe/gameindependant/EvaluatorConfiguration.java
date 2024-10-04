package com.hottes.caleb.ultimateticktacktoe.gameindependant;

public record EvaluatorConfiguration(double cValue, long maxRolloutDepth, int computeTime, int threads, int stupidity, boolean allowForcePlay, boolean maxMyCPU) {
}
