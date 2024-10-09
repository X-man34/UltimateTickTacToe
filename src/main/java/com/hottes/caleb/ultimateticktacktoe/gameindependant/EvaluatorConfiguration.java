package com.hottes.caleb.ultimateticktacktoe.gameindependant;

public record EvaluatorConfiguration(double cValue, long maxRolloutDepth, int computeTime, int threads, int stupidity, boolean allowForcePlay, boolean maxMyCPU) {
    @Override
    public String toString() {
        return "EvaluatorConfiguration{" +
                "cValue=" + cValue +
                ", maxRolloutDepth=" + maxRolloutDepth +
                ", computeTime=" + computeTime +
                ", threads=" + threads +
                ", stupidity=" + stupidity +
                ", allowForcePlay=" + allowForcePlay +
                ", maxMyCPU=" + maxMyCPU +
                '}';
    }
}
