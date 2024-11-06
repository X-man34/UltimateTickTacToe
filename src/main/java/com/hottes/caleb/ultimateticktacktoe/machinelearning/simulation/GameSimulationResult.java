package com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation;

import java.util.ArrayList;

public record GameSimulationResult(ArrayList<StateDatum> datums, int numMoves, double finalEval, long simTime, long finishTime) {
}
