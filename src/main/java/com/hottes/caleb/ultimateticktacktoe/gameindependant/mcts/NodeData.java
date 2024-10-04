package com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameState;

import java.text.DecimalFormat;
import java.util.BitSet;

public class NodeData {

    protected int numVisits;
    protected double totalScore;
    private final GameState gameState;
    protected final GameAction actionTaken;
    protected final String hash;
    protected final BitSet bitSet;
    public NodeData(int numVisits, double totalScore, GameState state, GameAction action) {
        this.gameState = state;
        this.numVisits = numVisits;
        this.totalScore = totalScore;
        this.actionTaken = action;
        hash = null;
        bitSet = null;
    }

    protected NodeData(int numVisits, double totalScore, GameAction action, String hash) {
        this.gameState = null;
        this.numVisits = numVisits;
        this.totalScore = totalScore;
        this.actionTaken = action;
        this.hash = hash;
        bitSet = null;
    }

    protected NodeData(int numVisits, double totalScore, GameAction action, BitSet data) {
        this.gameState = null;
        this.numVisits = numVisits;
        this.totalScore = totalScore;
        this.actionTaken = action;
        this.hash = null;
        bitSet = data;
    }

    public int getNumVisits() {
        return numVisits;
    }

    public void incrementNumVisits() {
        this.numVisits += 1;
    }

    public double getTotalScore() {
        return totalScore;
    }

    public void changeTotalScore(double delta) {
        this.totalScore += delta;
    }

    public GameState getGameState() {
        return gameState;
    }

    public GameAction getActionTaken() {
        return actionTaken;
    }


    @Override
    public String toString() {
        return "Node{" + "numVisits=" + numVisits + ", totalScore=" + totalScore + gameState.toString() +
                "actionTaken=" + actionTaken + '}';
    }
    public String getTreeString(DecimalFormat format, int numChildren) {
        return "Node{n=" + numVisits + ",t=" + totalScore + ",v=" + format.format(totalScore / numVisits) + ",c=" + numChildren + "," + actionTaken + " , is player: " + (getGameState().isPlayerOneTurn()?" 1":"-1") + " turn}";
    }
}
