package com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameState;

public class BitSetBasedUTTTNodeData extends NodeData {

    public BitSetBasedUTTTNodeData(int numVisits, double totalScore, GameState state, GameAction action) {
        super(numVisits, totalScore, action, state.getBitSet());
    }


    @Override
    public String toString() {
        return "BitSetNode{" + "numVisits=" + numVisits + ", totalScore=" + totalScore + getGameState().toString() +
                "actionTaken=" + actionTaken + '}';
    }

    @Override
    public GameState getGameState() {
        return new BoardState(bitSet);
    }
}
