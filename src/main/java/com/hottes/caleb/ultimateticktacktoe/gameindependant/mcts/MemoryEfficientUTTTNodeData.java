package com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameState;

/**
 * this class is functionally the same as a node data but uses 23 times less memory.
 * This is possible through storing the game state very concisley but comes at an unknown cpu use cost.
 * externally this class should be indistinguishale from a normal node data.
 */
public class MemoryEfficientUTTTNodeData extends NodeData{
    //does not accout for whether boards are active and breaks the algorithm
    public MemoryEfficientUTTTNodeData(int numVisits, double totalScore, GameState state, GameAction action) {
        super(numVisits, totalScore, action, state.getStringHash());
    }




    @Override
    public GameState getGameState() {
        return new BoardState(hash);
    }

}
