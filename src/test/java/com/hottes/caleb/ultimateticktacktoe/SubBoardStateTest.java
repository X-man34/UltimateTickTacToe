package com.hottes.caleb.ultimateticktacktoe;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SubBoardStateTest {
    private SubBoardState originalState;
    @BeforeEach
    void setUp() {
        originalState = new SubBoardState(3);
    }

    @Test
    @DisplayName("Get static evaluation of state")
    void getEvaluation() {
        SubBoardState testState = new SubBoardState(new double[][]{
                {0,0,0},
                {0,0,0},
                {0,0,0}},
                false, 3);

        assertEquals(BoardState.Evaluation.IN_PROGRESS.getlabel(), testState.getEvaluation(), "Emtpy board does not evaluate to in progress");
        testState = new SubBoardState(new double[][]{
                {1,1,1},
                {0,0,0},
                {0,0,0}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_ONE_WIN.getlabel(), testState.getEvaluation(), "Player one win upper row not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {-1,-1,-1},
                {0,0,0},
                {0,0,0}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_TWO_WIN.getlabel(), testState.getEvaluation(), "Player two win upper row not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {1,0,0},
                {1,0,0},
                {1,0,0}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_ONE_WIN.getlabel(), testState.getEvaluation(), "Player one win left side not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {-1,0,0},
                {-1,0,0},
                {-1,0,0}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_TWO_WIN.getlabel(), testState.getEvaluation(), "Player two win left side not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {1,0,0},
                {0,1,0},
                {0,0,1}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_ONE_WIN.getlabel(), testState.getEvaluation(), "Player one win main diagonal not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {-1,0,0},
                {0,-1,0},
                {0,0,-1}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_TWO_WIN.getlabel(), testState.getEvaluation(), "Player two win main diagonal not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {0,0,1},
                {0,1,0},
                {1,0,0}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_ONE_WIN.getlabel(), testState.getEvaluation(), "Player one win off diagonal not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {0,0,-1},
                {0,-1,0},
                {-1,0,0}},
                false, 3);
        assertEquals(BoardState.Evaluation.PLAYER_TWO_WIN.getlabel(), testState.getEvaluation(), "Player two win off diagonal not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {1,0,1},
                {0,-1,0},
                {1,0,0}},
                false, 3);
        assertEquals(BoardState.Evaluation.IN_PROGRESS.getlabel(), testState.getEvaluation(), "Arbitrary in progress game not evaluated correctly");

        testState = new SubBoardState(new double[][]{
                {1,-1,1},
                {1,-1,-1},
                {-1,1,1}},
                false, 3);
        assertEquals(BoardState.Evaluation.DRAW.getlabel(), testState.getEvaluation(), "Draw not evaluated correctly");

    }

    @Test
    @DisplayName("Test Action simulation")
    void simulateAction() {
        GameState newState = originalState.simulateAction(new GameAction(1, 1, 0), false);//simulate an action that does nothing
        newState.preformAction(new GameAction(0, 0, 1));//make a change. If simulate action returns a copy like it is supposed to then this change should not be reflected in the original state
        assertNotEquals(originalState.itemAt(0, 0), newState.itemAt(0, 0), "New state is pointer to old state");//preforms an action which should not actualy change the state of the board
        GameAction action = new GameAction(1, 1, -1);
        newState = originalState.simulateAction(action, false);
        assertEquals(newState.getStateAsCopy()[action.y][action.x], action.getMarker(), "New state does not contain marker in correct place");
        GameState invertedState = originalState.simulateAction(action, true);
        assertEquals(invertedState.getStateAsCopy()[action.y][action.x], -action.getMarker(), "Failed to invert board");
    }

}