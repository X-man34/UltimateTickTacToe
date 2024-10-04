package com.hottes.caleb.ultimateticktacktoe.gameindependant;

import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.SubBoardState;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static com.hottes.caleb.ultimateticktacktoe.Resources.generateMatrixPermutations;
import static org.junit.jupiter.api.Assertions.*;

class GameStateTest {

    private GameState state;
    private GameAction action;
    @BeforeEach
    void init() {
        state = new SubBoardState(3);
        action =  new GameAction(0, 1, 1);
    }

    @Test
    void getActions() {
        ArrayList<GameAction> actions = state.getActions();
        assertEquals(actions.size(), 9, "Empty sub board does not have 9 actions");

        state.preformAction(action);
        actions = state.getActions();
        for (int i = 0; i < state.getRows(); i++) {
            for (int j = 0; j < state.getCols(); j++) {
                //for each slot in the board check that if there is not marker there that an action was created for that place
                //if there was not marker make sure that no action was created
                int finalJ = j;
                int finalI = i;
                long numActions = actions.stream().filter(gameAction -> gameAction.x == finalI && gameAction.y == finalJ).count();//figure out how many actions are associated with this spot
                if (state.itemAt(i, j) != 0) {
                    //then this spot should have no actions associated with it
                    assertEquals(0, numActions, "In correct number of actions created for non empty slot");
                }else {
                    if (numActions != 1) {
                        System.out.println();
                    }
                    assertEquals(1, numActions, "In correct number of actions created for empty slot");
                }
            }
        }
    }

    @Test
    void preformAction() {
        state.preformAction(action);
        assertEquals(action.getMarker(), state.itemAt(action.y, action.x));
    }


    @Test
    void testShaHashCode() {
        ArrayList<GameAction> actions = state.getActions();
        Set<String> set = new HashSet<>();
        for (GameAction actionToSimulate : actions) {
            GameState newState = state.simulateAction(actionToSimulate, false);
            String newCode = newState.getSHAHash();
            assertFalse(set.contains(newCode), "Duplicate Sha Hashcodes created");
            set.add(newCode);
        }

    }

    @Test
    void testFastHashCode() {
        //create a list of all possible game states values will be -1, 0 or 1
        List<double[][]> rawPermutations = new ArrayList<>();
        generateMatrixPermutations(new double[3][3], 0, 0, rawPermutations);
        List<GameState> ticTacToeStates = new ArrayList<>();
        rawPermutations.forEach(doubles -> ticTacToeStates.add(new SubBoardState(doubles, false, 3)));
        assertEquals(0, (int) Math.round(Resources.Evaluation.DRAW.getlabel()), "Draw reward does not round to 0, could cause problems with fast hashcoding");
        Set<Integer> set = new HashSet<>();
        for (GameState state : ticTacToeStates) {
            int hashcode = Resources.fastTicTacToeHashcode(state.getStateAsCopy());
            assertFalse(set.contains(hashcode), "Duplicate fast Hashcodes created");
            set.add(hashcode);

        }

    }

    @Test
    void testHashConstructor() {
        GameState testState = new SubBoardState(3);
        testState.preformAction(new GameAction(1, 2, -1));
        String hash = testState.getStringHash();
        GameState newState = new SubBoardState(hash);
        assertEquals(testState.getStringHash(), newState.getStringHash(), "String hashes were not the same");
        assertEquals(testState.getSHAHash(), newState.getSHAHash(), "SHA hashes were not the same");
        assertEquals(testState, newState, "States were not the same");


    }
}