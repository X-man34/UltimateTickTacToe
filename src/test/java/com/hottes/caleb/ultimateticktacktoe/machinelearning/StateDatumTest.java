package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.StateDatum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;

import java.util.Optional;

class StateDatumTest {

    @Test
    void getIndex() {
        assertEquals(0, com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex(0, 0, 0, 0,3 ));
        assertEquals(4, com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex(0, 0, 1, 1,3 ));
        assertEquals(37, com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex(1, 1, 0, 1,3 ));
        assertEquals(69, com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex(2, 1, 2, 0,3 ));
        assertEquals(80, com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex(2, 2, 2, 2,3 ));
        assertEquals(20, com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex(0, 2, 0, 2,3 ));
        assertEquals(60, com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex(2, 0, 2, 0,3 ));
        assertEquals(49, Resources.getIndex(1, 2, 1, 1,3 ));


    }

    @Test
    void testConstructorAndGetters() {
        int boardSize = 3;
        //make a board state that we can preform some tests on
        BoardState boardState = new BoardState(boardSize);
        boardState.preformAction(new UltimateTickTacToeGameAction(0, 0, 0, 0, 1));
        boardState.preformAction(new UltimateTickTacToeGameAction(0, 0, 1, 0, -1));
        boardState.preformAction(new UltimateTickTacToeGameAction(1, 0, 2,2, 1));
        boardState.preformAction(new UltimateTickTacToeGameAction(2, 2, 1, 1, -1));

        //ask the computer what the best move is and in doing so create a search tree
        MCTSEvaluator evaluator = new MCTSEvaluator(boardState, new EvaluatorConfiguration(2, 1000, 5, 1, 0, false, false, Optional.empty()));//low compute time, this doesn't need to be accurate.
        evaluator.dispalyDialogAfterSearch = false;
        evaluator.log = false;
        evaluator.preformSearch();//make the tree

        //create a test datum based on what happened in that search
        StateDatum testDatum = new StateDatum(boardState,evaluator.tree.getRoot(), boardState.isPlayerOneTurn(), boardSize);

        //create actual value network input vector
        double[] actualValueNetworkInput = new double[] {1,0,0,-1,0,0,0,0,0,//0, 0
                0,0,0,0,0,0,0,0,0,//0, 1
                0,0,0,0,0,0,0,0,0,//0, 2
                0,0,0,0,0,0,0,0,1,//1, 0
                0,0,0,0,0,0,0,0,0,//1,1
                0,0,0,0,0,0,0,0,0,//1,2
                0,0,0,0,0,0,0,0,0,//2, 0
                0,0,0,0,0,0,0,0,0,//2, 1
                0,0,0,0,-1,0,0,0,0,//2,2
                -1,-1,-1,-1,1,-1,-1,-1,-1//activity of boards
                };

        //Test if the value network inputs are correct.
        assertArrayEquals(actualValueNetworkInput,testDatum.valueNetworkInput, "Value network input vector did not match expected");

        //test some other paramaters
        assertEquals(boardState.isPlayerOneTurn(), testDatum.isPlayerOneTurnOriginally);
        assertEquals(boardState, testDatum.boardStateForPrinting);

        //check that the probabilities add to 1 for the policy data
        double sum = 0;
        for (double prob : testDatum.policyNetworkOutput) {
            sum += prob;
        }
        assertTrue(Math.abs(sum - 1) <= .0001, "Probabilities don't add to 1");//have to account for floating point error
        //check that moves that are not legal have zero probability
        assertEquals(testDatum.policyNetworkOutput[76], 0, "Invalid move has nonzero probability");
        assertEquals(testDatum.policyNetworkOutput[5], 0, "Invalid move has nonzero probability");
        assertEquals(testDatum.policyNetworkOutput[20], 0, "Invalid move has nonzero probability");
        assertEquals(testDatum.policyNetworkOutput[30], 0, "Invalid move has nonzero probability");
        assertEquals(testDatum.policyNetworkOutput[62], 0, "Invalid move has nonzero probability");

        //check that one of the valid moves was calculated correctly based on the evaluator's tree
        final double[] prob = {0};
        UltimateTickTacToeGameAction hypoteticalActionTaken = new UltimateTickTacToeGameAction(1, 1, 0, 0, 1);
        evaluator.tree.getRoot().getChildren().forEach(nodeDataGenericTreeNode -> {
            if (hypoteticalActionTaken.equals(nodeDataGenericTreeNode.getData().getActionTaken())) {
                prob[0] = (double) nodeDataGenericTreeNode.getData().getNumVisits() / evaluator.tree.getRoot().getData().getNumVisits();
            }
        });

        assertEquals(prob[0], testDatum.policyNetworkOutput[Resources.getIndex(1, 1, 0, 0, boardSize)], "Probability not calculated correctly");
    }


}