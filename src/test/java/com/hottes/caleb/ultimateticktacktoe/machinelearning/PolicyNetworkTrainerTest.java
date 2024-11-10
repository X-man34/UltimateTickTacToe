package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.SubBoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluatorTest;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class PolicyNetworkTrainerTest extends MCTSEvaluatorTest {

    /**
     * tests if the default policy network or another policy network can make simple decision
     *
     */
    @Test
    void train() {
        MultiLayerNetwork policyNetwork = null;
        try {
             policyNetwork = MultiLayerNetwork.load(new File(Resources.class.getResource("policyNetworkV1_0.zip").getPath()), false);
        } catch (IOException e) {
            fail("Unable to load network");
        }
        BoardState testState = new BoardState(new SubBoardState[][]{
                {oneWin, empty, empty},
                {oneWin, empty, empty},
                {inProgress, empty, empty}},
                3);
        testState.setPlayerOneTurn(true);
        testState.setAllBoardsActivity(false);
        testState.setBoardActive(2, 0);
        INDArray output = policyNetwork.output(com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getPolicyNetworkInputV1(testState));
        UltimateTickTacToeGameAction action = com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getBestActionFromPolicyNetworkOutput(output.getRow(0).toDoubleVector(), 3, testState);
        UltimateTickTacToeGameAction correctAction = new UltimateTickTacToeGameAction(2, 0, 1, 0, 1);
        assertEquals(correctAction, action, "Network unable to win in won move");

    }
}