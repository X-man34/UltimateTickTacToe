package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.exception.ND4UnresolvedOutputVariables;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class ValueNetworkTrainerTest {

    @Test
    void getBoardTensor() {
        BoardState testState = GameController.getTestState();
        MCTSEvaluator evaluator = new MCTSEvaluator(testState, new EvaluatorConfiguration(2, 1000, 1, 1, 0, false, false));
        evaluator.dispalyDialogAfterSearch = false;
        evaluator.log = false;
        evaluator.preformSearch();
        StateDatum datum = new StateDatum(testState, evaluator.tree.getRoot(), true, 3);
        double[][][][] testTensor = ValueNetworkTrainer.getBoardTensor(datum.valueNetworkInput);
        double[][][] xChannel = new double[][][]{
                //maj row 0
                    {{1,1,1},//col 0
                     {0,0,0},
                     {0,0,0}},

                    {{1,1,1},//col 1
                    {0,0,0},
                    {0,0,0}},

                    {{0,0,0},//col2
                    {0,0,0},
                    {0,1,1}},

                    {{1,1,1},//col 0
                    {0,0,0},
                    {0,0,0}},
            //maj row 1
                    {{0,0,0},//col 1
                    {0,0,0},
                    {0,0,0}},

                    {{0,0,0},//col 2
                    {0,0,0},
                    {1,1,0}},
                //maj row 2
                    {{0,1,0},//col 0
                     {0,1,0},
                     {0,0,0}},

                    {{0,0,0},//col 1
                     {0,0,0},
                     {0,0,0}},

                    {{0,0,0},//col 2
                     {0,0,0},
                     {0,0,0}}
                };


        double[][][] oChannel = new double[][][]{
                //maj row 0
                        {{0,0,0},//col 0
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 1
                                {0,0,0},
                                {0,0,0}},

                        {{1,0,1},//col2
                                {1,0,1},
                                {0,0,0}},
                //maj row 1
                        {{0,0,0},//col 0
                                {0,0,0},
                                {0,0,0}},

                        {{1,1,1},//col 1
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 2
                                {0,0,0},
                                {0,0,0}},
                //maj row 2
                        {{0,0,0},//col 0
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 1
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 2
                                {0,0,0},
                                {0,0,0}}
        };


        double[][][] emptyChannel = new double[][][]{
                //maj row 0
                        {{0,0,0},//col 0
                                {1,1,1},
                                {1,1,1}},

                        {{0,0,0},//col 1
                                {1,1,1},
                                {1,1,1}},

                        {{0,1,0},//col2
                                {0,1,0},
                                {1,0,0}},
                //maj row 1
                        {{0,0,0},//col 0
                                {1,1,1},
                                {1,1,1}},

                        {{0,0,0},//col 1
                                {1,1,1},
                                {1,1,1}},

                        {{1,1,1},//col 2
                                {1,1,1},
                                {0,0,1}},
                //maj row 2
                        {{1,0,1},//col 0
                                {1,0,1},
                                {1,1,1}},

                        {{1,1,1},//col 1
                                {1,1,1},
                                {1,1,1}},

                        {{1,1,1},//col 2
                                {1,1,1},
                                {1,1,1}}
        };

        double[][][] activityChannel = new double[][][]{
                //maj row 0
                        {{0,0,0},//col 0
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 1
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 2
                                {0,0,0},
                                {0,0,0}},
                //maj row 1
                        {{0,0,0},//col 0
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 1
                                {0,0,0},
                                {0,0,0}},

                        {{1,1,1},//col 2
                                {1,1,1},
                                {1,1,1}},
                //maj row 2
                        {{0,0,0},//col 0
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 1
                                {0,0,0},
                                {0,0,0}},

                        {{0,0,0},//col 2
                                {0,0,0},
                                {0,0,0}}
        };

        double[][][][] expectedTensor = new double[][][][] {xChannel, oChannel, emptyChannel, activityChannel};

        assertTrue(Arrays.deepEquals(xChannel, testTensor[0]), "X Channel not equal: Expected:\n" + Arrays.deepToString(xChannel) + "\n Actual:\n" + Arrays.deepToString(testTensor[0]));
        assertTrue(Arrays.deepEquals(oChannel, testTensor[1]), "O Channel not equal: Expected:\n" + Arrays.deepToString(oChannel) + "\n Actual:\n" + Arrays.deepToString(testTensor[1]));
        assertTrue(Arrays.deepEquals(emptyChannel, testTensor[2]), "Emptyness Channel not equal: Expected:\n" + Arrays.deepToString(emptyChannel) + "\n Actual:\n" + Arrays.deepToString(testTensor[2]));
        assertTrue(Arrays.deepEquals(activityChannel, testTensor[3]), "Activity Channel not equal: Expected:\n" + Arrays.deepToString(activityChannel) + "\n Actual:\n" + Arrays.deepToString(testTensor[3]));
        assertTrue(Arrays.deepEquals(expectedTensor, testTensor), "Tensors not equal: Expected:\n" + Arrays.deepToString(expectedTensor) + "\n Actual:\n" + Arrays.deepToString(testTensor));

    }
}