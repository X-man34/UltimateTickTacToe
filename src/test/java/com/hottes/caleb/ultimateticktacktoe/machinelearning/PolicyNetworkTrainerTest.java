package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.SubBoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluatorTest;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import com.opencsv.CSVWriter;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Scanner;

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
    @Test
    @Disabled
    void visualizeFilters() {
        // Assuming you have a MultiLayerNetwork model
        try {
            MultiLayerNetwork model = MultiLayerNetwork.load(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\models\\policyNetworkMSE.001758\\checkpoint_34_MultiLayerNetwork.zip"), false);
            for (int i = 0; i < 100; i++) {
                try {
                    FilterDisplay.saveFilterAsImage(model, 1, i, "F:\\test\\filterLayer2" + i + ".png");
                }catch (Exception e) {
                    break;
                }
            }

        } catch (IOException e) {

        }


    }

    //when running this these were the results
    //for 81 predictions a 4.2 or so speedup while for 405 it was like 5.69x so as the number of predicitons increases so does the speedup, but only logarithmically.
    //so its good to batch them, but not like super good.
    @Test
    @Disabled
    void testSpeedOfPolicyNetworkEvaluation() {
        MultiLayerNetwork model = Resources.policyNetwork.get();
        BoardState emptyState = new BoardState(3);
        emptyState.setPlayerOneTurn(true);
        emptyState.setAllBoardsActivity(true);
        LinkedList<BoardState> statesToEval = new LinkedList<>();
        for (int i = 0; i < 5; i++) {
            for (GameAction action : emptyState.getActions()) {
                action.setMarker(1);
                statesToEval.add((BoardState) emptyState.simulateAction(action, false));
            }
        }

        int numTests = 100;
        double speedupSum = 0;
        for (int i = 0; i < numTests; i++) {
            //first make predictions one by one
            Iterator<BoardState> iterator = statesToEval.iterator();
            long startTime = System.currentTimeMillis();
            while (iterator.hasNext()) {
                INDArray output = model.output(com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getPolicyNetworkInputV1(iterator.next()));
            }
            long oneByOneTime = System.currentTimeMillis() - startTime;
            //now do it all at once
            startTime = System.currentTimeMillis();
            INDArray inputs = Nd4j.create(statesToEval.size(), 9,3,3,4);
            for (int j = 0; j < statesToEval.size(); j++) {
                INDArray row = com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getPolicyNetworkInputV1(statesToEval.get(j)).get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all());
                inputs.put(new INDArrayIndex[]{NDArrayIndex.point(j), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.all()}, row);

            }
            model.output(inputs);
            long allTogetherTime = System.currentTimeMillis() - startTime;
            speedupSum += (double) oneByOneTime / allTogetherTime;
        }
        double averageSpeedup = speedupSum / numTests;
        System.out.printf("Averge speedup over %d iterations was %.2fx", numTests, averageSpeedup);

    }


    @Test
    void computeStatsFromLog() {
        try {
            Scanner scanner = new Scanner(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\BenchmarkingOutptu.txt"));
            CSVWriter writer = new CSVWriter(new FileWriter("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\benchmarkingStats.csv"));
            writer.writeNext(new String[]{"Player", "Tree Traversal Time"});
            LinkedList<Double> aiTimes = new LinkedList<>();
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (line.startsWith("Average traversal time: ")) {
                    try {
                        double time = Double.parseDouble(line.split(" ")[3]);
                        if (time > 10) {
                            aiTimes.add(time);
                        }

                        writer.writeNext(new String[]{time < 10?"2":"1", String.valueOf(time)});
                    }catch (Exception e) {
                    }

                }
            }
            writer.close();
            System.out.println(com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getSampleStatistics(aiTimes, .99, "ms"));
        } catch (IOException e) {

        }

    }
}