package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.google.gson.stream.JsonToken;
import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.SubBoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;
import com.opencsv.CSVIterator;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import javafx.util.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

class ResourcesTest {

    protected final SubBoardState oneWin = new SubBoardState(new double[][]{
            {1,1,1},
            {0,0,0},
            {0,0,0}},
            false, 3);

    protected final SubBoardState twoWin = new SubBoardState(new double[][]{
            {-1,-1,-1},
            {0,0,0},
            {0,0,0}},
            false, 3);

    protected final SubBoardState empty = new SubBoardState(new double[][]{
            {0,0,0},
            {0,0,0},
            {0,0,0}},
            false, 3);

    protected final SubBoardState inProgress = new SubBoardState(new double[][]{
            {1,-1,1},
            {0,-1,0},
            {1,1,-1}},
            true, 3);

    protected final SubBoardState draw = new SubBoardState(new double[][]{
            {1,-1,1},
            {1,-1,-1},
            {-1,1,1}},
            false, 3);


    @Test
    @Disabled
    void getValueNetworkInputV1_0() {

        BoardState testState = GameController.getTestState();
        //System.out.println(Resources.getValueNetworkInputV1_0(testState));
        try {
            MultiLayerNetwork network = MultiLayerNetwork.load(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\src\\main\\resources\\com\\hottes\\caleb\\ultimateticktacktoe\\valueNetworkV1_0.zip"), true);
            network.init();
            System.out.println(testState);
            System.out.println(network.output(Resources.getValueNetworkInputV1(testState)).getDouble(0, 0));

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Test

    void testGetImageForAI() {

        try {
            DataSetPreProcessor preProcessor = toPreProcess -> {
                DataNormalization normalizer = new NormalizerMinMaxScaler();
                normalizer.fit(toPreProcess);
                normalizer.transform(toPreProcess);
            };
            ImageRegressionDataSetIterator iter = new ImageRegressionDataSetIterator( 76, 75, 3, new File("F:\\UltimateTickTackToeAITrainingData\\policyNetworkInputImages\\"), 10, "png");
            iter.setPreProcessor(preProcessor);
            DataSet dataset = iter.next(10);

            System.out.println(dataset.getFeatures().shapeInfoToString());
            System.out.println(dataset);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        //new ValueNetworkTrainer().train("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\");

    }
}