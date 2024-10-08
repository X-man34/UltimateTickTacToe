package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.UltimateTickTackToe;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationCalibration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.function.Consumer;

public class CreateTrainingData {
    public static void main(String[] args) {
        //simulate game
        EvaluatorConfiguration playerConfig = new EvaluatorConfiguration(2, 1000, 10, 1, 0, false, false);
        BoardState currentState = new BoardState(3);//here player one stays player one
        currentState = GameController.getTestState();
        System.out.println("Inital State: ");
        System.out.println(currentState);
        ArrayList<StateDatum> datums = new ArrayList<>();
        boolean gameRunning = true;
        double eval = 0;
        while (gameRunning) {
            BoardState stateToEval = currentState.getClone();
            if (!currentState.isPlayerOneTurn()) {
                stateToEval.invertState();
            }
            MCTSEvaluator evaluator = new MCTSEvaluator(stateToEval, playerConfig);
            evaluator.dispalyDialogAfterSearch = false;
            UltimateTickTacToeGameAction actionToTake = (UltimateTickTacToeGameAction) evaluator.preformSearch();
            datums.add(new StateDatum(stateToEval, evaluator.tree.getRoot(), currentState.isPlayerOneTurn()));
            actionToTake.setMarker(currentState.isPlayerOneTurn()?1:-1);
            currentState.preformAction(actionToTake);
            System.out.println(currentState);
            eval =  currentState.getEvaluation();
            if (Resources.Evaluation.IN_PROGRESS.getlabel() != eval) {
                //then the game is now over.
                break;
            }
        }
        System.out.println("Game Over");
        //set the evaluations
        double finalEval = eval;
        datums.forEach(stateDatum -> stateDatum.setEvalForThisState(stateDatum.isPlayerOneTurnOriginally? finalEval :-finalEval));
        System.out.println(datums);



//        try (RecordReader recordReader = new CSVRecordReader(1, ',')) {
//
//            recordReader.initialize(new FileSplit(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\irisData.csv")));
//
//            int featuresCount = 4;
//            int classCount = 3;
//            DataSetIterator iterator = new RecordReaderDataSetIterator(
//                    recordReader, 150, featuresCount, classCount);
//            org.nd4j.linalg.dataset.DataSet allData = iterator.next();
//
//            allData.shuffle(42);
//            DataNormalization normalizer = new NormalizerStandardize();
//            normalizer.fit(allData);
//            normalizer.transform(allData);
//
//            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
//            DataSet trainingData = testAndTrain.getTrain();
//            DataSet testData = testAndTrain.getTest();
//
//
//            MultiLayerConfiguration configuration
//                    = new NeuralNetConfiguration.Builder()
//                    .iterations(1000)
//                    .activation(Activation.TANH)
//                    .weightInit(WeightInit.XAVIER)
//                    .learningRate(0.1)
//                    .regularization(true).l2(0.0001)
//                    .list()
//                    .layer(0, new DenseLayer.Builder().nIn(featuresCount).nOut(3).build())
//                    .layer(1, new DenseLayer.Builder().nIn(3).nOut(3).build())
//                    .layer(2, new OutputLayer.Builder(
//                            LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                            .activation(Activation.SOFTMAX)
//                            .nIn(3).nOut(classCount).build())
//                    .backprop(true).pretrain(false)
//                    .build();
//
//            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
//            model.init();
//            model.fit(trainingData);
//
//            INDArray output = model.output(testData.getFeatureMatrix());
//            Evaluation eval = new Evaluation(3);
//            eval.eval(testData.getLabels(), output);
//
//            System.out.println(eval.stats());
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        } catch (InterruptedException e) {
//            throw new RuntimeException(e);
//        }
    }
    //fill this in and another one like it for the policy network so the data can be saved.
    public DataSet getValueNetworkDataset (ArrayList<StateDatum> data) {
        return  null;
    }
}
