package com.hottes.caleb.ultimateticktacktoe.machinelearning;


import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class ValueNetworkTrainer {
    private static final Logger log = LoggerFactory.getLogger(ValueNetworkTrainer.class);

    public static void main(String[] args) {
        try {
            DataSet allData = Resources.concatenateValueDataFiles(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data"), "series1", "value");
            int featuresCount = allData.numInputs();
            int classCount = allData.numOutcomes();
            allData.shuffle(42);
            DataNormalization normalizer = new NormalizerMinMaxScaler();
            normalizer.fit(allData);
            normalizer.transform(allData);


            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(.80);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();


            //90 data points
            int numHiddenNeurons = 500;

            MultiLayerConfiguration configuration
                    = new NeuralNetConfiguration.Builder()
                    .seed(2039402)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam.Builder().learningRate(.001).build())
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(featuresCount).nOut(numHiddenNeurons).build())
                    .layer(1, new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(2, new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(3, new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(4, new OutputLayer.Builder(
                            LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .nIn(numHiddenNeurons).nOut(classCount).build())
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            model.init();

//            UIServer uiServer = UIServer.getInstance();
//            StatsStorage statsStorage = new InMemoryStatsStorage();
//            uiServer.attach(statsStorage);
//            model.addListeners(new StatsListener(statsStorage, 10));
            //  model.addListeners(new ScoreIterationListener(500));

            System.out.println("beginning training");
            int numEpochs = 500;
            int batchSize = 50;
            DataSetIterator iterator = getIterator(trainingData, batchSize);
            for (int epoch = 0; epoch < numEpochs; epoch++) {
                iterator.reset(); // Reset the iterator for each epoch
                model.fit(iterator);

                if (epoch % 10 == 0) {
                    // Evaluate on validation set
                    INDArray validationOutput = model.output(testData.getFeatures());
                    RegressionEvaluation validationEvaluator = new RegressionEvaluation();
                    validationEvaluator.eval(testData.getLabels(), validationOutput);
                    System.out.println("Validation Stats for Epoch " + epoch + ": \n" + validationEvaluator.stats());

                    INDArray trainOutput = model.output(trainingData.getFeatures());
                    RegressionEvaluation trainindEval = new RegressionEvaluation();
                    trainindEval.eval(trainingData.getLabels(), trainOutput);
                    if (trainindEval.averageMeanSquaredError() * 1.25 < validationEvaluator.averageMeanSquaredError()) {
                        System.out.println("overfitting detected");
                        System.out.println("training MSE=" + trainindEval.averageMeanSquaredError());
                    }

                }

            }

            INDArray output = model.output(testData.getFeatures());
            RegressionEvaluation eval = new RegressionEvaluation();
            eval.eval(testData.getLabels(), output);
            log.info("\n{}", eval.stats());

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private static DataSetIterator getIterator(DataSet dataSet, int batchSize) {
        INDArray features = dataSet.getFeatures();
        INDArray labels = dataSet.getLabels();
        ArrayList<Pair<INDArray, INDArray>> arr = new ArrayList<>();
        for (int i = 0; i < features.rows(); i++) {
            arr.add(new Pair<>(features.getRow(i), labels.getRow(i)));
        }
        return new INDArrayDataSetIterator(arr, batchSize);
    }


    /**
     * This method converts a densely stored board state into a tensor form that can be easily understood by the model
     * Pasting from chatGPT:
     * <p>
     * <p>
     * 1 hot Encoding Structure:
     * For each small board:
     * Channel 1: "X" presence (1 for "X", 0 otherwise)
     * Channel 2: "O" presence (1 for "O", 0 otherwise)
     * Channel 3: Empty presence (1 if empty, 0 otherwise)
     * Channel 4: Active state (1 if active, 0 if inactive)
     * The full game state tensor will have the shape
     * (4,3,3,3,3) to represent all 9 boards.
     * so: (channel,majRow,majCol, minRow,minCol)
     * for board activity all entries for the activity channel for that subboard will be hot.
     * <p>
     * The input is 90 doubles the first 81 are either 1, -1, or 0 representing if X O or nobody occupies that specific square. The order of indexing is for each sub board top left to top right
     * then the next row left to right, and the third row. The order of indexing of the major boards is the same as that of the minor boards.
     *
     * @param dataArray
     * @return
     */
    protected static float[][][][][] getBoardTensor(double[] dataArray) {
        int boardSize = 3;
        float[][][][][] tensor = new float[4][boardSize][boardSize][boardSize][boardSize];
        for (int majRow = 0; majRow < boardSize; majRow++) {
            for (int majCol = 0; majCol < boardSize; majCol++) {
                float activity = dataArray[Resources.getActivityIndex(majRow, majCol, boardSize)] == 1?1:0;
                for (int minRow = 0; minRow < boardSize; minRow++) {
                    for (int minCol = 0; minCol < boardSize; minCol++) {
                        double val = dataArray[com.hottes.caleb.ultimateticktacktoe.Resources.getIndex(majRow, majCol, minRow, minCol, boardSize)];
                        tensor[1][majRow][majCol][minRow][minCol] = 0;//initially not O but if we determine that it is O then we will see later.
                        tensor[3][majRow][majCol][minRow][minCol] = activity;//set the activity channel
                        if (val == 0) {//empty channel first, most likly
                            tensor[2][majRow][majCol][minRow][minCol] = 1;
                        } else {
                            tensor[2][majRow][majCol][minRow][minCol] = 0;
                            if (val == 1) {
                                tensor[0][majRow][majCol][minRow][minCol] = 1;
                            } else {
                                tensor[0][majRow][majCol][minRow][minCol] = 0;
                                //so its not empty, and its not X so its O
                                tensor[1][majRow][majCol][minRow][minCol] = 1;
                            }//end not X
                        }//end not empty
                    }

                }

            }

        }
        return tensor;
    }//end method

}




