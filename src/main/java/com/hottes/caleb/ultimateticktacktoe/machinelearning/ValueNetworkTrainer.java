package com.hottes.caleb.ultimateticktacktoe.machinelearning;


import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.StateDatum;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.CheckpointListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.concurrent.TimeUnit;

import static com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getBoardIndex;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

public class ValueNetworkTrainer {
    private static final Logger log = LoggerFactory.getLogger(ValueNetworkTrainer.class);
    private static PrintStream outStream;
    public static void main(String[] args) throws FileNotFoundException {


        try {

//            DataSet dataFromFile = getInputDataSetFromRawFilepath("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data", "series1", "value");
//            dataFromFile.save(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\series1DataCombinedValue.bin"));
//            System.exit(0);

            String baseDir = "C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\models\\";
            String modelDir = baseDir + "valueNetwork" + System.currentTimeMillis() + "\\";
            new File(modelDir).mkdirs();
            outStream = new PrintStream(new File(modelDir + "trainingLog.log"));
            String dataset = "C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\series1DataCombinedValue.bin";

            //comcatenate all the data files together and massage the data into a usable format. Only needs to be done once and then the dataset can be accesses and laded when it is needed



            DataSet inputData = new DataSet();
            inputData.load(new File(dataset));

            int featuresCount = inputData.numInputs();
            int classCount = inputData.numOutcomes();
            inputData.shuffle(42);
            //doesn't work with rank 6 data, but the data should already be normalized.
//        DataNormalization normalizer = new NormalizerMinMaxScaler();
//        normalizer.fit(inputData);
//        normalizer.transform(inputData);


            SplitTestAndTrain testAndTrain = splitTestAndTrain(inputData, .8);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();


            //90 data points
            int numHiddenNeurons = 100;
            Subsampling3DLayer subsampling3DLayer = new Subsampling3DLayer();
            subsampling3DLayer.setKernelSize(new int[]{3, 3, 3});
            subsampling3DLayer.setDataFormat(Convolution3D.DataFormat.NDHWC);
            subsampling3DLayer.setStride(new int[] {2, 2, 2});
            subsampling3DLayer.setPoolingType(PoolingType.AVG);
            subsampling3DLayer.setDilation(new int[] {1, 1, 1});
            subsampling3DLayer.setPadding(new int[] {1, 1, 1});
            MultiLayerConfiguration configuration
                    = new NeuralNetConfiguration.Builder()
                    .seed(2039402)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam.Builder().learningRate(.001).build())
                    .list()
                    .layer(0, new Convolution3D.Builder().nIn(4).nOut(4).kernelSize(3, 3, 3).padding(1, 1, 1).stride(1, 1, 1).activation(Activation.IDENTITY).dataFormat(Convolution3D.DataFormat.NDHWC).build())
                    .layer(1, new Convolution3D.Builder().nIn(4).nOut(4).kernelSize(3, 3, 3).padding(1, 1, 1).dataFormat(Convolution3D.DataFormat.NDHWC).build())
                    .layer(2, new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(3, new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(4, new OutputLayer.Builder(
                            LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .nIn(numHiddenNeurons).nOut(classCount).build())
                    .setInputType(InputType.convolutional3D(Convolution3D.DataFormat.NDHWC, 9, 3, 3, 4))
                    .build();
            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            model.init();

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            model.addListeners(new StatsListener(statsStorage, 1));
            model.addListeners(new CheckpointListener.Builder(modelDir).saveEvery(30, TimeUnit.SECONDS).keepLast(2).build());

            log("beginning training");
            log("See web UI at http://localhost:9000/train/overview");
            int maxEpochs = 500;
            int epochsTrained = 0;
            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                model.fit(trainingData);
                epochsTrained++;
                if (epoch % 2 == 0) {
                    // Evaluate on validation set
                    INDArray validationOutput = model.output(testData.getFeatures());
                    RegressionEvaluation validationEvaluator = new RegressionEvaluation();
                    validationEvaluator.eval(testData.getLabels(), validationOutput);
                    log("Validation Stats for Epoch " + epoch + ": \n" + validationEvaluator.stats());

                    INDArray trainOutput = model.output(trainingData.getFeatures());
                    RegressionEvaluation trainindEval = new RegressionEvaluation();
                    trainindEval.eval(trainingData.getLabels(), trainOutput);
                    if (trainindEval.averageMeanSquaredError() * 1.15 < validationEvaluator.averageMeanSquaredError()) {
                        log("overfitting detected");
                        log("training MSE=" + trainindEval.averageMeanSquaredError());
                        break;
                    }

                }

            }

            INDArray output = model.output(testData.getFeatures());
            RegressionEvaluation eval = new RegressionEvaluation();
            eval.eval(testData.getLabels(), output);
            log("\n" + eval.stats());
            log("TestDataLabels: \n" + testData.getLabels());
            log("Actual labels: \n" + output);
            log("Trained for: " + epochsTrained + "epochs");
            log("Trained on dataset: " + dataset);
            log("Saving model");




            model.save(new File(modelDir + "valueNetworkMSE" + eval.averageMeanSquaredError() + ".zip"));
            log("Model Saved\n\n\nConfiguration\n\n\n");
            outStream.println(configuration.toJson());//dont' want this going to the console
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.exit(0);

    }

    private static DataSet getInputDataSetFromRawFilepath(String filepath, String seriesName, String prefix) throws FileNotFoundException {
        DataSet rawData = Resources.concatenateValueDataFiles(new File(filepath),seriesName, prefix);
        double [][][][][] inputVectors = new double[rawData.getFeatures().rows()][][][][];
        double[][] rawInputMatrix = rawData.getFeatures().toDoubleMatrix();
        for (int i = 0; i < rawData.getFeatures().rows(); i++) {
            inputVectors[i] = getBoardTensor(rawInputMatrix[i]);
        }
        int dataRows = rawData.getFeatures().rows();
        int inputChannels = 4;//num channels
        int boardSize = 3;//the other four dimensions are all boardsize, minor and major rows and colummns.


        double[] flatInputVectors = new double[dataRows  * boardSize * boardSize * boardSize * boardSize * inputChannels];
        int index = 0;
        for (int i = 0; i < dataRows; i++) {
            for (int j = 0; j < boardSize * boardSize; j++) {
                for (int k = 0; k < boardSize; k++) {
                    for (int l = 0; l < boardSize; l++) {
                        for (int m = 0; m < inputChannels; m++) {
                            flatInputVectors[index++] = inputVectors[i][j][k][l][m];
                        }
                    }
                }
            }
        }
        INDArray inputFeatures = Nd4j.create(flatInputVectors, new int[]{dataRows, boardSize * boardSize, boardSize, boardSize,inputChannels});
        return new DataSet(inputFeatures, rawData.getLabels());
    }



    /**
     * This method converts a densely stored board state into a tensor form that can be easily understood by the model
     *
     * <p>
     * <p>
     * This data structure needs to work with a 3D dimensional convolutional layer.
     * therefore the input must be 5D.
     * the shape of the input is as follows:
     * [batch or datapoint index, major index,minor row,minor col, channel]
     * the major indexes range from 0 to 8 and follow the standard top left to bottom right scheme that the data arrays use to store the data.
     * the value is 0 or 1 based on the channel
     *
     * Channel 1: "X" presence (1 for "X", 0 otherwise)
     * Channel 2: "O" presence (1 for "O", 0 otherwise)
     * Channel 3: Empty presence (1 if empty, 0 otherwise)
     * Channel 4: Active state (1 if active, 0 if inactive)
     * for board activity all entries for the activity channel for that subboard will be hot.
     * <p>
     * The input is 90 doubles the first 81 are either 1, -1, or 0 representing if X O or nobody occupies that specific square. The order of indexing is for each sub board top left to top right
     * then the next row left to right, and the third row. The order of indexing of the major boards is the same as that of the minor boards.
     *
     * @param dataArray the array of data to convert
     * @return the tensor representing the {@link com.hottes.caleb.ultimateticktacktoe.BoardState}
     * @see StateDatum
     * @see BoardState#getValueNetworkInputVector()
     */
    protected static double[][][][] getBoardTensor(double[] dataArray) {
        int boardSize = 3;
        double[][][][] tensor = new double[boardSize * boardSize][boardSize][boardSize][4];
        for (int majRow = 0; majRow < boardSize; majRow++) {
            for (int majCol = 0; majCol < boardSize; majCol++) {
                float activity = dataArray[Resources.getActivityIndex(majRow, majCol, boardSize)] == 1?1:0;
                for (int minRow = 0; minRow < boardSize; minRow++) {
                    for (int minCol = 0; minCol < boardSize; minCol++) {
                        double val = dataArray[Resources.getIndex(majRow, majCol, minRow, minCol, boardSize)];//1 if X, 0 if empty, -1 if O
                        tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][1] = 0;//initially not "O" but if we determine that it is "O" then we will set it later.
                        tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][3] = activity;//set the activity channel
                        if (val == 0) {//empty channel first, most likly
                            tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][2] = 1;
                        } else {
                            tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][2] = 0;
                            if (val == 1) {
                                tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][0] = 1;
                            } else {
                                tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][0] = 0;
                                //so its not empty, and its not X so its O
                                tensor[getBoardIndex(majRow, majCol, boardSize)][minRow][minCol][1] = 1;
                            }//end not X
                        }//end not empty
                    }

                }

            }

        }
        return tensor;
    }//end method



    /**
     * takes in a dataset of any rank and splits it into training and testing datasets at the given percentage.
     * The only reason I am making this method is because the built in one doesn't support datasets above rank 4 for some reason.
     * this operates the same as the one in the source code does, except it only works for rank 5
     * @param dataSet the dataset to split
     * @param percentTrain the ratio to split the data
     * @return a split test train object.
     */
    public static SplitTestAndTrain splitTestAndTrain(DataSet dataSet, double percentTrain) {
        if (percentTrain >= 1 || percentTrain <= 0) {
            throw new IllegalArgumentException("Percent train must be between 0 and 1 exclusive");
        }
        if (dataSet.getFeatures().rank() != 5) {
            throw new IllegalArgumentException("This method is for rank 5 splitting only");
        }
        long[] shape = dataSet.getFeatures().shape();
        int numTrainingRows = (int) Math.ceil(shape[0] * percentTrain);
        long numRows = shape[0];

        DataSet train = new DataSet();
        train.setFeatures(dataSet.getFeatures().get(interval(0, numTrainingRows), all(), all(), all(), all()));
        train.setLabels(dataSet.getLabels().get(interval(0, numTrainingRows), all()));
        DataSet test = new DataSet();
        test.setFeatures(dataSet.getFeatures().get(interval(numTrainingRows, numRows), all(), all(), all(), all()));
        test.setLabels(dataSet.getLabels().get(interval(numTrainingRows, numRows), all()));
        return new SplitTestAndTrain(train, test);
    }


    private static  void log(String message) {
        log.info(message);
        outStream.println(message);

    }

}




