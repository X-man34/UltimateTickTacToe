package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.CNN2DFormat;
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
import org.nd4j.enums.DataFormat;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.awt.*;
import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class PolicyNetworkTrainer extends NetworkTrainer {
    @Override
    public void train(String baseDir) {


        try {

//            preprocessPolicyImageInput("F:\\UltimateTickTackToeAITrainingData\\policyNetworkInputImages\\");
//            System.exit(0);
            String thisModelDir = baseDir + "models\\policyNetwork" + System.currentTimeMillis() + "\\";
            new File(thisModelDir).mkdirs();
            outStream = new PrintStream(thisModelDir + "trainingLog.log");
            String datasetDir = "F:\\UltimateTickTackToeAITrainingData\\policyNetworkInputImages\\";
            String labelsFile =  datasetDir + "labels.csv";
            int height = 76;
            int width = 76;
            int channels = 3;

//            ImageRegressionDataSetIterator iterator = new ImageRegressionDataSetIterator(width, height, channels, new File(datasetDir), 5);
//
//            System.out.println(matrix);
//            System.out.println(matrix.shapeInfoToString());


            DataSet inputData = new DataSet();
            System.out.println("loading data");
          //  inputData.load(new File(dataset));
            System.out.println("data loaded");

            int featuresCount = inputData.numInputs();
            int classCount = inputData.numOutcomes();
            inputData.shuffle(42);
            //doesn't work with rank 6 data, but the data should already be normalized.
//        DataNormalization normalizer = new NormalizerMinMaxScaler();
//        normalizer.fit(inputData);
//        normalizer.transform(inputData);


            SplitTestAndTrain testAndTrain = inputData.splitTestAndTrain(.8);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();


            int numHiddenNeurons = 250;
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
                    .updater(new Adam.Builder().learningRate(.006).build())
                    .list()
                    .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(4, 4).padding(1, 1).stride(1, 1).activation(Activation.IDENTITY).build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
                    .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(4, 4).padding(1, 1).stride(1, 1).activation(Activation.IDENTITY).build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
                    .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(4, 4).padding(1, 1).stride(1, 1).activation(Activation.IDENTITY).build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
                    .layer(new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(new OutputLayer.Builder(
                            LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .nIn(numHiddenNeurons).nOut(classCount).build())
                    .setInputType(InputType.convolutional(76, 76, 3))
                    .build();
            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            model.init();

            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);


            model.addListeners(new CheckpointListener.Builder(thisModelDir).saveEvery(30, TimeUnit.SECONDS).keepLast(2).build());

            log("beginning training");
            String link = "http://localhost:9000/train/overview";
            log("See web UI at " + link);
            if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                Desktop.getDesktop().browse(new URI(link));
            }
            int maxEpochs = 500;
            int epochsTrained = 0;
            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                if (epoch == 2) {
                    model.addListeners(new StatsListener(statsStorage, 1));//only add this after a bit so scale the graph so its useful
                }
                model.fit(trainingData);
                epochsTrained++;
                if (epoch % 2 == 0) {
                    // Evaluate on validation set
                    INDArray validationOutput = model.output(testData.getFeatures());
                    RegressionEvaluation validationEvaluator = new RegressionEvaluation();
                    validationEvaluator.eval(testData.getLabels(), validationOutput);


                    INDArray trainOutput = model.output(trainingData.getFeatures());
                    RegressionEvaluation trainindEval = new RegressionEvaluation();
                    trainindEval.eval(trainingData.getLabels(), trainOutput);

                    double percentDiff = Math.abs(trainindEval.averageMeanSquaredError() - validationEvaluator.averageMeanSquaredError()) / validationEvaluator.averageMeanSquaredError() * 100;
                    log("\nValidation Stats for Epoch " + epoch + ": \n" + validationEvaluator.averageMeanSquaredError() + "\nTrainTestDifference: " + percentDiff + " %");
                    if (percentDiff > 1) {
                        log("overfitting detected, Difference: " + percentDiff + " %");
                        break;
                    }

                }

            }

            INDArray output = model.output(testData.getFeatures());
            RegressionEvaluation eval = new RegressionEvaluation();
            eval.eval(testData.getLabels(), output);
            log("\n" + eval.averageMeanSquaredError());
            log("TestDataLabels: \n" + testData.getLabels());
            log("Actual labels: \n" + output);
            log("Trained for: " + epochsTrained + "epochs");
            //log("Trained on dataset: " + dataset);
            log("Saving model");




            model.save(new File(thisModelDir + "valueNetworkMSE" + eval.averageMeanSquaredError() + ".zip"));
            log("Model Saved\n\n\nConfiguration\n\n\n");
            outStream.println(configuration.toJson());//dont' want this going to the console
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
        System.exit(0);

    }
    /**
     * Loads regression labels from a CSV file.
     * Assumes each row corresponds to an image with a label vector.
     *
     * @param labelsFile Path to the CSV file
     * @return A map where the key is the image filename and the value is the label (double array)
     */
    public static Map<String, double[]> loadLabels(String labelsFile) throws IOException {
        Map<String, double[]> labelsMap = new HashMap<>();
        BufferedReader reader = new BufferedReader(new FileReader(labelsFile));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            String imageName = parts[0]; // Assuming the first column is the image name
            double[] label = new double[81]; // 80 elements for each label
            for (int i = 0; i < 81; i++) {
                label[i] = Double.parseDouble(parts[i + 1]); // Label values start from index 1
            }
            labelsMap.put(imageName, label);
        }
        reader.close();
        return labelsMap;
    }



    public static void main(String[] args) {

        String folderPath = "C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data";//for development
//        Scanner scanner = new Scanner(System.in);
//        File dataFolder = Resources.getDataFolderFromUser(scanner);
        //String folderPath = dataFolder.getAbsolutePath();
        new PolicyNetworkTrainer().train(folderPath + "\\");
    }
}
