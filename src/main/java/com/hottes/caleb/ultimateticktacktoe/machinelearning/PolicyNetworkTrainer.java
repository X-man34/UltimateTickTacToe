package com.hottes.caleb.ultimateticktacktoe.machinelearning;

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
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
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
//            preprocessPolicyImageInput("F:\\UltimateTickTackToeAITrainingData\\policyNetworkImageData\\", baseDir, "series1", .8, 3, 1);
//            System.exit(0);
            String thisModelDir = baseDir + "models\\policyNetwork" + System.currentTimeMillis() + "\\";
            new File(thisModelDir).mkdirs();
            outStream = new PrintStream(thisModelDir + "trainingLog.log");
            String dataDir = "F:\\UltimateTickTackToeAITrainingData\\policyNetworkImageData\\";
            String trainDir = dataDir + "train";
            String testDir = dataDir + "test";

            int height = 76;
            int width = 76;
            int channels = 3;
            int batchSize = 1000;

            DataSetPreProcessor normalizationPreProcessor = toPreProcess -> {
                DataNormalization normalizer = new NormalizerMinMaxScaler();
                normalizer.fit(toPreProcess);
                normalizer.transform(toPreProcess);
            };
            ImageRegressionDataSetIterator trainingDataIterator = new ImageRegressionDataSetIterator( width, height, channels, new File(trainDir), batchSize, "png");
            trainingDataIterator.setPreProcessor(normalizationPreProcessor);
            ImageRegressionDataSetIterator testDataIterator = new ImageRegressionDataSetIterator( width, height, channels, new File(testDir), batchSize, "png");
            testDataIterator.setPreProcessor(normalizationPreProcessor);
            DataSet validationSet = testDataIterator.next(6000);
            int classCount = trainingDataIterator.totalOutcomes();

            int numHiddenNeurons = 300;
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
                    .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(5, 5).padding(1, 1).stride(1, 1).activation(Activation.IDENTITY).build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
                    .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(5, 5).padding(1, 1).stride(1, 1).activation(Activation.IDENTITY).build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
                    .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(4, 4).padding(1, 1).stride(1, 1).activation(Activation.IDENTITY).build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
                    .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(4, 4).padding(1, 1).stride(1, 1).activation(Activation.IDENTITY).build())
                    .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
                    .layer(new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
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
            InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);

            model.addListeners(new CheckpointListener.Builder(thisModelDir).saveEvery(30, TimeUnit.SECONDS).keepLast(2).build());




            log("beginning training");
            String link = "http://localhost:9000/train/overview";
            log("See web UI at " + link);
            if (Desktop.isDesktopSupported() && Desktop.getDesktop().isSupported(Desktop.Action.BROWSE)) {
                Desktop.getDesktop().browse(new URI(link));
            }
            int maxEpochs = 500;
            int epoch = 0;
            int batch = 0;
            int lastEval = 0;
            for (epoch = 0; epoch < maxEpochs;) {
                if (!trainingDataIterator.hasNext()) {
                    trainingDataIterator.reset();
                    epoch++;
                }
                model.fit(trainingDataIterator.next());
                batch++;
                if (batch == 4) {
                    model.addListeners(new StatsListener(statsStorage, 1));

                }
                if (epoch > lastEval) {
                    System.out.println("Evaluating");
                    INDArray validationOutput = model.output(validationSet.getFeatures());
                    RegressionEvaluation validationEvaluator = new RegressionEvaluation();
                    validationEvaluator.eval(validationSet.getLabels(), validationOutput);

                    trainingDataIterator.reset();
                    INDArray trainOutput = model.output(trainingDataIterator);
                    RegressionEvaluation trainindEval = new RegressionEvaluation();
                    trainindEval.eval(trainingDataIterator.getAllLabels(), trainOutput);

                    double percentDiff = Math.abs(trainindEval.averageMeanSquaredError() - validationEvaluator.averageMeanSquaredError()) / validationEvaluator.averageMeanSquaredError() * 100;
                    log("\nValidation Stats for Epoch " + epoch + ": \n" + validationEvaluator.averageMeanSquaredError() + "\nTrainTestDifference: " + percentDiff + " %");
                    lastEval++;
                    trainingDataIterator.reset();//so it doesn't loop forever
                    if (percentDiff > 2.5) {
                        break;
                    }
                }

            }

            INDArray output = model.output(validationSet.getFeatures());
            RegressionEvaluation eval = new RegressionEvaluation();
            eval.eval(validationSet.getLabels(), output);

            log("\n" + eval.averageMeanSquaredError());
            log("TestDataLabels: \n" + validationSet.getLabels());
            log("Actual labels: \n" + output);
            log("Trained for: " + epoch + "epochs");
            //log("Trained on dataset: " + dataset);
            log("Saving model");




            model.save(new File(thisModelDir + "valueNetworkMSE" + eval.averageMeanSquaredError()+  ".zip"));
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

        if (args.length > 1)  {
            printHelpString();
        }
//        for (String arg : args) {
//            if (arg.equalsIgnoreCase("-h"))
//                printHelpString();
//            if (arg.equalsIgnoreCase("-s"))
//                showToString = false;
//            if (arg.equalsIgnoreCase("-m"))
//                printSectionSummaries = false;
//        }
        //printHelpString();


        String folderPath = "C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\";//for development
//        Scanner scanner = new Scanner(System.in);
//        File dataFolder = Resources.getDataFolderFromUser(scanner);
        //String folderPath = dataFolder.getAbsolutePath();
        new PolicyNetworkTrainer().train(folderPath + "\\");
    }

    private static void printHelpString() {
        System.err.println("Right now none of these options are supported and only exist as a sort of todo list for the developer\n");
        System.out.println("At most one argument allowed, basically you can switch which \"script\" is run\n");
        System.out.println("Anything input needed will be gathered through the command line and not args\n");
        System.out.println("Use \"-d\" to process raw data into policy network training data. This option generates a whole bunch of images and puts them in train and test directories along with the labels\n");
        System.out.println("Use \"-t\" to train a neural network\n");
        System.exit(0);
    }
}
