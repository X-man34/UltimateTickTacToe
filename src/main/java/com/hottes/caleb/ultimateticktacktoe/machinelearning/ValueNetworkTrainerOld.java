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
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.concurrent.TimeUnit;

public class ValueNetworkTrainerOld extends NetworkTrainer {



    @Override
    public void train(String baseDir) {


        try {

//            DataSet dataFromFile = getInputDataSetFromRawFilepath(baseDir, "series1", "value");
//            dataFromFile.save(new File(baseDir + "series1DataCombinedValue.bin"));
//            System.exit(0);

            String thisModelDir = baseDir + "models\\valueNetwork" + System.currentTimeMillis() + "\\";
            new File(thisModelDir).mkdirs();
            outStream = new PrintStream(thisModelDir + "trainingLog.log");
            String dataset = baseDir + "series1DataCombinedValue.bin";




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
            model.addListeners(new CheckpointListener.Builder(thisModelDir).saveEvery(30, TimeUnit.SECONDS).keepLast(2).build());

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




            model.save(new File(thisModelDir + "valueNetworkMSE" + eval.averageMeanSquaredError() + ".zip"));
            log("Model Saved\n\n\nConfiguration\n\n\n");
            outStream.println(configuration.toJson());//dont' want this going to the console
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        System.exit(0);

    }




    public static void main(String[] args) throws FileNotFoundException {
        new ValueNetworkTrainerOld().train("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\");
    }

}




