package com.hottes.caleb.ultimateticktacktoe.machinelearning;


import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
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
import java.io.IOException;
import java.io.PrintStream;
import java.util.concurrent.TimeUnit;

public class ValueNetworkTrainer extends NetworkTrainer {


    @Override
    public void train(String baseDir) {
        try {
            String thisModelDir = baseDir + "models/valueNetwork_" + System.currentTimeMillis() + "/";
            new File(thisModelDir).mkdirs();
            outStream = new PrintStream(thisModelDir + "trainingLog.log");

            String datasetPath = baseDir + "series1DataCombinedValue.bin";
            DataSet inputData = new DataSet();
            inputData.load(new File(datasetPath));
            inputData.shuffle(42);

            SplitTestAndTrain testAndTrain = splitTestAndTrain(inputData, 0.8);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            int numHiddenNeurons = 100;
            int classCount = inputData.numOutcomes();

            MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                    .seed(2039402)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam.Builder().learningRate(0.001).build())
                    .list()
                    .layer(0, new Convolution3D.Builder().nIn(4).nOut(4).kernelSize(3, 3, 3).padding(1, 1, 1).stride(1, 1, 1).activation(Activation.IDENTITY).dataFormat(org.deeplearning4j.nn.conf.layers.Convolution3D.DataFormat.NDHWC).build())
                    .layer(1, new Convolution3D.Builder().nIn(4).nOut(4).kernelSize(3, 3, 3).padding(1, 1, 1).dataFormat(Convolution3D.DataFormat.NDHWC).build())
                    .layer(2, new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(3, new DenseLayer.Builder().nIn(numHiddenNeurons).nOut(numHiddenNeurons).build())
                    .layer(4, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .nIn(numHiddenNeurons).nOut(classCount).build())
                    .setInputType(InputType.convolutional3D(Convolution3D.DataFormat.NDHWC, 9, 3, 3, 4))
                    .build();

            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            model.init();

            // Setup UI and stats storage
            UIServer uiServer = UIServer.getInstance();
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            model.addListeners(new StatsListener(statsStorage, 1));

            // Setup checkpointing
            CheckpointListener checkpointListener = new CheckpointListener.Builder(thisModelDir)
                    .saveEvery(30, TimeUnit.SECONDS)
                    .keepLast(2)
                    .build();
            model.addListeners(checkpointListener);

            log.info("Beginning training. See web UI at http://localhost:9000/train/overview");

            double bestTestError = Double.MAX_VALUE;
            int epochsWithoutImprovement = 0;
            int maxEpochs = 500;
            int maxEpochsWithoutImprovement = 20;

            for (int epoch = 0; epoch < maxEpochs; epoch++) {
                model.fit(trainingData);

                INDArray output = model.output(testData.getFeatures());
                RegressionEvaluation eval = new RegressionEvaluation();
                eval.eval(testData.getLabels(), output);
                double currentTestError = eval.averageMeanSquaredError();

                log.info("Epoch {}, Test MSE: {}", epoch, currentTestError);

                if (currentTestError < bestTestError) {
                    bestTestError = currentTestError;
                    epochsWithoutImprovement = 0;
                } else {
                    epochsWithoutImprovement++;
                }

                if (epochsWithoutImprovement >= maxEpochsWithoutImprovement) {
                    log.info("Stopping training early due to overfitting. Best Test MSE: {}", bestTestError);
                    break;
                }
            }

            model.save(new File(thisModelDir + "valueNetwork_finalMSE_" + bestTestError + ".zip"));
            log.info("Model Saved. Configuration:\n{}", configuration.toJson());
            outStream.println(configuration.toJson());

        } catch (IOException e) {
            log.error("Error during training", e);
        } finally {
            if (outStream != null) {
                outStream.close();
            }
        }
    }
}


