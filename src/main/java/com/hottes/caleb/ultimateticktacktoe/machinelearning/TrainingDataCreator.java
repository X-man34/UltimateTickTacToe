package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.AdversarialGameSimulation;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.GameSimulationResult;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.StateDatum;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Optional;
import java.util.Scanner;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * This class is really a script that is used to simulate games and create training data. When {@link TrainingDataCreator#main(String[])} is run it uses a CLI to gather some
 * quick information about where to save the data, what the name of your data series should be, the starting game number and the number of games you want to simulate, how long MCTS gets to think per move, and the number of threads to use.
 * The data series name is intended to help with data organization. For example, you could choose a name that corresponds with what version of the algorithm is being used to make the data and all data will be marked accordingly.
 * The number of threads has a big impact on data creation speed and CPU usage. Increasing threads does not increase the time it takes to simulate a game, but it does increase
 * the number of games that are simulated in parallel. Once you start creating data you can stop at any time by pressing q and confirming, although the system is kind of janky and slow and unreliable
 * you can also kill it with task manager or whatever, and it won't corrupt any data, unless maybe somehow its in the process of saving data (extremely unlikely.)
 * The data is saved game by game and has to be combined by {@link ValueNetworkTrainer} or similar before use. (unless you only want like 50 data-points)
 */
public class TrainingDataCreator {


    private static int completedGames = 0;
    /**
     * the main script method. Run this to create training data.
     * @param args command line args (not used)
     * @throws IOException if stuff goes wrong lol
     */
    public static void main(String[] args) throws IOException {

        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter filepath of data folder: ");
        String baseFolderInput = scanner.nextLine();

        File dataFolder = new File(baseFolderInput);
        if (!dataFolder.isDirectory() && dataFolder.exists()) {
            System.out.println("Invalid data folder path. Exiting. ");
            System.exit(1);
        }


        //so the data folder either doesn't exist or is actually a folder.
        System.out.print("Enter data series name: ");
        String seriesName = scanner.nextLine();

        File logFolder = new File(dataFolder.getAbsolutePath() + "/logs/" + seriesName);
        if (logFolder.exists()) {
            System.out.print("Data Series already exists, are you sure you want to add to it? (y/N): ");
            String decision = scanner.nextLine();
            if (!decision.equalsIgnoreCase("y")) {
                System.exit(0);
            }
        }
        File simFolder = new File(dataFolder.getAbsolutePath() + "/" + seriesName);

        System.out.print("Enter description (goes at top of log file): ");
        String description = scanner.nextLine();
        System.out.print("Enter game num to start with (>0): ");
        int gameStart = scanner.nextInt();
        gameStart = Math.max(1, gameStart);
        System.out.print("Num games (>0): ");
        int numGames = scanner.nextInt();
        numGames = Math.max(1, numGames);
        System.out.print("Enter compute time (s, >1): ");
        int computeTime = scanner.nextInt();
        computeTime = Math.max(2, computeTime);
        System.out.print("Enter num threads (1-30): ");
        int numThreads = scanner.nextInt();
        numThreads = Math.max(1, numThreads);
        numThreads = Math.min(30, numThreads);

        if (!logFolder.exists()) {
            logFolder.mkdirs();
        }
        if (!simFolder.exists()) {
            simFolder.mkdirs();
        }


        PrintStream console = System.out;
        EvaluatorConfiguration config = new EvaluatorConfiguration(2, 1000, computeTime, 1, 0, false, false, Optional.empty());
        //String description = " This data series is using a raw MCTS search to generate data." + config + " using 20 threads to sim 1000 games is the goal. ";
        try (ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads)) {
            for (int i = gameStart; i < numGames + 1; i++) {
                executor.submit(new TrainingGameSimulation(simFolder, new File(logFolder.getAbsolutePath() + "/game" + i + "$" + System.currentTimeMillis() + ".log"), seriesName + ",gameNum" + i, "DataSeries: " + seriesName + " game number: " + i + " \n\nDescription: " + description + ". \n\nEvaluator config=" + config, config, console));

            }
            while (completedGames < numGames - 1) {

                if (executor.getActiveCount() == 0) {
                    //try again in a sec to make sure they are reallly inactive
                    try {
                        Thread.sleep(1000);
                    } catch (InterruptedException _) {

                    }
                    if (executor.getActiveCount() == 0) {
                        for (int i = numGames; i < numGames + 5; i++) {
                            executor.submit(new TrainingGameSimulation(simFolder, new File(logFolder.getAbsolutePath() + "/game" + i + "$" + System.currentTimeMillis() + ".log"), seriesName + ",gameNum" + i, "DataSeries: " + seriesName + " Game submitted because inactive thread pool detected but completed game count not met consider changing game number, a task might have run into an error game number: " + i + " \n\nDescription: " + description + ". \n\nEvaluator config=" + config, config, console));

                        }
                    }

                }
                console.println("Completed: " + completedGames + " games out of " + numGames);
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException _) {

                }

            }
        }

        console.println("Done creating training data");
    }

    public static void incrementCompletedGames() {
        completedGames++;
    }


    /**
     * The purpose of this class is to abstract the simulation of games and handle the data saving here and make {@link AdversarialGameSimulation} more abstract
     * so it can be used more generally.
     */
    private static class TrainingGameSimulation implements Runnable{

        private final File dataFolder;
        private final File logFile;
        private final String baseValueFilename;
        private final String basePolicyFilename;
        private final String logMessage;
        private final EvaluatorConfiguration config;
        private final PrintStream console;



        /**
         * @param dataFolder  the folder in which to save the policy and value data for this game simulation
         * @param logFile     the log file
         * @param dataMessage the message
         * @param logMessage  a message to put at the top of the log file.
         */
        public TrainingGameSimulation(File dataFolder, File logFile, String dataMessage, String logMessage, EvaluatorConfiguration evaluatorConfiguration, PrintStream console) {
            this.dataFolder = dataFolder;
            this.logFile = logFile;
            this.basePolicyFilename = "policy$" + dataMessage + "$";
            this.baseValueFilename = "value$" + dataMessage + "$";
            this.logMessage = logMessage;
            this.console = console;
            config = evaluatorConfiguration;



        }

        @Override
        public void run() {
            try (PrintStream logger = new PrintStream(logFile)) {
                logger.println("Saving data to: " + dataFolder.getAbsolutePath());
                logger.println("Begin custom message: ");
                logger.println(logMessage);
                logger.println("End custom message");
                AdversarialGameSimulation sim = new AdversarialGameSimulation(logger, config,config, console);//because we are doing training data creation we want to use the same agent for both player one and player two.
                GameSimulationResult results = sim.run();
                //simulate game

                if (results != null) {
                    logger.println("Game ended in: " + results.numMoves() + " moves");
                    //set the evaluations
                    results.datums().forEach(stateDatum -> stateDatum.setEvalForThisState(stateDatum.isPlayerOneTurnOriginally ? results.finalEval() : -results.finalEval()));
                    logger.println("Actual data created in graphical format: ");
                    logger.println(results.datums());

                    DataSet valueNetworkData = getValueNetworkDataset(results.datums());
                    DataSet policyNetworkData = getPolicyNetworkDataset(results.datums());

                    logger.println("Value network data: ");
                    logger.println(valueNetworkData);

                    logger.println("Policy network data: ");
                    logger.println(policyNetworkData);

                    logger.println(dataFolder.getAbsolutePath() + "\\" + basePolicyFilename + results.finishTime() + ".bin");
                    policyNetworkData.save(new File(dataFolder.getAbsolutePath() + "\\" + basePolicyFilename + results.finishTime() + ".bin"));
                    valueNetworkData.save(new File(dataFolder.getAbsolutePath() + "\\" + baseValueFilename + results.finishTime() + ".bin"));
                    logger.println("Took: " + results.simTime() / 1000 + " s to simulate game");
                    String str = "Finished simulating game, see log at: " + logFile.getAbsolutePath() + " for details.";
                    logger.println(str);
                    console.println(str);
                    TrainingDataCreator.incrementCompletedGames();
                } else {
                    String str = "Simulation interrupted, shutting down without saving data. ";
                    logger.println(str);
                    console.println(str);
                }


            } catch (Exception e) {

                console.println(e);
            }



        }

        private DataSet getValueNetworkDataset(ArrayList<StateDatum> data) {
            double[][] inputs = new double[data.size()][data.getFirst().valueNetworkInput.length];
            double[][] outputs = new double[data.size()][1];
            for (int i = 0; i < data.size(); i++) {
                inputs[i] = data.get(i).valueNetworkInput;
                outputs[i] = new double[]{data.get(i).getEvalForThisState()};
            }
            return new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        }

        private DataSet getPolicyNetworkDataset(ArrayList<StateDatum> data) {
            double[][] inputs = new double[data.size()][data.getFirst().valueNetworkInput.length];
            double[][] outputs = new double[data.size()][data.getFirst().policyNetworkOutput.length];
            for (int i = 0; i < data.size(); i++) {
                inputs[i] = data.get(i).valueNetworkInput;
                outputs[i] = data.get(i).policyNetworkOutput;
            }
            return new DataSet(Nd4j.create(inputs), Nd4j.create(outputs));
        }

    }


}
