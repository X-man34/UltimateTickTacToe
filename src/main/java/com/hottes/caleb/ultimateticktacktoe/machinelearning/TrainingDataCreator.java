package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Objects;
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
        EvaluatorConfiguration config = new EvaluatorConfiguration(2, 1000, computeTime, 1, 0, false, false);
        //String description = " This data series is using a raw MCTS search to generate data." + config + " using 20 threads to sim 1000 games is the goal. ";
        try (ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads)) {
            for (int i = gameStart; i < numGames + 1; i++) {
                executor.submit(new AdversarialGameSimulation(simFolder, new File(logFolder.getAbsolutePath() + "/game" + i + "$" + System.currentTimeMillis() + ".log"), seriesName + ",gameNum" + i, "DataSeries: " + seriesName + " game number: " + i + " \n\nDescription: " + description + ". \n\nEvaluator config=" + config, config, console));

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
                            executor.submit(new AdversarialGameSimulation(simFolder, new File(logFolder.getAbsolutePath() + "/game" + i + "$" + System.currentTimeMillis() + ".log"), seriesName + ",gameNum" + i, "DataSeries: " + seriesName + " Game submitted because inactive thread pool detected but completed game count not met consider changing game number, a task might have run into an error game number: " + i + " \n\nDescription: " + description + ". \n\nEvaluator config=" + config, config, console));

                        }
                    }

                }
                console.println("Completed: " + completedGames + " games out of " + numGames);
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException _) {

                }

                console.println("finished sleeping: ");
            }
        }

        console.println("Done creating training data");
    }

    protected static void incrementCompletedGames() {
        completedGames++;
    }


}
