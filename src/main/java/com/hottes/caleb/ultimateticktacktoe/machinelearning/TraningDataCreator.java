package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Objects;
import java.util.Scanner;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class TraningDataCreator {


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

//        DataSet data = new DataSet();
//        data.load(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\data\\policy$series1gameNum0$1728452105059.bin"));
//        System.out.println(data);

        PrintStream console = System.out;
        EvaluatorConfiguration config = new EvaluatorConfiguration(2, 1000, computeTime, 1, 0, false, false);
        //String description = " This data series is using a raw MCTS search to generate data." + config + " using 20 threads to sim 1000 games is the goal. ";
        try (ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads)) {
            for (int i = gameStart; i < numGames + 1; i++) {
                executor.submit(new AdversarialGameSimulation(simFolder, new File(logFolder.getAbsolutePath() + "/game" + i + "$" + System.currentTimeMillis() + ".log"), seriesName + ",gameNum" + i, "DataSeries: " + seriesName + " game number: " + i + " \n\nDescription: " + description + ". \n\nEvaluator config=" + config, config));

            }
            while (executor.getCompletedTaskCount() < numGames) {


                console.println("Completed: " + executor.getCompletedTaskCount() + " games out of " + numGames);
                console.println("Press \"q\" to quit (up to 10 sec response delay).");
                console.println("Sleeping");
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException _) {
                    console.println("Finished sleeping");
                }
                if (scanner.hasNextLine()) {
                    if (scanner.nextLine().equalsIgnoreCase("q")) {
                        console.print("Are you sure? Type quickly and exactly \"I am sure\"");
                        String nextLine = "";
                        long startTime = System.currentTimeMillis();
                        while (!scanner.hasNextLine()) {
                            if ((System.currentTimeMillis() - startTime) > 10000) {
                                break;
                            }
                        }
                        nextLine = scanner.nextLine();
                        if (Objects.equals(nextLine, "I am sure")) {
                            console.println("Shutting down");
                            executor.shutdownNow();
                            numGames = 0;//exit while loop
                        } else {
                            System.out.println("Shutdown aborted, continuing. ");
                        }

                    }
                }
                console.println("next iter: numGames: " + numGames + " Competed tasks: " + executor.getCompletedTaskCount());
            }
        }


    }


}
