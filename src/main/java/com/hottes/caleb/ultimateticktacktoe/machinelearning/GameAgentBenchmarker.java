package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameState;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.AdversarialGameSimulation;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.GameSimulationResult;
import org.nd4j.shade.wstx.io.EBCDICCodec;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.sql.SQLOutput;
import java.util.EventListener;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * this class
 */
public class GameAgentBenchmarker {

    // Reset
    public static final String RESET = "\u001B[0m";

    // Color
    public static final String GREEN = "\u001B[32m";
    private int completedGames = 0;



    private final PrintStream logger;
    private final EvaluatorConfiguration playerOneConfig;
    private final EvaluatorConfiguration playerTwoConfig;
    private final int numGames;
    private final int numThreads;
    private int gamesWonByOne = 0;
    private int gamesWonByTwo = 0;
    private int drawGames = 0;
    private final PrintStream console = System.out;

    public static void main(String[] args) {
        try {
            EvaluatorConfiguration playerOneConfig = EvaluatorConfiguration.getInstance(new File("C:\\Users\\Caleb\\IdeaProjects\\UltimateTickTacToe\\evaluators\\good.zip"));
            System.out.println(playerOneConfig.computeTime());
            EvaluatorConfiguration playerTwoConfig = new EvaluatorConfiguration(2, 1000, 10, 1, 0, false, false, Optional.empty());
            new GameAgentBenchmarker(playerTwoConfig, playerOneConfig, 20, new PrintStream("F:\\asdf.log"), 20).benchMarkAgents();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }



    }

    public GameAgentBenchmarker(EvaluatorConfiguration playerOneConfig, EvaluatorConfiguration playerTwoConfig, int numGames, PrintStream logger, int numThreads) {
        this.logger = logger;
        this.playerOneConfig = playerOneConfig;
        this.playerTwoConfig = playerTwoConfig;
        this.numGames = numGames;
        this.numThreads = numThreads;
    }

    public void benchMarkAgents() {
        gamesWonByOne = 0;
        gamesWonByTwo = 0;
        drawGames = 0;
        completedGames = 0;
        try (ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads)) {
            for (int i = 1; i <= numGames; i++) {
                executor.submit(new BenchmarkingGameSimulation(playerOneConfig, playerTwoConfig, logger, i));

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
                            executor.submit(new BenchmarkingGameSimulation(playerOneConfig, playerTwoConfig, logger, i));

                        }
                    }

                }
                log("Completed: " + completedGames + " games out of " + numGames, logger);
                try {
                    Thread.sleep(10000);
                } catch (InterruptedException _) {

                }

            }
        }

        log("", logger);
        log("Player one won " + gamesWonByOne + " games (" + (double) gamesWonByOne / numGames * 100 + " %)", logger);
        log("Player two won " + gamesWonByTwo + " games (" + (double) gamesWonByTwo / numGames * 100 + " %)", logger);
        log(drawGames + " games ended in a draw. (" + (double) drawGames / numGames * 100 + " %)", logger);


    }

    private synchronized void gameCompleted(GameSimulationResult result, int game) {
        if (result != null) {
            if (result.finalEval() == BoardState.Evaluation.PLAYER_ONE_WIN.getlabel()) {
                log("Player One won game " + game, logger);
                gamesWonByOne++;
            } else if (result.finalEval() == BoardState.Evaluation.PLAYER_TWO_WIN.getlabel()) {
                log("Player Two won game " + game, logger);
                gamesWonByTwo++;
            }else {
                log("Game " + game + " ended in a draw", logger);
                drawGames++;
            }
            completedGames++;
        }
    }

    private class BenchmarkingGameSimulation implements Runnable{

        private AdversarialGameSimulation sim;
        private final int gamenum;
        public BenchmarkingGameSimulation(EvaluatorConfiguration playerOneConfig, EvaluatorConfiguration playerTwoConfig, PrintStream logger, int gameNUm) {
            sim = new AdversarialGameSimulation(logger, playerOneConfig, playerTwoConfig, System.out);
            this.gamenum = gameNUm;
        }

        @Override
        public void run() {
            GameSimulationResult result = sim.run();
            gameCompleted(result, gamenum);//maybe could use one liner but want to make sure synchronization doesn't gumm things up
        }
    }

    private void log(String message, PrintStream logger) {
        System.setOut(console);
        System.out.println(GREEN + message + RESET);
        logger.println(message);
    }

}
