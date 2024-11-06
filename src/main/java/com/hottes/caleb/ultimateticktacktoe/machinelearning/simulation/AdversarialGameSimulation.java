package com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
import com.hottes.caleb.ultimateticktacktoe.ui.GameController;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.PrintStream;
import java.util.ArrayList;

/**
 * simulates a UTTT game and saves the data from it to a specific folder.
 * value network data is named: "value&lt;message&gt;&lt;current time millis&gt;.bin" and policy data is named "policy&lt;message&gt;&lt;current time millis&gt;.bin"
 * For any given game sim the corresponding value and policy files will have the same timestamp.
 * you can specify a message as well so differentiate data
 */
public class AdversarialGameSimulation {
    private final PrintStream logger;
    private final EvaluatorConfiguration playerOneConfig;
    private final EvaluatorConfiguration playerTwoConfig;
    private final PrintStream console;


    public AdversarialGameSimulation(PrintStream logFileStream, EvaluatorConfiguration playerOneConfig, EvaluatorConfiguration playerTwoConfig, PrintStream console) {
        if (logFileStream == null) {
            this.logger = console;
        }else {
            this.logger = logFileStream;
        }

        this.console = console;
        this.playerOneConfig = playerOneConfig;
        this.playerTwoConfig = playerTwoConfig;


    }


    public GameSimulationResult run() {
        long startTime = System.currentTimeMillis();
        BoardState currentState = new BoardState(3);//here player one stays player one
        //currentState = GameController.getTestState();//can cause error for some reason when running as jar.
        logger.println("Inital State: ");
        logger.println(currentState);

        ArrayList<StateDatum> datums = new ArrayList<>();
        double eval = 0;
        int moves = 0;
        while (true) {
            if (Thread.interrupted()) {
                logger.println("Thread has been interrupted, aborting simulation");
                return null;
            }
            BoardState stateToEval = currentState.getClone();
            if (!currentState.isPlayerOneTurn()) {
                stateToEval.invertState();
            }
            logger.println("state to eval, currenty is playerone turn=" + currentState.isPlayerOneTurn());
            logger.println(stateToEval);
            MCTSEvaluator  evaluator = new MCTSEvaluator(stateToEval, currentState.isPlayerOneTurn()?playerOneConfig:playerTwoConfig);
            evaluator.dispalyDialogAfterSearch = false;
            evaluator.log = true;
            evaluator.logger = logger;//when preforming a search the evaluator will use this logger
            UltimateTickTacToeGameAction actionToTake = (UltimateTickTacToeGameAction) evaluator.preformSearch();
            datums.add(new StateDatum(stateToEval, evaluator.tree.getRoot(), currentState.isPlayerOneTurn(), 3));
            actionToTake.setMarker(currentState.isPlayerOneTurn() ? 1 : -1);
            currentState.preformAction(actionToTake);
            moves++;
            logger.println("current State");
            logger.println(currentState);

            eval = currentState.getEvaluation();
            if (BoardState.Evaluation.IN_PROGRESS.getlabel() != eval) {
                //then the game is now over.
                console.println("Game over");
                break;
            }
        }

        return new GameSimulationResult(datums, moves, eval, System.currentTimeMillis() - startTime, System.currentTimeMillis());


    }




}
