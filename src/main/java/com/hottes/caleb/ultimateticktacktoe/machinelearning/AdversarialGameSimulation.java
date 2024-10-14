package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.EvaluatorConfiguration;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.MCTSEvaluator;
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
public class AdversarialGameSimulation implements Runnable {
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
    public AdversarialGameSimulation(File dataFolder, File logFile, String dataMessage, String logMessage, EvaluatorConfiguration evaluatorConfiguration, PrintStream console) {
        this.dataFolder = dataFolder;
        this.logFile = logFile;
        this.basePolicyFilename = "policy$" + dataMessage + "$";
        this.baseValueFilename = "value$" + dataMessage + "$";
        this.logMessage = logMessage;
        this.console = console;
        config = evaluatorConfiguration;
//        if (!logFile.exists()) {
//            try {
//                logFile.createNewFile();
//            } catch (IOException e) {
//                throw new RuntimeException(e);
//            }
//        }

    }

    /**
     * has empty message
     *
     * @param dataFolder the folder in which to save the policy and value data for this game simulation
     * @param logFile    the log file
     */
    public AdversarialGameSimulation(File dataFolder, File logFile, EvaluatorConfiguration configuration, PrintStream console) {
        this(dataFolder, logFile, "", "", configuration, console);
    }


    @Override
    public void run() {
        console.println("starting thread");
        try (PrintStream logger = new PrintStream(logFile)) {
            long startTime = System.currentTimeMillis();
            logger.println("Saving data to: " + dataFolder.getAbsolutePath());
            logger.println("Begin custom message: ");
            logger.println(logMessage);
            logger.println("End custom message");
            //simulate game
            BoardState currentState = new BoardState(3);//here player one stays player one
            //currentState = GameController.getTestState();//can cause error for some reason when running as jar.
            logger.println("board state created");
            logger.println("Inital State: ");
            logger.println(currentState);

            ArrayList<StateDatum> datums = new ArrayList<>();
            logger.println("Made datums array");
            double eval = 0;
            int moves = 0;
            boolean saveData = true;
            logger.println("entering search loop");
            while (true) {
                if (Thread.interrupted()) {
                    logger.println("Thread has been interrupted, leaving while loop. ");
                    saveData = false;
                    console.println("Thread interrupted");
                    break;
                }
                BoardState stateToEval = currentState.getClone();
                if (!currentState.isPlayerOneTurn()) {
                    stateToEval.invertState();
                }
                logger.println("state to eval, currenty is playerone turn=" + currentState.isPlayerOneTurn());
                logger.println(stateToEval);
                MCTSEvaluator  evaluator = new MCTSEvaluator(stateToEval, config);
                logger.println("isntantiated evaluator");
                evaluator.dispalyDialogAfterSearch = false;
                evaluator.log = true;
                evaluator.logger = logger;//when preforming a search the evaluator will use this logger
                logger.println("starting search");
                console.println("starting search");
                UltimateTickTacToeGameAction actionToTake = (UltimateTickTacToeGameAction) evaluator.preformSearch();
                logger.println("search finished");
                console.println("serch finished");
                datums.add(new StateDatum(stateToEval, evaluator.tree.getRoot(), currentState.isPlayerOneTurn(), 3, console));
                console.println("datum added");
                actionToTake.setMarker(currentState.isPlayerOneTurn() ? 1 : -1);
                console.println("action taken");
                currentState.preformAction(actionToTake);
                console.println("action preformed");
                moves++;
                logger.println("current State");
                logger.println(currentState);

                eval = currentState.getEvaluation();
                console.println("evaluation determined");
                if (BoardState.Evaluation.IN_PROGRESS.getlabel() != eval) {
                    //then the game is now over.
                    console.println("Game over");
                    break;
                }
            }
            if (saveData) {
                logger.println("Game ended in: " + moves + " moves");
                //set the evaluations
                double finalEval = eval;
                datums.forEach(stateDatum -> stateDatum.setEvalForThisState(stateDatum.isPlayerOneTurnOriginally ? finalEval : -finalEval));
                logger.println("Actual data created in graphical format: ");
                logger.println(datums);

                DataSet valueNetworkData = getValueNetworkDataset(datums);
                DataSet policyNetworkData = getPolicyNetworkDataset(datums);

                logger.println("Value network data: ");
                logger.println(valueNetworkData);

                logger.println("Policy network data: ");
                logger.println(policyNetworkData);

                long finishTime = System.currentTimeMillis();
                logger.println(dataFolder.getAbsolutePath() + "\\" + basePolicyFilename + finishTime + ".bin");
                policyNetworkData.save(new File(dataFolder.getAbsolutePath() + "\\" + basePolicyFilename + finishTime + ".bin"));
                valueNetworkData.save(new File(dataFolder.getAbsolutePath() + "\\" + baseValueFilename + finishTime + ".bin"));
                logger.println("Took: " + (finishTime - startTime) / 1000 + " s to simulate game");
                String str = "Finished simulating game, see log at: " + logFile.getAbsolutePath() + " for details.";
                logger.println(str);
                console.println(str);
                TrainingDataCreator.incrementCompletedGames();
            } else {
                String str = "Simulation interuppted, shutting down without saving data. ";
                logger.println(str);
                console.println(str);
            }


        } catch (Exception e) {

            console.println(e);
            throw new RuntimeException(e);
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
