package com.hottes.caleb.ultimateticktacktoe.gameindependant;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.Resources;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts.BitSetBasedUTTTNodeData;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts.NodeData;
import com.hottes.caleb.ultimateticktacktoe.generictree.GenericTree;
import com.hottes.caleb.ultimateticktacktoe.generictree.GenericTreeNode;
import com.hottes.caleb.ultimateticktacktoe.machinelearning.simulation.StateDatum;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;
import javafx.application.Platform;
import javafx.scene.control.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

import static com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getIndex;

/**
 * a MCTS evalutor takes in a inital state and returns the best move for whichever player is denoted by the marker 1.
 * whatever is creating the evaluator needs to take care that the markers mean the right thing.
 */
public class MCTSEvaluator {
    public final long maxRolloutDepth;//make sure rollouts don't get stuck in infinite loop
    public final int THREADS;
    public final int STUPIDITY;
    public final boolean allowForcePlay;
    private final boolean useMaxCPU;
    private final int computeTime;
    public double C = 2;
    private final Optional<MultiLayerNetwork> optionalValueNetwork;
    private final Optional<MultiLayerNetwork> optionalPolicyNetwork;



    public static Random rand = new Random();
    public final GenericTree<NodeData> tree = new GenericTree<>();
    private final GameState initalState;
    public PrintStream logger = System.out;
    public boolean dispalyDialogAfterSearch = true;
    public boolean log = true;
    private ThreadPoolExecutor executor;
    private long completedTasks = 0;
    private boolean searching = false;
    private int maxDepth = 0;
    private long positionsSearched = 0;
    private long childCreationRaceConditions = 0;
    private long traversalTimeSum = 0;
    private int traversalCount;


    private ArrayList<String> warnings = new ArrayList<>();
    public MCTSEvaluator(GameState initalState) {
        this(initalState, Resources.DEFAULT_EVALUATOR_CONFIGURATION);
    }

    public MCTSEvaluator(GameState initalState, EvaluatorConfiguration configuration) {
        tree.setRoot(new GenericTreeNode<>(new NodeData(1, 0, initalState, null)));

        C = configuration.cValue();
        this.THREADS = configuration.threads();
        this.STUPIDITY = configuration.stupidity();
        this.allowForcePlay = configuration.allowForcePlay();
        this.maxRolloutDepth = configuration.maxRolloutDepth();
        this.initalState = initalState;
        computeTime = configuration.computeTime();
        useMaxCPU = configuration.maxMyCPU();
        optionalValueNetwork = configuration.valueNetwork();
        optionalValueNetwork.ifPresent(MultiLayerNetwork::init);
        optionalPolicyNetwork = configuration.policyNetwork();
        optionalPolicyNetwork.ifPresent(MultiLayerNetwork::init);
        addChildren(tree.getRoot());


    }


    public GameAction preformSearch() {

        System.setOut(logger);
        if (!useMaxCPU && (THREADS <= 1)) {
            return singleThreadedSearch();
        }
        warnings.clear();
        searching = true;

        if (useMaxCPU) {
            executor = (ThreadPoolExecutor) Executors.newCachedThreadPool();
        } else {
            executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(THREADS);
        }

        long startTime = System.currentTimeMillis();
        //long submitted = 0;
        //create a ridiculous amount of tasks, more than the user is allowed to ask for and then if it takes long enough, or the complted iteration count is big enough, stop
        completedTasks = 0;
        maxDepth = 0;
        positionsSearched = 0;
        childCreationRaceConditions = 0;
        traversalCount = 0;
        traversalTimeSum = 0;
        long submittedTasks = 0;
        while (searching) {
            //check if we have satisfied the end condition.
            if ((System.currentTimeMillis() - startTime) >= (computeTime * 1000L)) {
                //then its been long enough
                searching = false;
                executor.shutdownNow();

            }
            if (Thread.interrupted()) {
                logger.println("Ending search becuase thread is interrupted. ");
                break;
            }
            try {
                Thread.sleep(1);//don't want to overload the CPU with excessive looping
            } catch (InterruptedException _) {

            }
            //maintains a steady flow of tasks and keeps the thread pool running at maximum capability.
            //so if the threads are limited it will roughly correlate to a cpu useage, if its a cached thread pool then it should max out cpu always.
            long taskDeficit = getCompletedTasks() - submittedTasks + 25;
            for (int i = 0; i < taskDeficit; i++) {
                submittedTasks++;
                executor.execute(new PreformBatchOfIterationsTask(Resources.MULTTHREADED_BATCH_SIZE, tree.getRoot()));
            }
        }
        if (log) {
            System.out.println("Determined best move using roughly: " + getCompletedTasks() * Resources.MULTTHREADED_BATCH_SIZE + " iterations");
            System.out.println("Searched as far ahead as: " + maxDepth + " moves");
            double avgTraversalTime = (double) traversalTimeSum / traversalCount;
            System.out.println("Average traversal time: " + avgTraversalTime + " ms");
            System.out.println("Searched: " + positionsSearched + " states");
            System.out.println("Detected: " + childCreationRaceConditions + " node expansion race conditions");
            if (!warnings.isEmpty()) {
                System.out.println("Warnings while preforming search: " + warnings);
            }
        }

        if (dispalyDialogAfterSearch) {
            Platform.runLater(this::displayDialog);
        }
        return getCurrentBestMove();
    }

    private GameAction singleThreadedSearch() {
        warnings.clear();
        searching = true;
        traversalTimeSum = 0;
        traversalCount = 0;
        long startTime = System.currentTimeMillis();
        int iterations = 0;
        while (searching) {
            preformIteration(tree.getRoot());
            iterations += 1;
            if ((System.currentTimeMillis() - startTime) >= (computeTime * 1000L)) {//convert from s to ms
                searching = false;
            } else if (iterations >= Resources.EVALUATION_ITERATION_HARD_LIMIT) {
                searching = false;
            }

        }
        if (log) {
            System.out.println("Determined best move using: " + iterations + "Iterations");
            double avgTraversalTime = (double) traversalTimeSum / traversalCount;
            System.out.println("Average traversal time: " + avgTraversalTime + " ms");
            if (!warnings.isEmpty()) {
                System.out.println("Warnings while preforming search: " + warnings);
            }

        }

        if (dispalyDialogAfterSearch) {
            Platform.runLater(() -> displayDialog());
        }
        return getCurrentBestMove();
    }

    /**
     * Implements the monte carlo tree search algorithm and preforms an iteration on a node
     * typically the node passes is the root node, however in multithreaded contexts each of the inital children may be explored seperatly
     *
     * @see <a href="https://www.youtube.com/watch?v=UXW2yZndl7U">Really good MCTS exlpanation</a>
     */
    public void preformIteration(GenericTreeNode<NodeData> currentNode) {
        //traverse tree to leaf node using UCB1 algorithm
        int depth = 0;
        long traversalStart = System.currentTimeMillis();
        while (currentNode.hasChildren()) {
            currentNode = getBestChild(currentNode);
            depth++;
        }
        if (depth > maxDepth) {
            setMaxDepth(depth);
        }
        traversalTimeSum += System.currentTimeMillis() - traversalStart;
        traversalCount++;
        double scoreToAdd;
        //now that we have traversed the tree we are deep in and it is unlikly that another thread will have to wait for this node to unlock.
        synchronized (currentNode) {
            //we have now traversed the tree to a node that is a leaf because it has no children.

            if (currentNode.getData().getNumVisits() == 0) {
                //this node has not been visited yet, so preform a rollout and
                scoreToAdd = evaluateEndPoint(currentNode.getData().getGameState());
            } else {
                //this node has been visited but has no children yet, so determine what all its children are (if any) and roll one of them out
                addChildren(currentNode);
                if (currentNode.hasChildren()) {
                    incrementPositionsSearched();
                    currentNode = currentNode.getChildAt(0);//we could also try picking a random child so the order in which the action algorithm returns the actions does not bias which part of the board we often explore.
                    scoreToAdd = evaluateEndPoint(currentNode.getData().getGameState());
                } else {
                    //then we must have reached a terminal state because we tried to
                    scoreToAdd = currentNode.getData().getGameState().getEvaluation();
                }
            }
            //now we have successfully expanded the tree
            //time to backpropagate and add stuff.
            currentNode.getData().incrementNumVisits();
            currentNode.getData().changeTotalScore(scoreToAdd);
        }

        //this doesn't need to be synchronized I think even if another thread modifies one of the nodes before an iteration of this loop finishes, we shouldn't care becuase we are just adding to totals.
        while (currentNode.getParent() != null) {//this will get to the second to last node which will have the root as its parent. It will increment the root and be done.
            currentNode = currentNode.getParent();
            currentNode.getData().incrementNumVisits();
            scoreToAdd = -scoreToAdd;//invert so it reflects who we are talking about
            currentNode.getData().changeTotalScore(scoreToAdd);
        }
    }



    /**
     * if there is a policy network for the config present than that is used to decide on an action via the given probability distribution.
     * for details see {@link MCTSEvaluator#getChildToExploreFromPolicyNetworkOutput(INDArray, BoardState, int)}
     *if there is no policy network available or the best child cannot be found, UCB is used as a default.
     * @param currentNode
     * @return
     */
    private GenericTreeNode<NodeData> getBestChild(GenericTreeNode<NodeData> currentNode) {
        if (optionalPolicyNetwork.isPresent()) {
            BoardState state =  (BoardState) currentNode.getData().getGameState();
            UltimateTickTacToeGameAction actionToTake = getChildToExploreFromPolicyNetworkOutput(optionalPolicyNetwork.get().output(com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getPolicyNetworkInputV1( (BoardState) currentNode.getData().getGameState())), state, state.getBoardSize());
            for (GenericTreeNode<NodeData> child : currentNode.getChildren()) {
                if (child.getData().getActionTaken() == actionToTake) {
                    return child;
                }
            }
        }
        return getBestChildViaUCB(currentNode);


    }

    /**
     * This method uses the output from the policy network to determine which legal action to explore.
     * it does this by treating the output from the neural network as a probability mass function and choosing a random move accordingly.
     * before using the pmf illegal moves are set to 0 and the data is scaled.
     * @return
     */
    public UltimateTickTacToeGameAction getChildToExploreFromPolicyNetworkOutput(INDArray output, BoardState currentState, int boardSize) {
        //make all illegal moves 0
        for (int majorRow = 0; majorRow < boardSize; majorRow++) {
            for (int minorRow = 0; minorRow < boardSize; minorRow++) {
                for (int majorCol = 0; majorCol < boardSize; majorCol++) {
                    for (int minorCol = 0; minorCol < boardSize; minorCol++) {
                        UltimateTickTacToeGameAction actionHere = new UltimateTickTacToeGameAction(majorRow, majorCol, minorRow, minorCol, currentState.isPlayerOneTurn() ? 1 : -1);
                        if (!currentState.isLegal(actionHere)) {
                            output.put(0, getIndex(majorRow, majorCol, minorRow, minorCol, boardSize), 0);
                        }


                    }

                }
            }
        }
        //normalize so its a valid pmf
        output = output.div(output.sum(1));
        UltimateTickTacToeGameAction action = getActionFromPMF(output.getRow(0).toDoubleVector(), currentState.isPlayerOneTurn() ? 1 : -1);
        if (!currentState.isLegal(action)) {
            System.out.println("Defaulting to first legal action");
            return (UltimateTickTacToeGameAction) currentState.getActions().getFirst();
        }else {
            return action;
        }

    }

    private UltimateTickTacToeGameAction getActionFromPMF(double[] pmf, double marker) {
        double[] cumulativeDistribution = new double[pmf.length];
        cumulativeDistribution[0] = pmf[0];
        for (int i = 1; i < pmf.length; i++) {
            cumulativeDistribution[i] = cumulativeDistribution[i - 1] + pmf[i];
        }
        double rand = Math.abs(Math.random() - .000001);//if it ever is 1 then it could return illegal values
        //binary search for value
        int low = 0;
        int high = cumulativeDistribution.length - 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            if (rand < cumulativeDistribution[mid]) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getActionFromFlatIndex(low, marker, 3);
    }


    /**
     * uses the upper confidence bound function to determine what the best child is.
     *
     * @param currentNode the parent to consider
     * @return the best child
     * @see MCTSEvaluator#getUCB1(GenericTreeNode)
     */
    private GenericTreeNode<NodeData> getBestChildViaUCB(GenericTreeNode<NodeData> currentNode) {
        double bestUCB1 = Double.NEGATIVE_INFINITY;
        GenericTreeNode<NodeData> bestChild = null;//will not produce null pointer because of context if algorithm is implemented correctly.

        synchronized (currentNode) {
            for (GenericTreeNode<NodeData> child : currentNode.getChildren()) {
                double newUCB = getUCB1(child);
                if (newUCB > bestUCB1) {
                    bestChild = child;
                    bestUCB1 = newUCB;
                    if (bestUCB1 == Double.POSITIVE_INFINITY) {
                        break;//no use looking for better children when we won't find any
                    }
                }

            }
        }
        return bestChild;
    }



    /**
     * when confronted with a set of nodes to choose from, whichever node maximizes this function is the node that should be investigated.
     * this formula is the average evalulation of this state over each time it has been visited plus c * sqrt(ln(parentVisits)/visits)
     * this formula is designed to strike a balance between exploration of uninvestigated moves and further examining moves that have already shown good potential.
     * That balance is determined through the C value
     * <p>
     * for preformance reasons this method assumes that the supplied node has a parent and that the parent node has been visited a positive number of times.
     *
     * @param node the node we are considering
     * @return a double representing how interesting this node is.
     */
    double getUCB1(GenericTreeNode<NodeData> node) {
        if (node.getData().getNumVisits() <= 0) {//>= as opposed to == in case somehow this node has a negative value this would prevent a negative in the square root function.
            return Double.POSITIVE_INFINITY;//we can't divide by zero and this is the intended behaviour
        }
        return (node.getData().getTotalScore() / node.getData().getNumVisits()) + (C * Math.sqrt(Math.log(node.getParent().getData().getNumVisits()) / node.getData().getNumVisits()));

    }




    /**
     * expands the given node. the game actions generated
     *
     * @param parent
     */
    private void addChildren(GenericTreeNode<NodeData> parent) {
        //multithreading can cause this method to be called even when it parent already has children.
        if (parent.hasChildren()) {
            childCreationRaceConditions++;
        }
        if (parent.getData().getGameState().getEvaluation() != 0) {
            //then this is a terminal state. The actions function will return actions if there are empty squares regarless of whether or not we are in a terminal state.
            return;
        }
        double marker = parent.getData().getGameState().isPlayerOneTurn() ? 1 : -1;
        for (GameAction action : parent.getData().getGameState().getActions()) {
            action.setMarker(marker);
            parent.addChild(new GenericTreeNode<>(new BitSetBasedUTTTNodeData(0, 0, parent.getData().getGameState().simulateAction(action, false), action)));
        }
    }




    /**
     * this method is called when MCTS has visited  a new node and wants an evaluation of it. traditionally this is preformed by a random rollout but this function
     * will attempt to load a value network and use it. If anything goes wrong the default behavior is a random rollout
     * @param currentState the state to evaluate
     * @return the evaluation of that state
     */
    private double evaluateEndPoint(GameState currentState) {
        if (optionalValueNetwork.isPresent()) {
            //model is initzlized in constructor
            try {
                return optionalValueNetwork.get().output(com.hottes.caleb.ultimateticktacktoe.machinelearning.Resources.getValueNetworkInputV1( (BoardState) currentState)).getDouble(0,0) * (currentState.isPlayerOneTurn()?1:-1);//the AI always sees the board as if its player 1's turn, so invert the score if otherwise.
            } catch (Exception e) {
                warnings.add("Failed to use value network to make prediction: " + e);
                return preformRollout(currentState);
            }

        }else {
            return preformRollout(currentState);
        }

    }



    /**
     * Preforms a random playout of a game. The game is played until the state becomes terminal. Then the evaluation of this state is returned.
     * determines if player 1 won wins the game or not. whatever calls this should take care to ensure that player 1 is set appropriatly and the board may need to be inverted in that case.
     *
     * @param currentState the game state
     * @return
     */
    private double preformRollout(GameState currentState) {
        boolean isWon = false;
        long turns = 0;
        while (!isWon && (turns < maxRolloutDepth)) {
            //generate possible actions for current state
            ArrayList<GameAction> actions = currentState.getActions();//the marker on these actions is always 1
            if (actions.isEmpty()) {
                //then we have reached a terminal state and random
                break;
            }
            GameAction action = actions.get(rand.nextInt(actions.size()));
            if (!currentState.isPlayerOneTurn())
                action.invertMarker();//then we need to flip the marker so the other player plays
            currentState = currentState.simulateAction(action, false);
            isWon = currentState.getEvaluation() != 0;
            turns++;

        }
        return currentState.getEvaluation();
    }

    /**
     * Looks through the root's children and returns the action taken to reach the child with the highest UCB1 value
     *
     * @return the current estimation for the best move to take.
     */
    public GameAction getCurrentBestMove() {
        double bestVisits = Double.NEGATIVE_INFINITY;
        GameAction bestAction = null;
        for (GenericTreeNode<NodeData> child : tree.getRoot().getChildren()) {
            if (child.getData().getNumVisits() >= bestVisits) {
                bestAction = child.getData().getActionTaken();
                bestVisits = child.getData().getNumVisits();
            }

        }

        return bestAction;
    }





    @Override
    public String toString() {
        return getNodeString(tree.getRoot(), 0, new DecimalFormat("#.#"));
    }
    //recursive method to get string representation of tree
    private static String getNodeString(GenericTreeNode<NodeData> node, int depth, DecimalFormat format) {
        StringBuilder builder = new StringBuilder();
        //add stuff for this node.

        if (depth > 0) {
            builder.repeat(' ', (depth - 1) * 4);
            builder.append("|---");
        }
        builder.append("Node{t=").append(node.getData().getTotalScore()).append(",n=").append(node.getData().getNumVisits()).append(",v=").append(format.format(node.getData().getTotalScore() / node.getData().getNumVisits())).append(",c=").append(node.getNumberOfChildren()).append(",").append(node.getData().getActionTaken()).append("}\n");
        for (GenericTreeNode<NodeData> child : node.getChildren()) {
            if (child.getData().getNumVisits() > 0) {
                //builder.repeat(' ', depth * 4).append("|\n");
                builder.append(getNodeString(child, depth + 1, format));
            }

        }
        return builder.toString();

    }


    public void displayDialog() {
        DecimalFormat format = new DecimalFormat("#.#");
        Dialog<String> dialog = new Dialog<>();
        TreeView<String> treeView = new TreeView<>();
        TreeItem<String> rootItem = new TreeItem<>();
        rootItem.setValue(tree.getRoot().getData().getTreeString(format, tree.getRoot().getNumberOfChildren()));
        treeView.setRoot(rootItem);
        for (GenericTreeNode<NodeData> child : tree.getRoot().getChildren()) {
            addNode(treeView.getRoot(), child, format);
        }
        dialog.getDialogPane().setContent(treeView);
        dialog.setResizable(true);
        dialog.getDialogPane().getButtonTypes().addAll(ButtonType.OK, ButtonType.CANCEL);
        dialog.showAndWait();

    }

    private void addNode(TreeItem<String> parent, GenericTreeNode<NodeData> child, DecimalFormat format) {
        TreeItem<String> newItem = new TreeItem<>(child.getData().getTreeString(format, child.getNumberOfChildren()));
        parent.getChildren().add(newItem);
        for (GenericTreeNode<NodeData> grandChild : child.getChildren()) {
            if (grandChild.getData().getNumVisits() > 0) {
                addNode(newItem, grandChild, format);
            }

        }
    }

    public void forcePlay() {
        if (executor != null) {
            executor.shutdownNow();
        }

        searching = false;
    }

    private synchronized void incrementCompletedTask() {
        completedTasks++;
    }

    private synchronized long getCompletedTasks() {
        return completedTasks;
    }

    private synchronized void setMaxDepth(int depth) {
        maxDepth = depth;
    }

    private void incrementPositionsSearched() {
        positionsSearched++;
    }

    public enum EndCondition {
        TIME,
        ITERATIONS
    }

    private class PreformBatchOfIterationsTask implements Runnable {

        private final int batchSize;
        private final GenericTreeNode<NodeData> nodeToExplore;

        public PreformBatchOfIterationsTask(int theBatchSize, GenericTreeNode<NodeData> node) {
            super();
            this.batchSize = theBatchSize;
            nodeToExplore = node;
        }

        @Override
        public void run() {
            for (int i = 0; i < batchSize; i++) {
                preformIteration(nodeToExplore);
                if (Thread.interrupted()) {
                    return;
                }
            }
            incrementCompletedTask();
        }
    }
}
