package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts.NodeData;
import com.hottes.caleb.ultimateticktacktoe.generictree.GenericTreeNode;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;

import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.Optional;

import static com.hottes.caleb.ultimateticktacktoe.Resources.getIndex;

/**
 * contains data about a particular game state that can be used to train both the policy network and the value network.
 * this is an intermediate data structure used to hold data during the playout of a game and it also contains code for taking other objects and extracting data from them
 */
public class StateDatum {

    /**
     * the probabilities that MCTS thinks we should take for each position on the board.
     * all illegal moves are automatically zero and the rest are the number of visits/numParentVisits
     */
    final double[] policyNetworkOutput;
    /**
     * the state of the board from the player whose turn it is to move's perspective.
     * so if its player 2's turn we need to invert the board. this variable also has 9 more numbers representing boolean of if each board is active or not.
     */
    final double[] valueNetworkInput;
    final boolean isPlayerOneTurnOriginally;
    final BoardState boardStateForPrinting;
    private final int boardSize;
    private double evalForThisState = 0;

    /**
     * takes in a board state and root tree node and extracts as much data as possible
     * instantiates a value network inputs array as well as extracting the MCTS probabilities for each cell and storing them in a policy network input vector.
     * the board state should always have the player who is about to play be player 1
     * mutlithreading searches can cause issues with child duplication, this constructor just takes the first valid child so it is best to run the search single threaded or fix the bug when using this.
     *
     * @param theBoardState           the board state that was just evaluated
     * @param rootNode                the root of the tree representing the evaluation
     * @param originalIsPlayerOneTurn whose turn it was origianlly (which may not be reflected correctly in the provided board state)
     */
    public StateDatum(BoardState theBoardState, GenericTreeNode<NodeData> rootNode, boolean originalIsPlayerOneTurn, int boardSize) {

        valueNetworkInput = theBoardState.getValueNetworkInputVector();
        policyNetworkOutput = new double[(int) Math.round(Math.pow(boardSize, 4))];
        isPlayerOneTurnOriginally = originalIsPlayerOneTurn;
        boardStateForPrinting = theBoardState;
        this.boardSize = boardSize;
        //make sure the plyer who is to play is represented by 1
        if (!theBoardState.isPlayerOneTurn()) {
            theBoardState.invertState();
        }
        for (int majRow = 0; majRow < boardSize; majRow++) {
            for (int majCol = 0; majCol < boardSize; majCol++) {
                for (int minRow = 0; minRow < boardSize; minRow++) {
                    for (int minCol = 0; minCol < boardSize; minCol++) {
                        //for all the cells on the main board
                        //figure if this spot is a legal action
                        UltimateTickTacToeGameAction actionToGetHere = new UltimateTickTacToeGameAction(majRow, majCol, minRow, minCol, 1);//marker always 1 becuase this is from the perspective of the player who is to play
                        if (!theBoardState.isLegal(actionToGetHere)) {
                            //if the action ain't legal, then we know the prob is 0
                            policyNetworkOutput[getIndex(majRow, majCol, minRow, minCol, boardSize)] = 0;
                            continue;
                        }
                        //I will assume that MCTS worked right and there is only one child item per child state. this should be the case, but multithreading is also a thing. To improve data accuracy, make sure that duplicate child race condition bug does not occur or that the search is preformed
                        //on a single thread.

                        //figure out which child if any took this action to get here.
                        Optional<GenericTreeNode<NodeData>> possibleChild = rootNode.getChildren().stream().filter(nodeDataGenericTreeNode -> nodeDataGenericTreeNode.getData().getActionTaken().equals(actionToGetHere)).findFirst();
                        policyNetworkOutput[getIndex(majRow, majCol, minRow, minCol, boardSize)] = possibleChild.map(nodeDataGenericTreeNode -> (double) nodeDataGenericTreeNode.getData().getNumVisits() / rootNode.getData().getNumVisits()).orElse(0.0);

                    }
                }

            }
        }


    }

    public StateDatum(BoardState theBoardState, GenericTreeNode<NodeData> rootNode, boolean originalIsPlayerOneTurn, int boardSize, PrintStream asdf) {
        asdf.println("constructing state datum");
        valueNetworkInput = theBoardState.getValueNetworkInputVector();
        asdf.println(2);
        policyNetworkOutput = new double[(int) Math.round(Math.pow(boardSize, 4))];
        asdf.println(3);
        isPlayerOneTurnOriginally = originalIsPlayerOneTurn;
        asdf.println(4);
        boardStateForPrinting = theBoardState;
        asdf.println(5);
        this.boardSize = boardSize;
        asdf.println(6);
        //make sure the plyer who is to play is represented by 1
        if (!theBoardState.isPlayerOneTurn()) {
            theBoardState.invertState();
        }
        asdf.println(7);
        for (int majRow = 0; majRow < boardSize; majRow++) {
            for (int majCol = 0; majCol < boardSize; majCol++) {
                for (int minRow = 0; minRow < boardSize; minRow++) {
                    for (int minCol = 0; minCol < boardSize; minCol++) {
                        //for all the cells on the main board
                        //figure if this spot is a legal action
                        asdf.println(8);
                        UltimateTickTacToeGameAction actionToGetHere = new UltimateTickTacToeGameAction(majRow, majCol, minRow, minCol, 1);//marker always 1 becuase this is from the perspective of the player who is to play
                        asdf.println(9);
                        if (!theBoardState.isLegal(actionToGetHere)) {
                            //if the action ain't legal, then we know the prob is 0
                            policyNetworkOutput[getIndex(majRow, majCol, minRow, minCol, boardSize)] = 0;
                            continue;
                        }
                        asdf.println(10);
                        //I will assume that MCTS worked right and there is only one child item per child state. this should be the case, but multithreading is also a thing. To improve data accuracy, make sure that duplicate child race condition bug does not occur or that the search is preformed
                        //on a single thread.

                        //figure out which child if any took this action to get here.
                        Optional<GenericTreeNode<NodeData>> possibleChild = rootNode.getChildren().stream().filter(nodeDataGenericTreeNode -> nodeDataGenericTreeNode.getData().getActionTaken().equals(actionToGetHere)).findFirst();
                        asdf.println(11);
                        policyNetworkOutput[getIndex(majRow, majCol, minRow, minCol, boardSize)] = possibleChild.map(nodeDataGenericTreeNode -> (double) nodeDataGenericTreeNode.getData().getNumVisits() / rootNode.getData().getNumVisits()).orElse(0.0);
                        asdf.println(12);

                    }
                }

            }
        }

        asdf.println("state datum cosntruvted");
    }


    public double getEvalForThisState() {
        return evalForThisState;
    }

    public void setEvalForThisState(double evalForThisState) {
        this.evalForThisState = evalForThisState;
    }

    @Override
    public String toString() {
        return "StateDatum{" +
                "MCTSProbabilites=" + getProbString() +
                ", state=" + getStateString() +
                ", evalForThisState=" + evalForThisState +
                ", isPlayerOneTurnOriginally=" + isPlayerOneTurnOriginally +
                '}';
    }


    private String getStateString() {
        DecimalFormat format = new DecimalFormat(" .;-.");
        StringBuilder builder = new StringBuilder();
        builder.append("\n");
        for (int majorRow = 0; majorRow < boardSize; majorRow++) {
            for (int minorRow = 0; minorRow < boardSize; minorRow++) {
                for (int majorCol = 0; majorCol < boardSize; majorCol++) {
                    for (int minorCol = 0; minorCol < boardSize; minorCol++) {
                        builder.append("|").append(valueNetworkInput[Resources.getActivityIndex(majorRow, majorCol, boardSize)] == 1 ? "*" : " ").append(format.format(valueNetworkInput[getIndex(majorRow, majorCol, minorRow, minorCol, boardSize)])).append(" ");
                    }
                    builder.append("|");

                }
                builder.append("\n");
            }
            if (majorRow != boardSize - 1) {
                builder.append("|=====+=====+=====||=====+=====+=====||=====+=====+=====|\n");
            } else {
                builder.append("\n");
            }
        }
        return builder.toString();
    }

    private String getProbString() {
        DecimalFormat format = new DecimalFormat(" .####;-.####");
        StringBuilder builder = new StringBuilder();
        builder.append("\n");
        for (int majorRow = 0; majorRow < 3; majorRow++) {
            for (int minorRow = 0; minorRow < 3; minorRow++) {
                for (int majorCol = 0; majorCol < 3; majorCol++) {
                    for (int minorCol = 0; minorCol < 3; minorCol++) {
                        builder.append("|").append(" ").append(format.format(policyNetworkOutput[getIndex(majorRow, majorCol, minorRow, minorCol, 3)])).append(" ");
                    }
                    builder.append("|");

                }
                builder.append("\n");
            }
            if (majorRow != 3 - 1) {
                builder.append("|========+========+========||========+========+========||========+========+========|\n");
            } else {
                builder.append("\n");
            }
        }
        return builder.toString();
    }
}
