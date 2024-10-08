package com.hottes.caleb.ultimateticktacktoe.machinelearning;

import com.hottes.caleb.ultimateticktacktoe.BoardState;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.mcts.NodeData;
import com.hottes.caleb.ultimateticktacktoe.generictree.GenericTree;
import com.hottes.caleb.ultimateticktacktoe.generictree.GenericTreeNode;
import com.hottes.caleb.ultimateticktacktoe.ui.PlayerBox;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * contains data about a particular game state that can be used to train both the policy network and the value network.
 */
public class StateDatum {

    /**
     * the probabilities that MCTS thinks we should take for each position on the board.
     * all illegal moves are automatically zero and the rest are the number of visits/numParentVisits
     */
    final double[] MCTSProbabilites;
    /**
     * the state of the board from the player whose turn it is to move's perspective.
     * so if its player 2's turn we need to invert the board.
     */
    final double[] state;

    private double evalForThisState = 0;
    final boolean isPlayerOneTurnOriginally;
    final BoardState boardStateForPrinting;

    public StateDatum(BoardState theBoardState, GenericTreeNode<NodeData> rootNode, boolean originalIsPlayerOneTurn) {
        MCTSProbabilites = new double[81];
        state = new double[81];
        isPlayerOneTurnOriginally = originalIsPlayerOneTurn;
        boardStateForPrinting = theBoardState;
        ArrayList<GameAction> legalActions = theBoardState.getActions();
        //make sure the plyer who is to play is represented by 1
        if (!theBoardState.isPlayerOneTurn()) {
            theBoardState.invertState();
        }
        ArrayList<UltimateTickTacToeGameAction> legalTicTackToeActions = new ArrayList<>();
        legalActions.forEach(action -> legalTicTackToeActions.add((UltimateTickTacToeGameAction) action));
        for (int majRow = 0; majRow < 3; majRow++) {
            for (int majCol = 0; majCol < 3; majCol++) {
                for (int minRow = 0; minRow < 3; minRow++) {
                    for (int minCol = 0; minCol < 3; minCol++) {
                        state[getIndex(majRow, majCol, minRow, minCol)] = theBoardState.getMinorBoardAt(majRow, majCol).itemAt(minRow, minCol);
                        //only fiddle around with slow stuff it the board is active. otherwise, we already know the answer.
                        if (!theBoardState.getMinorBoardAt(majRow, majCol).isActive()) {
                            MCTSProbabilites[getIndex(majRow, majCol, minRow, minCol)] = 0;
                        }else {
                            //for all the probabilities.
                            //figure out if this action is legal at this state
                            int finalMajRow = majRow;
                            int finalMajCol = majCol;
                            int finalMinRow = minRow;
                            int finalMinCol = minCol;
                            Optional<UltimateTickTacToeGameAction> possibleAction = legalTicTackToeActions.stream().filter(action -> (action.x == finalMinCol) && (action.y == finalMinRow) && (action.majorRow == finalMajRow) && (action.majorCol == finalMajCol)).findFirst();

                            possibleAction.ifPresentOrElse(ultimateTickTacToeGameAction -> {
                                //so the spot we are at is a valid move. Figure out what MCTS thought its probability was and assign that to the probabilty for where we are at
                                //uses custom equals to get all the children where this legal action was taken to get there. There should only be one but
                                //when using multithreading there is as of now an unresolved bug where duplicate children can be created.
                                //therefore we will use the sum of the visits of all the children
                                Optional<GenericTreeNode<NodeData>> childrenWithThisAction = rootNode.getChildren().stream().filter(nodeDataGenericTreeNode -> nodeDataGenericTreeNode.getData().getActionTaken().equals(ultimateTickTacToeGameAction)).findFirst();

                                double childVisits = 0;
                                if (childrenWithThisAction.isPresent()) {
                                    childVisits = childrenWithThisAction.get().getData().getNumVisits();
                                }
                                MCTSProbabilites[getIndex(finalMajRow, finalMajCol, finalMinRow, finalMinCol)] = childVisits / rootNode.getData().getNumVisits();
                            }, () -> {
                                //the spot we are at was not a valid move so assign a probabilty of zero
                                MCTSProbabilites[getIndex(finalMajRow, finalMajCol, finalMinRow, finalMinCol)] = 0;
                            });
                        }

                    }
                }

            }
        }


    }

    /**
     * takes in information about a ultimate tick tac toe square and converts it to a linear index.
     * this is so the entire board can be represented as a linear array for vectorization.
     * @param majRow the index of the major row
     * @param majCol the index of the major column
     * @param minRow the index of the minor row
     * @param finalMinCol the index of the minor column
     * @return the linear index of this spot.
     */
    public static int getIndex(int majRow, int majCol, int minRow, int finalMinCol) {
        int majorIndex = majRow * 3 + majCol;
        int minorIndex = minRow * 3 + finalMinCol;
        return majorIndex * 9 + minorIndex;
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
                ", state=" + boardStateForPrinting +
                ", evalForThisState=" + evalForThisState +
                ", isPlayerOneTurnOriginally=" + isPlayerOneTurnOriginally +
                '}';
    }


    public String getProbString() {
        DecimalFormat format = new DecimalFormat(" .####;-.####");
        StringBuilder builder = new StringBuilder();
        builder.append("\n");
        for (int majorRow = 0; majorRow < 3; majorRow++) {
            for (int minorRow = 0; minorRow < 3; minorRow++) {
                for (int majorCol = 0; majorCol < 3; majorCol++) {
                    for (int minorCol = 0; minorCol < 3; minorCol++) {
                        builder.append("|").append(" ").append(format.format(MCTSProbabilites[getIndex(majorRow, majorCol, minorRow, minorCol)])).append(" ");
                    }
                    builder.append("|");

                }
                builder.append("\n");
            }
            if (majorRow != 3 - 1) {
                builder.append("|========+========+========||========+========+========||========+========+========|\n");
            }else {
                builder.append("\n");
            }
        }
        return builder.toString();
    }
}
