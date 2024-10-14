package com.hottes.caleb.ultimateticktacktoe;


import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameAction;
import com.hottes.caleb.ultimateticktacktoe.gameindependant.GameState;
import com.hottes.caleb.ultimateticktacktoe.ui.UltimateTickTacToeGameAction;

import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.io.PrintStream;
import java.util.BitSet;

import static com.hottes.caleb.ultimateticktacktoe.Resources.LINE_STROKE;
import static com.hottes.caleb.ultimateticktacktoe.Resources.drawToken;

public class SubBoardState extends GameState {

    public final Rectangle2D.Double[][] subBoardBoundingBoxes;
    private final int boardSize;
    private boolean active;
    /**
     * creates a new minor board
     *
     * @param theBoardSize the board will be square and the board size is the size of the major and minor boards
     */
    public SubBoardState(int theBoardSize, boolean isActive) {
        super(theBoardSize, theBoardSize);
        this.boardSize = theBoardSize;
        this.active = isActive;
        this.subBoardBoundingBoxes = new Rectangle2D.Double[boardSize][boardSize];
    }

    public SubBoardState(int theBoardSize) {
        this(theBoardSize, true);
    }

    public SubBoardState(double[][] theState, boolean isActive, int theBoardSize) {
        super(theState);
        this.active = isActive;
        this.boardSize = theBoardSize;
        this.subBoardBoundingBoxes = new Rectangle2D.Double[boardSize][boardSize];
    }

    /**
     * assumes all input data is valid
     *
     * @param hash
     */
    public SubBoardState(String hash) {
        super(hash);
        this.boardSize = getRows();
        this.subBoardBoundingBoxes = new Rectangle2D.Double[boardSize][boardSize];


    }

    /**
     * Evaluations are 1 for player 1 wins, -1 for player 1 loses, 0 for intederminate state and -.25 for draw
     * Copy past from copilot LLM
     *
     * @return the static evaluation of the game state
     */
    @Override
    public double getEvaluation() {
        return BoardState.getTicTacToeEvaluationBruteForce(state);
    }


    @Override
    public BitSet getBitSet() {
        throw new UnsupportedOperationException();
    }

    /**
     * takes in a graphics context and a bouding box saying where its allowed to draw and draws the current state of this sub board in that bounding box
     *
     * @param g           The graphics2D context to be used
     * @param boundingBox The bounding box in which board needs to be drawn.
     */
    public void render(Graphics2D g, Rectangle2D.Double boundingBox) {

        //if active draw a square to declare that it is active
        if (this.active) {
            g.setPaint(Resources.ACTIVE_BOARD_COLOR);
            g.fillRect((int) boundingBox.x, (int) boundingBox.y, (int) boundingBox.width, (int) boundingBox.height);
        }

        //calculate layout constants
        double width = boundingBox.width;
        double height = boundingBox.height;
        double horizontalSpacing = width * Resources.HORIZONTAL_SPACING_FACTOR;
        double verticalSpacing = height * Resources.VERTICAL_SPACING_FACTOR;

        double squareWidth = (width - (boardSize + 1) * horizontalSpacing) / boardSize;
        double squareHeight = (height - (boardSize + 1) * verticalSpacing) / boardSize;


        //make this method render and X or O on a board if it is won. If it is not won than do not draw an X or O but draw the grid lines and board state.
        // Make sure that the small square bounding boxes are always drawn as they determine where input is received from.
        double eval = getEvaluation();
        boolean renderPlayerTokens = !drawToken(g, boundingBox, eval);//if a big X or O was drawn then somebody won

        //calculate the sub board bounding boxes (for rendereing and input detection) and render the grid lines and player tokens if needed.
        for (int i = 0; i < boardSize; i++) {
            if (renderPlayerTokens && (i < boardSize - 1)) {
                //draw the grid lines, this saves on the number of loops I need
                g.setPaint(Color.BLACK);
                //vertical lines
                g.fillRect((int) (((i + 1) * (horizontalSpacing + squareWidth)) + boundingBox.x), (int) boundingBox.y, LINE_STROKE, (int) boundingBox.height);
                //horizontal lines
                g.fillRect((int) boundingBox.x, (int) (((i + 1) * (verticalSpacing + squareHeight)) + boundingBox.y), (int) boundingBox.width, LINE_STROKE);
            }
            for (int j = 0; j < boardSize; j++) {
                //for each sub board square
                //have to offset these calculations to put them in absolute coords
                Rectangle2D.Double tickTacToeSquareBoudingBox = new Rectangle2D.Double(boundingBox.x + (horizontalSpacing * (i + 1)) + (squareWidth * i), boundingBox.y + (verticalSpacing * (j + 1)) + (squareHeight * j), squareWidth, squareHeight);
                subBoardBoundingBoxes[i][j] = tickTacToeSquareBoudingBox;

                if (renderPlayerTokens) {
                    drawToken(g, tickTacToeSquareBoudingBox, state[j][i]);
                }
            }
        }


    }//end render

    @Override
    public GameState simulateAction(GameAction action, boolean invertBoard) {
        SubBoardState newState = new SubBoardState(this.getStateAsCopy(), this.isActive(), this.boardSize);//have to use this.getState() in order for this method to return a true copy.
        newState.preformAction(action);
        if (invertBoard) {
            newState.invertState();
        }
        return newState;

    }

    @Override
    public boolean isMoveLegal(GameAction action) {
        if (!(action instanceof UltimateTickTacToeGameAction)) {
            return false;
        }
        //so this is a valid move object
        return this.state[action.y][action.x] == 0;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean isActive) {
        this.active = isActive;
    }

    public enum TicTacToeResult {
        PLAYER1_WINS,
        PLAYER_NEG1_WINS,
        DRAW,
        IN_PROGRESS
    }


    //Loopuptable creation code
    //        JSONObject lookupTable = new JSONObject();
//        List<double[][]> rawPermutations = new ArrayList<>();
//        generateMatrixPermutations(new double[3][3], 0, 0, rawPermutations);
//        List<SubBoardState> ticTacToeStates = new ArrayList<>();
//        rawPermutations.forEach(doubles -> ticTacToeStates.add(new SubBoardState(doubles, false, 3)));
    //ticTacToeStates.forEach(subBoardState -> lookupTable.append(subBoardState.getSHAHash(),evaluateBoard(subBoardState).name()));
//        try {
//            FileWriter writer = new FileWriter("lookuptable.json");
//            writer.write(lookupTable.toString(4));
//            writer.flush();
//            writer.close();
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }


}
